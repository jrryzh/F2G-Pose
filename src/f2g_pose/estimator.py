"""Public inference API; no category, CAD, pose or size annotations are inputs."""
from pathlib import Path
import threading
import numpy as np
from .preprocess import prepare


class PoseEstimator:
    def __init__(self, checkpoint, *, config=None, radio_repository=None,
                 radio_checkpoint=None, device='cuda:0', seed=0):
        import torch
        import yaml
        from easydict import EasyDict
        from .model import F2GPose
        self.device, self.seed = torch.device(device), int(seed)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('F2G-Pose requires an NVIDIA GPU and compiled PointNet++ CUDA extension.')
        torch.cuda.set_device(self.device)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if config is None:
            config = payload.get('config')
            if config is None:
                raise ValueError('A historical checkpoint requires --config; exported weights embed their config.')
        elif isinstance(config, (str, Path)):
            config = yaml.safe_load(Path(config).read_text())
        self.config = config
        model_config = EasyDict(config['model'])
        if radio_repository: model_config.radio_repository = str(radio_repository)
        if radio_checkpoint: model_config.radio_checkpoint = str(radio_checkpoint)
        self.model = F2GPose(model_config)
        state = payload.get('state_dict', payload.get('base_model', payload))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        result = self.model.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith('base_model.radio_encoder.')]
        if missing or result.unexpected_keys:
            raise ValueError(f'Incompatible checkpoint: missing={missing}; unexpected={result.unexpected_keys}')
        self.model.to(self.device).eval()
        self.model.base_model.radio_encoder.requires_grad_(False)
        self.lock = threading.RLock()
        self.checkpoint = Path(checkpoint)

    def preprocess(self, rgb, depth, K, mask, depth_scale):
        return prepare(rgb, depth, K, mask, depth_scale, seed=self.seed,
                       image_size=self.config.get('image_size', 336))

    def forward(self, observation, return_shape=True):
        import torch
        with self.lock, torch.cuda.device(self.device), torch.inference_mode():
            self.model.eval()
            self.model.base_model.radio_encoder.eval()
            ret = self.model(*observation.tensors(self.device), return_shape=return_shape)
            return self.decode(ret, observation.center)

    @staticmethod
    def decode(ret, center):
        from .geometry import compute_rotation_matrix_from_ortho6d
        center = np.asarray(center, dtype=np.float32)
        R = compute_rotation_matrix_from_ortho6d(ret[2]).detach().cpu().numpy()[0]
        t = ret[3].detach().cpu().numpy()[0] + center
        size = ret[4].detach().cpu().numpy()[0]
        if not all(np.isfinite(x).all() for x in (R, t, size)):
            raise RuntimeError('Model produced nonfinite pose or dimensions.')
        T = np.eye(4, dtype=np.float32); T[:3, :3] = R; T[:3, 3] = t
        result = dict(T_cam_obj=T, rotation=R, translation_m=t, size_m=size,
                      coordinate_frame='camera: x right, y down, z forward',
                      shape_frame='camera', observed_center_m=center)
        if ret[1] is not None:
            result['points_cam_m'] = ret[1].detach().cpu().numpy()[0] + center
        return result

    def predict(self, rgb, depth, K, mask, depth_scale, return_shape=True):
        return self.forward(self.preprocess(rgb, depth, K, mask, depth_scale), return_shape)
