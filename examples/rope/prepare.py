"""Regenerate the teapot upload files from the bundled ROPE frame."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import numpy as np
from PIL import Image
from f2g_pose.data import load_frame


root = Path(__file__).resolve().parent
prefix = root / 'raw/ROPE/000157/000800_'
rgb, depth, instance_mask, frame_meta = load_frame(str(prefix))
target = next(obj for obj in frame_meta.objects
              if str(obj.meta.oid) == 'real-teapot_002' and obj.mask_id == 1)
assert target.is_valid
intr = frame_meta.camera.intrinsics
scale = rgb.shape[0] / intr.height
assert np.isclose(rgb.shape[1] / intr.width, scale)
K = np.array([[intr.fx * scale, 0, intr.cx * scale],
              [0, intr.fy * scale, intr.cy * scale], [0, 0, 1]], dtype=np.float32)
binary_mask = (instance_mask == 1).astype(np.uint8) * 255
assert binary_mask.sum() > 50 * 255
Image.fromarray(rgb).save(root / 'rgb.png')
np.save(root / 'depth.npy', depth.astype(np.float32))
Image.fromarray(binary_mask).save(root / 'mask.png')
np.savetxt(root / 'K.txt', K)
record = dict(frame='000157/000800_', object_id='real-teapot_002',
              mask_id=1, class_label=130, class_name='teapot')
(root / 'sample_manifest.jsonl').write_text(json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n')
(root / 'input.json').write_text(json.dumps(dict(sample=record, depth_scale=1.0,
    depth_unit='meters', channel_order='RGB', mask_values=[0, 255],
    alignment='RGB, depth and instance mask from the same ROPE frame'), indent=2) + '\n')
paths = sorted(path for path in root.rglob('*') if path.is_file()
               and path.name not in ('SHA256SUMS', 'prepare.py', 'README.md'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'SHA256SUMS').write_text(''.join(f'{sha(path)}  {path.relative_to(root)}\n' for path in paths))
print(f'Prepared ROPE teapot example; foreground pixels: {int((binary_mask > 0).sum())}')
