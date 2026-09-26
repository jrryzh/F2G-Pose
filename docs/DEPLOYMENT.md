# Deployment

[← README](../README.md) · [Installation](INSTALL.md) · [Local demo](USAGE.md#interactive-demo)

The local NVIDIA GPU demo is tested. The Dockerfile and ZeroGPU app are provided for deployment; container execution and hosted ZeroGPU still need validation on the target service.

## Docker

Place `f2g-pose.pt` and `radio-v2.5-l_half.pth.tar` in `weights/`, then build and run:

```bash
docker build --build-arg TORCH_CUDA_ARCH_LIST=8.9 -t f2g-pose:local .
docker run --rm --gpus all -p 127.0.0.1:7860:7860 \
  -v "$PWD/weights:/weights:ro" \
  -v "$PWD/demo_outputs:/outputs" f2g-pose:local
```

Open http://127.0.0.1:7860. Use architecture `8.9` for RTX 4090 or `9.0` for H200. The image includes the pinned RADIO source; weights are loaded from the mounted directory.

To enable SAM, add `sam_vit_b_01ec64.pth` to `weights/` and run:

```bash
docker run --rm --gpus all -p 127.0.0.1:7860:7860 \
  -v "$PWD/weights:/weights:ro" \
  -v "$PWD/demo_outputs:/outputs" f2g-pose:local \
  f2g-pose demo --checkpoint /weights/f2g-pose.pt \
  --sam-checkpoint /weights/sam_vit_b_01ec64.pth \
  --host 0.0.0.0 --port 7860 --output-root /outputs
```

## Hugging Face Docker Space

Use the same Dockerfile with the Docker SDK and app port **7860**. Provide an NVIDIA GPU, mount or download the weights, and configure `F2G_RADIO_CHECKPOINT` and the demo checkpoint path for your storage location.

## Hugging Face ZeroGPU

`deploy/zerogpu_app.py` uses the same Gradio interface and creates the model inside a `@spaces.GPU` callback. Use a separate environment from the PyTorch 2.4 reference installation:

```bash
python -m pip install -r deploy/zerogpu-requirements.txt
python -m pip install --no-build-isolation --no-deps \
  ./third_party/pointnet2_ops ./third_party/chamfer
python -m pip install --no-build-isolation --no-deps .
```

The deployment requirements pin PyTorch 2.8. Build the extensions with a CUDA toolkit and architecture compatible with the assigned GPU; the [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu) describes the hosted runtime.

Set these environment variables:

| Variable | Value |
| --- | --- |
| `F2G_CHECKPOINT` | Path to `f2g-pose.pt` |
| `F2G_RADIO_REPOSITORY` | Path to the pinned RADIO checkout |
| `F2G_RADIO_CHECKPOINT` | Path to the RADIO v2.5-L weight file |
| `F2G_SAM_CHECKPOINT` | Optional SAM ViT-B checkpoint |
| `F2G_OUTPUT_ROOT` | Optional output directory; default `demo_outputs` |

Copy the metadata from [deploy/README.space.md](../deploy/README.space.md) into the Space's root README. The app loads F2G-Pose per GPU allocation. `deploy/probe_zerogpu.py` provides a local compatibility check; PyTorch 2.8 / CUDA 12.8 checks have passed on RTX 4090, while the hosted allocation path remains untested.
