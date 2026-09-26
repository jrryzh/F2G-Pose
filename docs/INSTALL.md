# Installation

[← README](../README.md) · [Usage](USAGE.md)

You need Linux, an NVIDIA GPU, Python 3.10, and the CUDA 12.4 development toolkit. The tested environment uses PyTorch 2.4.0, torchvision 0.19.0, and cutoop 0.1.0.

## 1. Create the environment

Run these commands from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock-cu124.txt
```

The lock file contains the tested dependencies, including Gradio and SAM. `requirements-cu124.txt` lists the main dependencies if you maintain your own environment.

## 2. Build the CUDA extensions

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST='8.9'  # RTX 4090; use '9.0' for H200
export MAX_JOBS=4

python -m pip install --no-build-isolation --no-deps \
  ./third_party/pointnet2_ops ./third_party/chamfer
python -m pip install -e '.[demo,test]'
```

Set `CUDA_HOME` to your toolkit location and choose the architecture for your GPU. Rebuild both extensions after changing PyTorch, CUDA, or GPU architecture; remove their generated `build/` directories first.

To check the installation:

```bash
python -m pytest tests -q
python -m pip check
f2g-pose --help
```

## 3. Set up RADIO

F2G-Pose uses **RADIO v2.5-L**, with features from layers 8, 16, and 23. Install this pinned source revision:

```bash
git clone https://github.com/NVlabs/RADIO.git third_party_local/RADIO
git -C third_party_local/RADIO checkout fbd19ec1e68483482d7d96a59cb639880a8b33ed

export F2G_RADIO_REPOSITORY="$PWD/third_party_local/RADIO"
export F2G_RADIO_CHECKPOINT="$PWD/weights/radio-v2.5-l_half.pth.tar"

python scripts/download_weights.py \
  --url 'https://huggingface.co/nvidia/RADIO/resolve/main/radio-v2.5-l_half.pth.tar?download=true' \
  --sha256 50324e8cb086885126a896ad9ecbac46355cf0e4c1d8363763b01693a4618ff1 \
  --output weights/radio-v2.5-l_half.pth.tar
```

Set the two `F2G_RADIO_*` variables in each new shell, or pass `--radio-repository` and `--radio-checkpoint` to the CLI. RADIO has its own [NVIDIA license and model card](https://huggingface.co/nvidia/RADIO).

## 4. Add the F2G-Pose weights

Download the [F2G-Pose checkpoint](https://huggingface.co/J3rr1/F2G-Pose) and verify its SHA-256:

```bash
python scripts/download_weights.py \
  --url 'https://huggingface.co/J3rr1/F2G-Pose/resolve/main/f2g-pose.pt?download=true' \
  --sha256 bff06bd58a42fd805d8232f9ab7b4612b8eb833edf6b6cccd23e5040a8c9ca18 \
  --output weights/f2g-pose.pt
```

If you downloaded the complete weight package, run `sha256sum -c SHA256SUMS` from that directory before copying `f2g-pose.pt` to `weights/`.

The model configuration is included in the checkpoint. You are ready to [launch the demo or run inference](USAGE.md).

## Optional: SAM object selection

To select an object with clicks or a box instead of uploading a mask, download the official [SAM ViT-B](https://github.com/facebookresearch/segment-anything) checkpoint:

```bash
python scripts/download_weights.py \
  --url https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  --sha256 ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912 \
  --output weights/sam_vit_b_01ec64.pth
```

Pass `--sam-checkpoint weights/sam_vit_b_01ec64.pth` when starting the demo.
