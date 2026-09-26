---
title: F2G-Pose
sdk: gradio
sdk_version: 5.49.1
python_version: '3.10'
app_file: deploy/zerogpu_app.py
---

# F2G-Pose demo

Estimate an object's 6D pose, size, and completed shape from RGB-D input.

1. Upload aligned RGB and raw depth.
2. Upload a mask, or select the object with SAM when enabled.
3. Enter the camera intrinsics and depth units, then click **Estimate pose**.
4. View the result and download JSON or PLY.

For setup, CUDA extensions, and weights, see [Deployment](../docs/DEPLOYMENT.md).
Copy the YAML header above into the Space's root README. Hosted ZeroGPU validation is pending.
