# ROPE teapot example

This is one real Omni6DPose ROPE evaluation object: frame `000157/000800_`, object `real-teapot_002`, instance-mask ID `1`. It is a single-sample example, not a full-test result.

## Predict

After installing F2G-Pose and its separate RADIO backbone, run from the repository root:

```bash
f2g-pose predict --checkpoint weights/f2g-pose.pt \
  --rgb examples/rope/rgb.png --depth examples/rope/depth.npy \
  --mask examples/rope/mask.png --intrinsics examples/rope/K.txt \
  --depth-scale 1 --output outputs/rope-teapot
```

`depth.npy` contains `float32` **meters**. `mask.png` contains only target ID 1, encoded as 0/255. `K.txt` is a 3×3 matrix for the supplied RGB image. The output directory contains `pose.json` and `shape_camera.ply`.

For the Gradio demo, upload `rgb.png`, `depth.npy`, and `mask.png`; select **Meters** and enter the four values from `K.txt` (`fx`, `fy`, `cx`, `cy`).

## Evaluate this object

The `raw/` directory has the original frame and the ROPE object metadata required by the existing evaluator. Run:

```bash
f2g-pose eval --checkpoint weights/f2g-pose.pt \
  --data-root examples/rope/raw/ROPE --object-meta examples/rope/raw/Meta/real_obj_meta.json \
  --split rope --protocol full --manifest examples/rope/sample_manifest.jsonl \
  --batch-size 1 --workers 0 --metric-workers 1 --output outputs/rope-teapot-eval
```

The custom manifest contains exactly one object. Its `metrics.json` must report `is_full_test: false`, even though `--protocol full` selects the full metric protocol.

## Source and conversion

The source is [Omni6DPose ROPE](https://github.com/Omni6DPose/Omni6DPoseAPI#-omni6dpose-dataset), sequence `000157`, frame `000800_`, together with `Meta/real_obj_meta.json`. The bundled `raw/ROPE/000157/` files retain the dataset names and bytes. Run `python examples/rope/prepare.py` to regenerate the upload files: the dataset API loads aligned RGB and metric depth; the script selects mask value 1, scales the frame intrinsics to the RGB resolution, saves `float32` depth in meters, and writes the one-object manifest. Check all raw and converted file hashes with `cd examples/rope && sha256sum -c SHA256SUMS`.

The [Omni6DPose API repository](https://github.com/Omni6DPose/Omni6DPoseAPI) labels its code MIT. Its [dataset page](https://jiyao06.github.io/Omni6DPose/download/) and API README identify ROPE as a separately downloaded dataset; neither states a separate license grant for redistributing dataset frames. The sample retains upstream attribution, and dataset assets should be handled under the upstream dataset terms. The API's MIT notice does not automatically cover these images, depth maps, masks, or annotations.
