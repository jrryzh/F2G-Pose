# Evaluation

[← README](../README.md) · [Run evaluation](USAGE.md#evaluation)

## Test results

These results evaluate one F2G-Pose checkpoint on the full SOPE and ROPE test sets. Every test frame and metadata-valid object is included, with invalid inputs recorded in the exclusion manifests.

| Dataset | AUC25 ↑ | AUC50 ↑ | AUC75 ↑ | VUS 5°2cm ↑ | VUS 5°5cm ↑ | VUS 10°2cm ↑ | VUS 10°5cm ↑ | Rotation ↓ | Translation ↓ |
| ------- | ------- | ------- | ------- | ----------- | ----------- | ------------ | ------------ | ---------- | ------------- |
| SOPE    | 60.2436 | 44.7312 | 17.6862 | 21.4113     | 25.1540     | 36.2412      | 43.2072      | 14.734926° | 0.823538 cm   |
| ROPE    | 46.7463 | 28.1905 | 6.1675  | 11.8337     | 16.5743     | 22.4612      | 31.8768      | 26.987203° | 1.181104 cm   |

AUC and VUS are percentages; higher is better. Rotation and translation are mean errors; lower is better. The model package includes unrounded values and per-category results. The table reports the final full-test evaluation.

## Reproduction settings

| Setting           | Value                                                          |
| ----------------- | -------------------------------------------------------------- |
| Hardware          | NVIDIA H200                                                    |
| Software          | Python 3.10, PyTorch 2.4.0 / CUDA 12.4, cutoop 0.1.0           |
| Precision         | BF16 autocast; FP32 pose-head decoding                         |
| Batch / seed      | 128 / 0                                                        |
| TF32              | CUDA matmul off; cuDNN on                                      |
| Shards            | 3 contiguous SOPE shards; 5 contiguous ROPE shards             |
| Workers per shard | 4 data-loader workers, 4 metric workers; CPU library threads 1 |
| Model mode        | Pose and size, with shape completion skipped                   |

[Usage](USAGE.md#evaluation) provides full, subset, and sharded commands. Full and pose-only pose heads were exactly equal on the tested real inputs in FP32 and BF16.

Set `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` for the reported configuration and inspect `environment.backend_flags` in `run.json`. In embedded Python, also set `torch.backends.cuda.matmul.allow_tf32 = False`. GPU architecture, precision, and batch composition can affect floating-point results.
