# Validation

[← README](../README.md) · [Evaluation results](EVALUATION.md)

The following checks have passed for the release code and selected model:

| Area | Check |
| --- | --- |
| Installation | Clean Python 3.10 / PyTorch 2.4.0 + CUDA 12.4 environment, both CUDA extensions, wheel install, dependency check, and example inference |
| Tests | 15 tests covering preprocessing, depth units, coordinate restoration, metrics, evaluation merging, and demo isolation |
| Model compatibility | Four real inputs: pre-cleanup code, renamed code, original weight package, and updated weight package agree exactly in FP32 and BF16; all 477 weight tensors are identical |
| Pose-only mode | Pose and size match the full model on the tested inputs |
| Gradients | RADIO remains frozen; feature fusion receives finite, nonzero gradients |
| Training | Real SOPE, batch size 2, 16 samples, one epoch limit and zero workers: random and release-weight starts each completed two updates, checkpoint save, and resume to update three; losses and gradients stayed finite, fusion weights changed, RADIO stayed frozen, and both resumed checkpoints produced inference outputs |
| Preprocessing | 10,000 crop windows, 540 mask/RNG cases, and 80 real train/eval inputs match the earlier implementation exactly |
| Demo | Uploaded masks, SAM preview/confirmation, three example sets, separate sessions, and JSON/PLY downloads |
| Bundled example | The ROPE teapot's CLI prediction, single-object evaluation, and actual Gradio upload produced JSON/PLY outputs; custom-manifest evaluation reports `is_full_test=false` |
| Additional environment | CUDA 12.8 / PyTorch 2.8 extension and inference checks on RTX 4090 |

Training checks are short integration runs; the 100-epoch configuration is the full training recipe. The complete metric evaluations preceded the equivalent crop-helper replacement. Docker container execution and hosted ZeroGPU remain untested.

To run the automated tests:

```bash
python -m pytest tests -q
```
