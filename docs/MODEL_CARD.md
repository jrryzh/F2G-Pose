# Model card

[← README](../README.md) · [Evaluation](EVALUATION.md)

F2G-Pose estimates category-level object pose, metric size, and completed shape from a segmented RGB-D observation. It is trained on synthetic SOPE data and evaluated on SOPE and ROPE.

## At a glance

| Item            | Value                                                                                  |
| --------------- | -------------------------------------------------------------------------------------- |
| RGB backbone    | Frozen RADIO v2.5-L                                                                    |
| RGB crop        | 336 × 336 pixels                                                                       |
| Observed points | 1,024                                                                                  |
| Completed shape | 2,048 points in camera coordinates                                                     |
| Geometry model  | DGCNN tokenization and a Mixture-of-Experts Transformer                                |
| Outputs         | Object-to-camera rotation and translation, object dimensions, optional completed shape |
| Runtime         | Linux with an NVIDIA GPU and CUDA extensions                                           |

Inputs are aligned RGB, depth, a foreground mask, and camera intrinsics. No category label, instance CAD model, or reference view is required at inference. Translation, dimensions, and completed points are returned in meters. See [Usage](USAGE.md#input-formats) for the coordinate convention.

## Weights

The [F2G-Pose checkpoint](https://huggingface.co/J3rr1/F2G-Pose/resolve/main/f2g-pose.pt?download=true) contains the model configuration and F2G inference parameters. RADIO is loaded separately. The checkpoint's SHA-256 is `bff06bd58a42fd805d8232f9ab7b4612b8eb833edf6b6cccd23e5040a8c9ca18`. The public model repository also includes evaluation results and checksums.

The model was selected from archived checkpoints and verified on both complete test snapshots. [Evaluation](EVALUATION.md) lists the metrics; [Validation](VALIDATION.md) summarizes code checks.

## Practical considerations

Depth and mask quality matter: reflective or transparent surfaces, missing depth, severe occlusion, and unfamiliar objects can reduce accuracy. Symmetric objects can admit several plausible rotations. Shape completion predicts unobserved geometry, so applications should account for uncertainty when using it for physical interaction.

The local NVIDIA GPU path is tested. Docker and hosted ZeroGPU deployment are described in [Deployment](DEPLOYMENT.md), with their current validation status.

## License

F2G-Pose integration code uses Apache-2.0. RADIO and other third-party components retain their original terms; see [THIRD_PARTY.md](../THIRD_PARTY.md).
