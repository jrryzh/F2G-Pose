# Third-party acknowledgments

F2G-Pose builds on the following projects. Integration code is Apache-2.0; included third-party code retains the licenses listed here.

| Component | Source | License / notice |
| --- | --- | --- |
| Geometry transformer and completion layers | [PoinTr / AdaPoinTr](https://github.com/yuxumin/PoinTr) | [MIT](third_party/POINTR_LICENSE); derived portions of `model.py` and `transformer.py` |
| PointNet++ CUDA operations | [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) | [Unlicense](third_party/pointnet2_ops/UNLICENSE) |
| Chamfer distance extension | Thibault Groueix / Haozhe Xie implementation distributed with PoinTr | Original author headers and [PoinTr MIT notice](third_party/POINTR_LICENSE) |
| Affine crops and DZI augmentation | [GDR-Net](https://github.com/THU-DA-6D-Pose-Group/GDR-Net/tree/1be9fe73292fd748087aa88d7bf987434f271ebb) and [CenterNet](https://github.com/xingyizhou/CenterNet/tree/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c) | [GDR-Net Apache-2.0](third_party/GDRNET_LICENSE), [CenterNet MIT](third_party/CENTERNET_LICENSE) |
| NOCS helper ancestry | [NOCS](https://github.com/hughw19/NOCS_CVPR2019) | [MIT notice](third_party/NOCS_LICENSE) |
| Dataset API and metrics | [Omni6DPoseAPI / cutoop 0.1.0](https://github.com/Omni6DPose/Omni6DPoseAPI) | [MIT](third_party/OMNI6DPOSE_LICENSE); dataset assets have their own terms |
| Image backbone | [NVIDIA RADIO v2.5-L](https://huggingface.co/nvidia/RADIO) | [NVIDIA Source Code License](third_party/RADIO_LICENSE); source and weights installed separately |
| Optional segmentation | [Segment Anything](https://github.com/facebookresearch/segment-anything) | Apache-2.0; checkpoint downloaded separately |

The research pipeline also draws on [SGPA](https://github.com/ck-kai/SGPA), [FS-Net](https://github.com/DC1991/FS_Net), and NOCS. The release's square-window and mask-boundary routines are newly implemented; affine/DZI adaptations use the licensed GDR-Net and CenterNet sources above.

GDR-Net helper sources are `core/utils/data_utils.py` and `core/base_data_loader.py`. Local changes adapt the configuration interface and use pixel coordinates. RADIO source is pinned to `fbd19ec1e68483482d7d96a59cb639880a8b33ed`; its parameters are separate from the F2G-Pose weight package.
