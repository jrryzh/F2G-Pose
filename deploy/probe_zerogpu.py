"""Run inside a separate deployment environment after building CUDA extensions."""
import json,os
import torch
from pointnet2_ops import pointnet2_utils
x=torch.randn(2,1024,3,device='cuda',requires_grad=True)
indices=pointnet2_utils.furthest_point_sample(x,128)
y=pointnet2_utils.gather_operation(x.transpose(1,2).contiguous(),indices)
y.square().mean().backward();torch.cuda.synchronize()
print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),pointnet_forward_backward='pass')))
# A hosted ZeroGPU test still requires a Space and @spaces.GPU call boundary.
