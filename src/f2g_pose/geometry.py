import torch
import numpy as np

def compute_rotation_matrix_from_ortho6d(ortho6d):
    """Convert a 6D orthogonal representation to a rotation matrix."""
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]
    x = torch.nn.functional.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = torch.nn.functional.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    matrix = torch.stack([x, y, z], dim=-1)
    return matrix

def single_rotation_matrix_to_ortho6d(R):
    """Convert one 3x3 rotation matrix to its 6D representation."""
    if isinstance(R, np.ndarray):
        return np.concatenate((R[:, 0], R[:, 1]))
    elif isinstance(R, torch.Tensor):
        return torch.cat((R[:, 0], R[:, 1]))
    else:
        raise TypeError('Input type is not supported')

def single_rotation_matrix_from_ortho6d(ortho6d):
    """Convert one 6D representation to a 3x3 rotation matrix."""
    if isinstance(ortho6d, np.ndarray):
        a1 = ortho6d[:3]
        a2 = ortho6d[3:6]
        a1_norm = np.linalg.norm(a1)
        if a1_norm == 0:
            raise ValueError('Zero vector encountered in a1 normalization.')
        a1 = a1 / a1_norm
        dot_product = np.dot(a1, a2)
        a2 = a2 - dot_product * a1
        a2_norm = np.linalg.norm(a2)
        if a2_norm == 0:
            raise ValueError('Zero vector encountered in a2 normalization.')
        a2 = a2 / a2_norm
        a3 = np.cross(a1, a2)
        rotation_matrix = np.stack((a1, a2, a3), axis=1)
        return rotation_matrix
    elif isinstance(ortho6d, torch.Tensor):
        a1 = ortho6d[:3]
        a2 = ortho6d[3:6]
        a1_norm = torch.norm(a1)
        if a1_norm == 0:
            raise ValueError('Zero vector encountered in a1 normalization.')
        a1 = a1 / a1_norm
        dot_product = torch.dot(a1, a2)
        a2 = a2 - dot_product * a1
        a2_norm = torch.norm(a2)
        if a2_norm == 0:
            raise ValueError('Zero vector encountered in a2 normalization.')
        a2 = a2 / a2_norm
        a3 = torch.cross(a1, a2, dim=-1)
        rotation_matrix = torch.stack((a1, a2, a3), dim=1)
        return rotation_matrix
    else:
        raise TypeError('Input type is not supported')

def fps(points, number):
    from pointnet2_ops import pointnet2_utils
    indices = pointnet2_utils.furthest_point_sample(points.contiguous(), number)
    return pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()
