"""Aligned RGB-D preprocessing matching the historical 336-pixel pipeline."""
from dataclasses import dataclass
import numpy as np
import cv2
cv2.setNumThreads(1)
from scipy.spatial import KDTree
from .crop import defor_2D, get_2d_coord_np, get_bbox, aug_bbox_eval, aug_bbox_DZI, crop_resize_by_warp_affine


@dataclass
class Observation:
    points: np.ndarray
    rgb: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    center: np.ndarray
    valid_depth_pixels: int

    def tensors(self, device):
        import torch
        return tuple(torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).to(device)
                     for a in (self.points, self.rgb, self.rows, self.columns))


def intrinsics_matrix(K):
    K = np.asarray(K, dtype=np.float32)
    if K.shape == (4,):
        fx, fy, cx, cy = K
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or
        K[0, 0] <= 0 or K[1, 1] <= 0 or
        not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-6 or abs(K[1, 0]) > 1e-6):
        raise ValueError('K must be finite [fx, fy, cx, cy] or a pinhole 3×3 matrix with positive focal lengths and zero skew.')
    return K


def validate(rgb, depth, K, mask, depth_scale):
    rgb, depth, mask = np.asarray(rgb), np.asarray(depth), np.asarray(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('RGB must be an H×W×3 uint8 image in RGB channel order.')
    if depth.ndim != 2 or mask.ndim != 2 or rgb.shape[:2] != depth.shape or depth.shape != mask.shape:
        raise ValueError('RGB, depth and binary mask must be aligned and have identical H×W dimensions.')
    if not np.issubdtype(depth.dtype, np.number):
        raise ValueError('Depth must contain numeric raw depth values.')
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError('Mask must be binary (0/1 or 0/255).')
    mask = mask > 0
    if not mask.any():
        raise ValueError('Mask is empty; select one target object.')
    if not np.isscalar(depth_scale) or not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError('depth_scale must be a positive number of meters per raw depth unit, e.g. 0.001 for millimeters.')
    depth = depth.astype(np.float32) * depth_scale
    valid = np.isfinite(depth) & (depth > 0) & (depth <= 1000)
    if (valid & mask).sum() < 50:
        raise ValueError('Fewer than 50 valid depth pixels inside the target mask.')
    depth = np.where(valid, depth, 0).astype(np.float32)
    return rgb, depth, intrinsics_matrix(K), mask


def sample_indices(length, count, rng):
    return (np.tile(np.arange(length), (count + length - 1) // length)[:count]
            if length < count else rng.permutation(length)[:count])


def prepare(rgb, depth, K, mask, depth_scale, *, seed=0, rng=None,
            image_size=336, n_points=1024, remove_outliers=True, augment=False):
    rgb, depth, K, mask = validate(rgb, depth, K, mask, depth_scale)
    rng = np.random.default_rng(seed) if rng is None else rng
    h, w = depth.shape
    ys, xs = np.where(mask)
    rmin, rmax, cmin, cmax = get_bbox([ys.min(), xs.min(), ys.max(), xs.max()], h, w)
    bbox = np.array([cmin, rmin, cmax, rmax])
    if augment:
        # Original DZI uses NumPy's global RNG, seeded by the training worker.
        center, scale = aug_bbox_DZI(dict(DZI_TYPE='uniform', DZI_SCALE_RATIO=.25,
                                         DZI_SHIFT_RATIO=.25, DZI_PAD_SCALE=1.5), bbox, h, w)
    else:
        center, scale = aug_bbox_eval(bbox, h, w)
    def crop(a, interpolation):
        return crop_resize_by_warp_affine(a, center, scale, 224, interpolation=interpolation)
    size = (image_size, image_size)
    coords = crop(get_2d_coord_np(w, h).transpose(1, 2, 0), cv2.INTER_NEAREST)
    coords = (cv2.resize(coords, size, interpolation=cv2.INTER_LINEAR) * (image_size / 224)).astype(int)
    cropped_rgb = cv2.resize(crop(rgb, cv2.INTER_LINEAR), size, interpolation=cv2.INTER_CUBIC)
    cropped_depth = cv2.resize(crop(depth, cv2.INTER_NEAREST), size, interpolation=cv2.INTER_NEAREST)
    cropped_mask = cv2.resize(crop(mask.astype(np.float32), cv2.INTER_NEAREST), size, interpolation=cv2.INTER_NEAREST)
    if augment:
        cropped_mask = defor_2D(cropped_mask, rand_r=3, rand_pro=.5)
    rows, cols = np.where((cropped_depth > 0) & (cropped_mask > 0))
    if len(rows) < 50:
        raise ValueError('Fewer than 50 valid depth pixels after the model crop.')
    z = cropped_depth[rows, cols]
    scaled_K = K.copy(); scaled_K[:2] *= image_size / 224
    xy = coords[rows, cols]
    points = np.stack(((xy[:, 0] - scaled_K[0, 2]) * z / scaled_K[0, 0],
                       (xy[:, 1] - scaled_K[1, 2]) * z / scaled_K[1, 1], z), -1).astype(np.float32)
    if remove_outliers:
        ids = sample_indices(len(points), int(1.5 * n_points), rng)
        points, rows, cols = points[ids], rows[ids], cols[ids]
        distances, _ = KDTree(points).query(points, k=51)
        distances = distances[:, 1:].mean(1)
        keep = distances <= distances.mean() + distances.std()
        points, rows, cols = points[keep], rows[keep], cols[keep]
        if len(points) < 50:
            raise ValueError('Fewer than 50 points remain after outlier removal.')
    ids = sample_indices(len(points), n_points, rng)
    points, rows, cols = points[ids], rows[ids], cols[ids]
    center = np.median(points, axis=0)
    return Observation((points-center).astype(np.float32),
                       (cropped_rgb.transpose(2, 0, 1) / 255).astype(np.float32),
                       rows.astype(np.int64), cols.astype(np.int64), center, len(z))
