from pathlib import Path
import json
import numpy as np
from PIL import Image


def load_inputs(rgb, depth, mask, K):
    rgb = np.asarray(Image.open(rgb).convert('RGB'))
    depth = load_depth(depth)
    mask = np.asarray(Image.open(mask)) if mask else None
    if isinstance(K, (str, Path)):
        K = np.loadtxt(K) if Path(K).suffix.lower() != '.json' else np.array(json.loads(Path(K).read_text()))
    return rgb, depth, K, mask


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+'\n')


def write_ply(path, points):
    points = np.asarray(points)
    with Path(path).open('w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f, points, fmt='%.8f')


def save_prediction(result, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'pose.json', {k:v for k,v in result.items() if k != 'points_cam_m'})
    if 'points_cam_m' in result: write_ply(output/'shape_camera.ply', result['points_cam_m'])
    return output


def load_depth(depth):
    dp = Path(depth)
    if dp.suffix.lower() == '.npy':
        depth = np.load(dp, allow_pickle=False)
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError('NPY depth must contain floating-point raw depth values.')
    elif dp.suffix.lower() == '.png':
        depth = np.asarray(Image.open(dp))
        if depth.dtype not in (np.uint16, np.int32) or depth.ndim != 2:
            raise ValueError('PNG depth must be a raw 16-bit grayscale image, not a visualization.')
    else:
        raise ValueError('Depth format must be uint16 PNG or floating-point NPY.')
    return depth
