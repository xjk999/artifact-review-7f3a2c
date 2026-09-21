"""Keep the source NIfTI grid. Tiling is voxel-preserving, not ROI resizing."""
from itertools import product

import nibabel as nib
import numpy as np


def load_volume(path):
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI: {path}, shape={image.shape}")
    array = image.get_fdata(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite image values: {path}")
    return image, array


def require_same_grid(reference, other, name):
    if reference.shape != other.shape or not np.allclose(reference.affine, other.affine, atol=1e-4):
        raise ValueError(f"{name} is not registered to NCCT. Register/resample paired data first.")


def save_volume(array, reference, path, dtype=np.float32):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    header.set_slope_inter(1, 0)
    output = nib.Nifti1Image(np.asarray(array, dtype=dtype), reference.affine, header)
    output.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    output.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(output, str(path))


def normalize_hu(array, limits):
    # Per-volume clipping followed by min/max normalization.
    clipped = np.clip(array, *limits)
    low, high = float(clipped.min()), float(clipped.max())
    if high <= low:
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - low) / (high - low)).astype(np.float32)


def tile_origins(mask, size, margin=16, overlap=0.25):
    foreground = np.where(mask > 0)
    if not len(foreground[0]):
        raise ValueError("SegResNet produced an empty aorta mask")
    lo = [max(0, int(axis.min()) - margin) for axis in foreground]
    hi = [min(n, int(axis.max()) + margin + 1) for axis, n in zip(foreground, mask.shape)]
    stride = max(1, int(size * (1 - overlap)))
    starts = []
    for a, b, n in zip(lo, hi, mask.shape):
        if b - a <= size:
            starts.append([max(0, min((a + b - size) // 2, max(0, n - size)))])
        else:
            values = list(range(a, b - size + 1, stride))
            if values[-1] != b - size:
                values.append(b - size)
            starts.append(values)
    return list(product(*starts))


def extract_tile(array, origin, size):
    result = np.zeros((size, size, size), dtype=array.dtype)
    shape = tuple(min(size, n - start) for n, start in zip(array.shape, origin))
    source = tuple(slice(start, start + count) for start, count in zip(origin, shape))
    dest = tuple(slice(0, count) for count in shape)
    result[dest] = array[source]
    return result


def add_tile(total, counts, tile, origin):
    shape = tuple(min(s, n - start) for s, n, start in zip(tile.shape, total.shape, origin))
    dest = tuple(slice(start, start + count) for start, count in zip(origin, shape))
    source = tuple(slice(0, count) for count in shape)
    total[dest] += tile[source]
    counts[dest] += 1


def xyz_to_tensor(array, device):
    import torch
    # SimpleITK volumes are Z,Y,X; NIfTI/nibabel arrays are X,Y,Z.
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 1, 0))).float()[None, None].to(device)


def tensor_to_xyz(tensor):
    return tensor.detach().float().cpu().numpy()[0, 0].transpose(2, 1, 0)
