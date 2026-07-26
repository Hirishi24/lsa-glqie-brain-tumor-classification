"""Tumour-centred local crop utilities."""

from __future__ import annotations

import numpy as np

from .preprocessing import resize_array


def tumor_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return inclusive-exclusive bounding box (y0, y1, x0, x1) for positive mask pixels."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("Cannot compute tumour bbox for empty mask")
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def expand_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], margin: float, shift: tuple[int, int] = (0, 0)) -> tuple[int, int, int, int]:
    """Expand and clip a bounding box."""
    y0, y1, x0, x1 = bbox
    h, w = y1 - y0, x1 - x0
    pad_y = max(1, int(round(h * margin)))
    pad_x = max(1, int(round(w * margin)))
    dy, dx = shift
    ny0 = max(0, y0 - pad_y + dy)
    ny1 = min(shape[0], y1 + pad_y + dy)
    nx0 = max(0, x0 - pad_x + dx)
    nx1 = min(shape[1], x1 + pad_x + dx)
    if ny1 <= ny0:
        ny1 = min(shape[0], ny0 + 1)
    if nx1 <= nx0:
        nx1 = min(shape[1], nx0 + 1)
    return ny0, ny1, nx0, nx1


def local_crop(image: np.ndarray, mask: np.ndarray, size: int, margin: float, shift: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Extract tumour-centred crop and resize it."""
    bbox = expand_bbox(tumor_bbox(mask), image.shape, margin, shift)
    y0, y1, x0, x1 = bbox
    crop = image[y0:y1, x0:x1]
    return resize_array(crop, size, order=1).astype(np.float32)

