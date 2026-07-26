"""MRI preprocessing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def resize_array(arr: np.ndarray, size: int, order: int) -> np.ndarray:
    """Resize a 2D array to square size using scipy.ndimage.zoom."""
    if arr.shape == (size, size):
        return arr.copy()
    factors = (size / arr.shape[0], size / arr.shape[1])
    return zoom(arr, factors, order=order)


def normalize_image(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Clip nonzero intensities by percentiles and min-max scale to [0, 1]."""
    x = image.astype(np.float32, copy=False)
    nonzero = x[x != 0]
    values = nonzero if nonzero.size else x.ravel()
    lo, hi = np.percentile(values, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def preprocess_image_mask(image: np.ndarray, mask: np.ndarray, image_size: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalize image and resize image/mask while preserving alignment."""
    norm = normalize_image(image, low, high)
    image_resized = resize_array(norm, image_size, order=1).astype(np.float32)
    mask_resized = resize_array((mask > 0).astype(np.uint8), image_size, order=0)
    return image_resized, (mask_resized > 0).astype(np.uint8)

