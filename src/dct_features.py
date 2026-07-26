"""DCT feature extraction."""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn

from .preprocessing import resize_array


def zigzag_indices(height: int, width: int) -> list[tuple[int, int]]:
    """Return deterministic zigzag order from low to high spatial frequencies."""
    indices: list[tuple[int, int]] = []
    for s in range(height + width - 1):
        diag = [(i, s - i) for i in range(max(0, s - width + 1), min(height - 1, s) + 1)]
        if s % 2 == 0:
            diag.reverse()
        indices.extend(diag)
    return indices


def dct_coefficients(image: np.ndarray, size: int, n_coefficients: int, exclude_dc: bool = True) -> np.ndarray:
    """Resize image, apply orthonormal 2D DCT, and return zigzag coefficients."""
    resized = resize_array(image.astype(np.float32), size, order=1)
    coeff = dctn(resized, type=2, norm="ortho")
    order = zigzag_indices(coeff.shape[0], coeff.shape[1])
    values = []
    for y, x in order:
        if exclude_dc and y == 0 and x == 0:
            continue
        values.append(float(coeff[y, x]))
        if len(values) >= n_coefficients:
            break
    return np.asarray(values, dtype=np.float64)

