"""DCT feature tests."""

from __future__ import annotations

import numpy as np

from src.dct_features import dct_coefficients, zigzag_indices


def test_global_dct_feature_length():
    coeff = dct_coefficients(np.eye(32), 16, 12, exclude_dc=True)
    assert coeff.shape == (12,)
    assert np.isfinite(coeff).all()


def test_zigzag_starts_at_dc():
    assert zigzag_indices(4, 4)[0] == (0, 0)

