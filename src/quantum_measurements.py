"""Measurement helpers for compatibility with the project structure."""

from __future__ import annotations

import numpy as np


def probabilities_sum_to_one(probs: np.ndarray, atol: float = 1e-5) -> bool:
    """Return whether class probabilities are normalized row-wise."""
    return bool(np.allclose(np.asarray(probs).sum(axis=1), 1.0, atol=atol))

