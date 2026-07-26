"""Fixed-budget global-local coefficient allocation."""

from __future__ import annotations

import numpy as np

METHODS_PRIMARY = ["global_only", "local_only", "fixed_4g_4l", "random_allocation", "proposed_lsa_glqie", "classical_proposed"]
METHODS_ALL = METHODS_PRIMARY + ["fixed_2g_6l", "fixed_6g_2l", "shuffled_size_allocation", "lesion_size_only", "svm_classical_proposed"]


def allocation_for_method(method: str, category: str, rng: np.random.Generator | None = None) -> tuple[int, int]:
    """Return (global_count, local_count) for an eight-slot method."""
    if method == "global_only":
        return 8, 0
    if method == "local_only":
        return 0, 8
    if method == "fixed_2g_6l":
        return 2, 6
    if method in {"fixed_4g_4l", "classical_proposed", "svm_classical_proposed"}:
        return 4, 4
    if method == "fixed_6g_2l":
        return 6, 2
    if method in {"proposed_lsa_glqie", "shuffled_size_allocation"}:
        return {"small": (2, 6), "medium": (4, 4), "large": (6, 2)}[category]
    if method == "random_allocation":
        if rng is None:
            rng = np.random.default_rng(0)
        return [(2, 6), (4, 4), (6, 2)][int(rng.integers(0, 3))]
    raise ValueError(f"Unknown method {method}")


def build_coefficients(global_coeffs: np.ndarray, local_coeffs: np.ndarray, method: str, categories: np.ndarray, seed: int = 0) -> np.ndarray:
    """Construct exactly eight input coefficients per sample."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, cat in enumerate(categories):
        g, l = allocation_for_method(method, str(cat), rng)
        row = np.concatenate([global_coeffs[i, :g], local_coeffs[i, :l]])
        if row.size != 8:
            raise ValueError(f"Method {method} produced {row.size} coefficients instead of 8")
        rows.append(row)
    return np.vstack(rows).astype(np.float64)


def shuffle_categories(categories: np.ndarray, seed: int) -> np.ndarray:
    """Deterministically shuffle lesion-size labels while preserving counts."""
    rng = np.random.default_rng(seed)
    out = np.array(categories, dtype=object).copy()
    rng.shuffle(out)
    return out

