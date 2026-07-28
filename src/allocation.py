"""Fixed-budget global-local coefficient allocation."""

from __future__ import annotations

import numpy as np

METHODS_PRIMARY = [
    "global_only",
    "local_only",
    "fixed_4g_4l",
    "random_allocation",
    "proposed_lsa_glqie",
    "classical_proposed",
    "classical_fixed_4g_4l",
    "svm_classical_proposed",
]
METHODS_ALL = METHODS_PRIMARY + [
    "fixed_2g_6l",
    "fixed_6g_2l",
    "shuffled_size_allocation",
    "lesion_size_only",
    "svm_classical_fixed_4g_4l",
]


def allocation_profiles(total_coefficients: int = 12) -> dict[str, tuple[int, int]]:
    """Return small/medium/large global-local profiles for a fixed budget."""
    total = int(total_coefficients)
    if total < 2:
        raise ValueError(f"total_coefficients must be at least 2, got {total}")
    small_g = max(1, round(total * 0.25))
    medium_g = total // 2
    large_g = min(total - 1, round(total * 0.75))
    return {
        "small": (small_g, total - small_g),
        "medium": (medium_g, total - medium_g),
        "large": (large_g, total - large_g),
    }


def allocation_for_method(method: str, category: str, rng: np.random.Generator | None = None, total_coefficients: int = 12) -> tuple[int, int]:
    """Return (global_count, local_count) for a fixed-budget method."""
    total = int(total_coefficients)
    profiles = allocation_profiles(total)
    if method == "global_only":
        return total, 0
    if method == "local_only":
        return 0, total
    if method == "fixed_2g_6l":
        return profiles["small"]
    if method in {"fixed_4g_4l", "classical_fixed_4g_4l", "svm_classical_fixed_4g_4l"}:
        return profiles["medium"]
    if method == "fixed_6g_2l":
        return profiles["large"]
    if method in {"proposed_lsa_glqie", "classical_proposed", "svm_classical_proposed", "shuffled_size_allocation"}:
        return profiles[category]
    if method == "random_allocation":
        if rng is None:
            rng = np.random.default_rng(0)
        return list(profiles.values())[int(rng.integers(0, 3))]
    raise ValueError(f"Unknown method {method}")


def build_coefficients(
    global_coeffs: np.ndarray,
    local_coeffs: np.ndarray,
    method: str,
    categories: np.ndarray,
    seed: int = 0,
    total_coefficients: int = 12,
) -> np.ndarray:
    """Construct exactly total_coefficients input coefficients per sample."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, cat in enumerate(categories):
        g, l = allocation_for_method(method, str(cat), rng, total_coefficients)
        row = np.concatenate([global_coeffs[i, :g], local_coeffs[i, :l]])
        if row.size != int(total_coefficients):
            raise ValueError(f"Method {method} produced {row.size} coefficients instead of {total_coefficients}")
        rows.append(row)
    return np.vstack(rows).astype(np.float64)


def shuffle_categories(categories: np.ndarray, seed: int) -> np.ndarray:
    """Deterministically shuffle lesion-size labels while preserving counts."""
    rng = np.random.default_rng(seed)
    out = np.array(categories, dtype=object).copy()
    rng.shuffle(out)
    return out
