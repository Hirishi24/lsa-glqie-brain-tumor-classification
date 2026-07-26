"""Lesion geometry and size categories."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

from .roi_processing import tumor_bbox


def lesion_features(mask: np.ndarray) -> dict[str, float]:
    """Compute lesion-size and geometry features from a binary mask."""
    m = mask > 0
    area = float(m.sum())
    total = float(m.size)
    y0, y1, x0, x1 = tumor_bbox(m.astype(np.uint8))
    bbox_area = float((y1 - y0) * (x1 - x0))
    eroded = binary_erosion(m) if area > 0 else m
    perimeter = float(np.logical_and(m, ~eroded).sum())
    equiv = float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0
    compactness = float((perimeter * perimeter) / (4.0 * np.pi * area + 1e-12)) if area > 0 else np.nan
    return {
        "tumor_pixel_area": area,
        "tumor_area_ratio": area / total,
        "bounding_box_area": bbox_area,
        "bounding_box_ratio": bbox_area / total,
        "perimeter": perimeter,
        "equivalent_diameter": equiv,
        "compactness": compactness,
    }


def fit_size_thresholds(ratios: np.ndarray, quantiles: list[float]) -> tuple[float, float]:
    """Fit small and medium cut points from training tumour-area ratios."""
    q1, q2 = np.quantile(np.asarray(ratios, dtype=float), quantiles)
    return float(q1), float(q2)


def assign_size_categories(ratios: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    """Assign small, medium or large using training-fitted thresholds."""
    q1, q2 = thresholds
    cats = np.full(len(ratios), "large", dtype=object)
    cats[np.asarray(ratios) < q1] = "small"
    cats[(np.asarray(ratios) >= q1) & (np.asarray(ratios) < q2)] = "medium"
    return cats


def add_size_categories(df: pd.DataFrame, train_mask: np.ndarray, quantiles: list[float]) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Add lesion_size_category using thresholds learned on training rows only."""
    thresholds = fit_size_thresholds(df.loc[train_mask, "tumor_area_ratio"].to_numpy(), quantiles)
    out = df.copy()
    out["lesion_size_category"] = assign_size_categories(out["tumor_area_ratio"].to_numpy(), thresholds)
    return out, thresholds

