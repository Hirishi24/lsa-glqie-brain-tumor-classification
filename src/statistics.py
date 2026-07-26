"""Statistical summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def confidence_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Normal-approximation CI over fold values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    mean = arr.mean()
    half = 1.96 * arr.std(ddof=1) / np.sqrt(arr.size)
    return float(mean - half), float(mean + half)


def paired_permutation(a: np.ndarray, b: np.ndarray, seed: int = 0, n_resamples: int = 5000) -> float:
    """Two-sided paired sign-flip permutation test on mean difference."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return np.nan
    obs = abs(diff.mean())
    samples = []
    for _ in range(n_resamples):
        signs = rng.choice([-1.0, 1.0], size=diff.size)
        samples.append(abs((diff * signs).mean()))
    return float((np.asarray(samples) >= obs).mean())


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d."""
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2 or diff.std(ddof=1) == 0:
        return np.nan
    return float(diff.mean() / diff.std(ddof=1))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-adjust p-values."""
    indexed = sorted([(p, i) for i, p in enumerate(p_values) if np.isfinite(p)])
    adjusted = [np.nan] * len(p_values)
    m = len(indexed)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adjusted[i] = running
    return adjusted


def compare_methods(fold_metrics: pd.DataFrame, proposed: str, baselines: list[str], metric: str, seed: int, n_resamples: int) -> pd.DataFrame:
    """Compute paired fold-level comparisons."""
    rows = []
    for base in baselines:
        p = fold_metrics[fold_metrics["method"] == proposed].sort_values("fold")
        b = fold_metrics[fold_metrics["method"] == base].sort_values("fold")
        merged = p[["fold", metric]].merge(b[["fold", metric]], on="fold", suffixes=("_proposed", "_baseline"))
        if len(merged) < 5:
            rows.append({"comparison": f"{proposed} vs {base}", "metric": metric, "status": "insufficient paired samples", "wilcoxon_p": np.nan, "permutation_p": np.nan, "cohen_d": np.nan})
            continue
        a = merged[f"{metric}_proposed"].to_numpy()
        c = merged[f"{metric}_baseline"].to_numpy()
        try:
            w_p = float(wilcoxon(a, c).pvalue)
        except Exception:
            w_p = np.nan
        rows.append({"comparison": f"{proposed} vs {base}", "metric": metric, "status": "ok", "wilcoxon_p": w_p, "permutation_p": paired_permutation(a, c, seed, n_resamples), "cohen_d": cohen_d_paired(a, c)})
    out = pd.DataFrame(rows)
    out["holm_permutation_p"] = holm_adjust(out["permutation_p"].tolist())
    return out

