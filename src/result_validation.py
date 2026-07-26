"""Validate result completeness."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plotting import FIGURE_NAMES
from .tables import TABLE_NAMES


def validate_results(output_dir: Path) -> dict[str, object]:
    """Validate expected final artifacts and probability files."""
    issues = []
    table_dir = output_dir / "tables"
    plot_dir = output_dir / "plots"
    for i, name in enumerate(TABLE_NAMES, start=1):
        for ext in ["csv", "md", "tex"]:
            path = table_dir / f"table_{i}_{name}.{ext}"
            if not path.exists() or path.stat().st_size == 0:
                issues.append(f"missing_or_empty:{path}")
    for i, name in enumerate(FIGURE_NAMES, start=1):
        for ext in ["png", "pdf"]:
            path = plot_dir / f"figure_{i}_{name}.{ext}"
            if not path.exists() or path.stat().st_size == 0:
                issues.append(f"missing_or_empty:{path}")
    pred = output_dir / "predictions" / "slice_predictions.csv"
    if pred.exists() and pred.stat().st_size:
        df = pd.read_csv(pred)
        prob_cols = ["prob_class_1", "prob_class_2", "prob_class_3"]
        if not df.empty and not ((df[prob_cols].sum(axis=1) - 1.0).abs() < 1e-4).all():
            issues.append("slice_probabilities_not_normalized")
    return {"valid": not issues, "issues": issues, "table_count": len(TABLE_NAMES), "figure_count": len(FIGURE_NAMES)}

