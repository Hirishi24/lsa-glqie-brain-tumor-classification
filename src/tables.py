"""Final table writers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLE_NAMES = [
    "dataset_and_split_summary",
    "main_method_results",
    "per_class_and_lesion_size_results",
    "allocation_and_encoding_ablations",
    "robustness_results",
    "statistical_and_resource_summary",
]


def save_table(df: pd.DataFrame, output_dir: Path, number: int, name: str) -> None:
    """Save table as CSV, Markdown and LaTeX."""
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    stem = f"table_{number}_{name}"
    df.to_csv(table_dir / f"{stem}.csv", index=False)
    (table_dir / f"{stem}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    (table_dir / f"{stem}.tex").write_text(df.to_latex(index=False), encoding="utf-8")

