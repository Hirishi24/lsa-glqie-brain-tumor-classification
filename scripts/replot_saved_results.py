"""Regenerate final plots and robustness table from saved result CSVs.

This script is intended for post-processing an existing results directory. It
does not rerun training and it does not fabricate missing robustness analyses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.config import DEFAULT_CONFIG, deep_update
from src.plotting import create_all_figures
from src.tables import save_table


def _load_config(results_dir: Path) -> dict:
    config_path = results_dir / "configs" / "resolved_config.yaml"
    if not config_path.exists():
        return DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        return deep_update(DEFAULT_CONFIG, yaml.safe_load(f) or {})


def _load_metadata(results_dir: Path) -> pd.DataFrame:
    pred = pd.read_csv(results_dir / "predictions" / "slice_predictions.csv")
    metadata = (
        pred[["file_number", "patient_id", "true_label", "lesion_size_category"]]
        .drop_duplicates("file_number")
        .rename(columns={"true_label": "class_label"})
        .sort_values("file_number")
        .reset_index(drop=True)
    )
    ratio_path = results_dir / "plots" / "data" / "figure_1_dataset_overview_tumor_area_ratios.csv"
    if ratio_path.exists():
        ratios = pd.read_csv(ratio_path)
        if "tumor_area_ratio" in ratios and len(ratios) == len(metadata):
            metadata["tumor_area_ratio"] = ratios["tumor_area_ratio"].to_numpy()
    if "tumor_area_ratio" not in metadata:
        metadata["tumor_area_ratio"] = pd.NA
    return metadata


def _load_fold_metrics(results_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(results_dir / "plots" / "data" / "figure_4_main_performance_comparison_fold_metrics.csv")
    per_class_path = results_dir / "plots" / "data" / "figure_5_confusion_and_per_class_results_per_class_fold_metrics.csv"
    if per_class_path.exists():
        per_class = pd.read_csv(per_class_path)
        shared = ["seed", "fold", "method"]
        extra_cols = [c for c in per_class.columns if c not in base.columns and c not in shared]
        if extra_cols:
            base = base.merge(per_class[shared + extra_cols], on=shared, how="left")
    return base


def _rewrite_table5(results_dir: Path) -> None:
    table5 = pd.DataFrame(
        [
            {"analysis": "Shot robustness", "status": "not_run_in_saved_outputs", "how_to_generate": "rerun with --shots 128, 256, 512, and 1024"},
            {"analysis": "Effective noise", "status": "not_run_in_saved_outputs", "how_to_generate": "rerun with --noise-level 0.001, 0.005, and 0.01"},
            {"analysis": "ROI perturbation", "status": "not_run_in_saved_outputs", "how_to_generate": "rerun with ROI perturbation configuration"},
            {"analysis": "Context margin", "status": "not_run_in_saved_outputs", "how_to_generate": "rerun with data.local_margin set to 0, 0.10, 0.15, and 0.25"},
            {"analysis": "Coefficient budget", "status": "not_run_in_saved_outputs", "how_to_generate": "rerun with 4, 8, and 12 coefficient budgets"},
        ]
    )
    save_table(table5, results_dir, 5, "robustness_results")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate plots from saved LSA-GLQIE results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--data-dir", default=None, help="Optional dataset directory for real MRI example panels")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    cfg = _load_config(results_dir)
    metadata = _load_metadata(results_dir)
    fold_metrics = _load_fold_metrics(results_dir)
    patient_predictions = pd.read_csv(results_dir / "predictions" / "patient_predictions.csv")
    _rewrite_table5(results_dir)
    validation = create_all_figures(results_dir, cfg, metadata, fold_metrics, patient_predictions, args.data_dir)
    print(f"Regenerated {len(validation)} figure groups in {results_dir / 'plots'}")
    print(f"Rewrote table 5 in {results_dir / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

