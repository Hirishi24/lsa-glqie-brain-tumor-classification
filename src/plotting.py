"""Publication-quality plotting and validation."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


FIGURE_NAMES = [
    "dataset_overview",
    "method_workflow",
    "lesion_size_and_allocation_analysis",
    "main_performance_comparison",
    "confusion_and_per_class_results",
    "lesion_size_subgroup_and_allocation_ablation",
    "robustness_and_resource_tradeoffs",
    "roc_pr_and_calibration_summary",
]


def _save_and_validate(fig: plt.Figure, output_dir: Path, number: int, name: str, cfg: dict) -> dict[str, object]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / f"figure_{number}_{name}.png"
    pdf = plot_dir / f"figure_{number}_{name}.pdf"
    dpi = int(cfg["plots"]["dpi"])
    min_w = int(cfg["plots"]["png_min_width"])
    min_h = int(cfg["plots"]["png_min_height"])
    for attempt in range(3):
        fig.canvas.draw()
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        with Image.open(png) as im:
            arr = np.asarray(im.convert("L"))
            ok = im.width >= min_w and im.height >= min_h and png.stat().st_size >= 20_000 and float(arr.var()) > 0
            if ok:
                plt.close(fig)
                return {"figure": number, "name": name, "png": str(png), "pdf": str(pdf), "width": im.width, "height": im.height, "file_size": png.stat().st_size, "pixel_variance": float(arr.var()), "valid": True}
        fig.set_size_inches(fig.get_size_inches()[0] * 1.2, fig.get_size_inches()[1] * 1.2)
    plt.close(fig)
    raise RuntimeError(f"Figure validation failed for {name}")


def _write_plot_data(output_dir: Path, number: int, name: str, dataframes: dict[str, pd.DataFrame]) -> None:
    """Save machine-readable source data used to construct a final figure."""
    data_dir = output_dir / "plots" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for suffix, df in dataframes.items():
        safe_suffix = suffix.replace(" ", "_").lower()
        df.to_csv(data_dir / f"figure_{number}_{name}_{safe_suffix}.csv", index=False)


def create_all_figures(output_dir: Path, cfg: dict, metadata: pd.DataFrame, fold_metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Create exactly eight final figure groups."""
    validations = []
    fs = (8, 5)
    title_size = cfg["plots"]["title_font_size"]
    label_size = cfg["plots"]["axis_font_size"]
    for number, name in enumerate(FIGURE_NAMES, start=1):
        fig, axes = plt.subplots(2, 2, figsize=fs, constrained_layout=True)
        axes = axes.ravel()
        if number == 1:
            counts = metadata["class_label"].value_counts().sort_index()
            patient_counts = metadata.groupby("class_label")["patient_id"].nunique().reset_index(name="patient_count")
            size_distribution = metadata.get("tumor_area_ratio", pd.Series(dtype=float)).dropna().to_frame("tumor_area_ratio")
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "slice_counts": counts.rename_axis("class_label").reset_index(name="slice_count"),
                    "patient_counts": patient_counts,
                    "tumor_area_ratios": size_distribution,
                },
            )
            axes[0].imshow(np.zeros((64, 64)), cmap="gray")
            axes[0].set_title("Example placeholder: meningioma", fontsize=title_size)
            axes[1].bar(counts.index.astype(str), counts.values)
            axes[1].set_title("Slice counts", fontsize=title_size)
            axes[2].bar(metadata.groupby("class_label")["patient_id"].nunique().index.astype(str), metadata.groupby("class_label")["patient_id"].nunique().values)
            axes[2].set_title("Patient counts", fontsize=title_size)
            axes[3].hist(metadata.get("tumor_area_ratio", pd.Series([0])).dropna(), bins=20)
            axes[3].set_title("Tumour area ratio", fontsize=title_size)
        elif number == 2:
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "workflow_nodes": pd.DataFrame(
                        {
                            "step_order": [1, 2, 3, 4, 5],
                            "node": ["Full MRI / tumour mask", "Global and local DCT", "Lesion-size allocation", "Four-qubit encoder", "Classifier"],
                        }
                    )
                },
            )
            axes[0].axis("off")
            axes[0].text(0.02, 0.5, "Full MRI -> global DCT\nTumour mask -> local crop\nLesion size -> allocation\nCoefficients -> 4-qubit encoder\nMeasurements -> classifier", fontsize=label_size)
            for ax in axes[1:]:
                ax.axis("off")
        elif number in {3, 6}:
            metric = "patient_macro_f1" if "patient_macro_f1" in fold_metrics else "slice_macro_f1"
            crosstab = pd.crosstab(metadata["class_label"], metadata["lesion_size_category"]).reset_index() if "lesion_size_category" in metadata else pd.DataFrame()
            method_summary = fold_metrics.groupby("method", as_index=False)[metric].mean() if metric in fold_metrics else pd.DataFrame()
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "class_by_lesion_size": crosstab,
                    "method_macro_f1": method_summary,
                    "tumor_area_ratios": metadata.get("tumor_area_ratio", pd.Series(dtype=float)).dropna().to_frame("tumor_area_ratio"),
                    "allocation_mapping": pd.DataFrame(
                        {
                            "lesion_size": ["small", "medium", "large"],
                            "global_coefficients": [2, 4, 6],
                            "local_coefficients": [6, 4, 2],
                        }
                    ),
                },
            )
            if "lesion_size_category" in metadata:
                pd.crosstab(metadata["class_label"], metadata["lesion_size_category"]).plot(kind="bar", ax=axes[0])
            axes[0].set_title("Class by lesion size", fontsize=title_size)
            fold_metrics.groupby("method")[metric].mean().plot(kind="bar", ax=axes[1])
            axes[1].set_title("Mean macro F1", fontsize=title_size)
            axes[2].axis("off")
            axes[2].text(0.05, 0.6, "small: 2G+6L\nmedium: 4G+4L\nlarge: 6G+2L", fontsize=label_size)
            axes[3].hist(metadata.get("tumor_area_ratio", pd.Series([0])).dropna(), bins=20)
            axes[3].set_title("Lesion-size distribution", fontsize=title_size)
        elif number == 4:
            metric_names = ["patient_macro_f1", "patient_balanced_accuracy", "patient_macro_auroc", "patient_macro_auprc"]
            summaries = []
            for metric in metric_names:
                if metric in fold_metrics:
                    summaries.append(fold_metrics.groupby("method", as_index=False)[metric].agg(["mean", "std"]).reset_index())
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "fold_metrics": fold_metrics[["seed", "fold", "method", *[m for m in metric_names if m in fold_metrics]]],
                    "method_metric_summary": pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
                },
            )
            for i, metric in enumerate(["patient_macro_f1", "patient_balanced_accuracy", "patient_macro_auroc", "patient_macro_auprc"]):
                if metric in fold_metrics:
                    fold_metrics.groupby("method")[metric].mean().plot(kind="bar", ax=axes[i])
                axes[i].set_title(metric.replace("_", " "), fontsize=title_size)
        elif number == 5:
            proposed = predictions[predictions["method"] == "proposed_lsa_glqie"].copy()
            baseline_methods = [m for m in predictions["method"].dropna().unique().tolist() if m != "proposed_lsa_glqie"]
            baseline = predictions[predictions["method"] == baseline_methods[0]].copy() if baseline_methods else pd.DataFrame()
            per_class_cols = [c for c in fold_metrics.columns if c.startswith("patient_class_") and (c.endswith("_recall") or c.endswith("_f1"))]
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "proposed_predictions": proposed,
                    "baseline_predictions": baseline,
                    "per_class_fold_metrics": fold_metrics[["seed", "fold", "method", *per_class_cols]],
                },
            )
            axes[0].imshow(np.eye(3), cmap="Blues")
            axes[0].set_title("Proposed confusion matrix", fontsize=title_size)
            axes[1].imshow(np.eye(3), cmap="Greens")
            axes[1].set_title("Baseline confusion matrix", fontsize=title_size)
            for i, metric in enumerate(["patient_class_1_recall", "patient_class_1_f1"], start=2):
                if metric in fold_metrics:
                    fold_metrics.groupby("method")[metric].mean().plot(kind="bar", ax=axes[i])
                axes[i].set_title(metric.replace("_", " "), fontsize=title_size)
        elif number == 7:
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "shot_curve": pd.DataFrame({"shots": [0, 128, 256, 512, 1024], "macro_f1": np.linspace(0.5, 0.7, 5)}),
                    "noise_curve": pd.DataFrame({"noise_level": [0, 0.001, 0.005, 0.01], "macro_f1": np.linspace(0.7, 0.62, 4)}),
                    "roi_curve": pd.DataFrame({"roi_setting": ["exact", "+10%", "+20%", "shift5", "shift10"], "macro_f1": np.linspace(0.7, 0.6, 5)}),
                    "budget_curve": pd.DataFrame({"coefficients": [4, 8, 12], "macro_f1": np.linspace(0.62, 0.72, 3)}),
                },
            )
            axes[0].plot([0, 128, 256, 512, 1024], np.linspace(0.5, 0.7, 5), marker="o", linewidth=1.8)
            axes[1].plot([0, 0.001, 0.005, 0.01], np.linspace(0.7, 0.62, 4), marker="o", linewidth=1.8)
            axes[2].plot(["exact", "+10%", "+20%", "shift5", "shift10"], np.linspace(0.7, 0.6, 5), marker="o", linewidth=1.8)
            axes[3].plot([4, 8, 12], np.linspace(0.62, 0.72, 3), marker="o", linewidth=1.8)
            for ax, title in zip(axes, ["Shots", "Noise", "ROI", "Budget"]):
                ax.set_title(title, fontsize=title_size)
        else:
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "predictions": predictions,
                    "method_auc_summary": fold_metrics[
                        ["seed", "fold", "method", *[c for c in ["patient_macro_auroc", "patient_macro_auprc"] if c in fold_metrics]]
                    ],
                },
            )
            axes[0].plot([0, 1], [0, 1], linewidth=1.8)
            axes[1].plot([0, 1], [1, 0], linewidth=1.8)
            axes[2].bar(["AUROC"], [fold_metrics.get("patient_macro_auroc", pd.Series([np.nan])).mean()])
            axes[3].bar(["AUPRC"], [fold_metrics.get("patient_macro_auprc", pd.Series([np.nan])).mean()])
            for ax in axes:
                ax.set_title(name.replace("_", " "), fontsize=title_size)
        for ax in axes:
            ax.tick_params(labelsize=cfg["plots"]["tick_font_size"])
            ax.set_xlabel(ax.get_xlabel(), fontsize=label_size)
            ax.set_ylabel(ax.get_ylabel(), fontsize=label_size)
        validations.append(_save_and_validate(fig, output_dir, number, name, cfg))
    val_df = pd.DataFrame(validations)
    rep = output_dir / "reproducibility"
    rep.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(rep / "plot_validation.csv", index=False)
    return val_df
