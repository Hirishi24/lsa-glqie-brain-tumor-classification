"""End-to-end experiment orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .allocation import METHODS_ALL, METHODS_PRIMARY, build_coefficients, shuffle_categories
from .classical_models import fit_predict_svm, fit_torch_logreg, predict_proba_torch
from .dataset_io import BrainTumorDataset, EXPECTED_ARCHIVES, build_metadata, discover_dataset, load_cvind, sha256_file
from .dataset_validation import official_fold_patient_leakage, validate_dataset
from .dct_features import dct_coefficients
from .environment import save_environment, select_device
from .lesion_size import add_size_categories, lesion_features
from .metrics import aggregate_patient_predictions, compute_metrics
from .plotting import create_all_figures
from .preprocessing import preprocess_image_mask
from .quantum_encoding import fit_robust_scaler, quantum_features, transform_robust
from .quantum_simulator import logical_resource_counts
from .result_validation import validate_results
from .roi_processing import local_crop
from .splitting import patient_wise_folds
from .statistics import compare_methods, confidence_interval
from .tables import TABLE_NAMES, save_table


def ensure_result_dirs(output_dir: Path) -> None:
    """Create all required result directories."""
    for name in ["logs", "configs", "cache", "cache/processed_features", "cache/fold_cache", "predictions", "models", "tables", "plots", "reproducibility"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)
        path.write_text("Parquet unavailable; see CSV with same stem.\n", encoding="utf-8")


def _quick_balanced_metadata(metadata: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    chunks = []
    per_class = max(1, max_samples // int(metadata["class_label"].nunique()))
    for _, group in metadata.groupby("class_label"):
        chunks.append(group.sample(n=min(per_class, len(group)), random_state=seed))
    return pd.concat(chunks).sort_values("file_number").reset_index(drop=True)


def _feature_cache_path(output_dir: Path, cfg: dict[str, Any], metadata: pd.DataFrame) -> Path:
    key = {
        "files": [int(x) for x in metadata["file_number"].tolist()],
        "data": cfg["data"],
        "features": cfg["features"],
    }
    import hashlib

    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return output_dir / "cache" / "processed_features" / f"features_{digest}.joblib"


def compute_base_features(dataset: BrainTumorDataset, metadata: pd.DataFrame, cfg: dict[str, Any], output_dir: Path, resume: bool) -> pd.DataFrame:
    """Compute preprocessing, DCT and lesion geometry features."""
    cache = _feature_cache_path(output_dir, cfg, metadata)
    if resume and cache.exists():
        return joblib.load(cache)
    rows = []
    max_coeff = max(12, int(cfg["features"]["total_coefficients"]))
    for _, rec in metadata.sort_values("file_number").iterrows():
        sample = dataset.read_sample(int(rec["file_number"]))
        image, mask = preprocess_image_mask(
            np.asarray(sample["image"]),
            np.asarray(sample["tumorMask"]),
            int(cfg["data"]["image_size"]),
            float(cfg["data"]["percentile_clip_low"]),
            float(cfg["data"]["percentile_clip_high"]),
        )
        local = local_crop(image, mask, int(cfg["data"]["local_view_size"]), float(cfg["data"]["local_margin"]))
        global_coeff = dct_coefficients(image, int(cfg["data"]["global_view_size"]), max_coeff, bool(cfg["features"]["exclude_dc"]))
        local_coeff = dct_coefficients(local, int(cfg["data"]["local_view_size"]), max_coeff, bool(cfg["features"]["exclude_dc"]))
        geo = lesion_features(mask)
        row = {
            "file_number": int(rec["file_number"]),
            "patient_id": str(rec["patient_id"]),
            "class_label": int(rec["class_label"]),
            **geo,
            **{f"global_dct_{i}": global_coeff[i] for i in range(max_coeff)},
            **{f"local_dct_{i}": local_coeff[i] for i in range(max_coeff)},
        }
        rows.append(row)
    features = pd.DataFrame(rows).sort_values("file_number").reset_index(drop=True)
    joblib.dump(features, cache)
    return features


def _labels_zero_based(labels: np.ndarray) -> np.ndarray:
    return labels.astype(int) - 1


def _train_eval_method(
    method: str,
    features: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: dict[str, Any],
    device: str,
    seed: int,
    fold: int,
    output_dir: Path,
    categories: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    total_coefficients = int(cfg["features"]["total_coefficients"])
    global_cols = [f"global_dct_{i}" for i in range(total_coefficients)]
    local_cols = [f"local_dct_{i}" for i in range(total_coefficients)]
    g = features[global_cols].to_numpy()
    l = features[local_cols].to_numpy()
    y = _labels_zero_based(features["class_label"].to_numpy())
    model_dir = output_dir / "models" / f"fold_{fold}" / method
    model_dir.mkdir(parents=True, exist_ok=True)
    size_only = features[["tumor_area_ratio", "bounding_box_ratio", "equivalent_diameter", "compactness"]].fillna(0).to_numpy()
    coeff_method = method
    if method == "shuffled_size_allocation":
        coeff_method = "proposed_lsa_glqie"
    elif method in {"classical_fixed_4g_4l", "svm_classical_fixed_4g_4l"}:
        coeff_method = "fixed_4g_4l"
    elif method in {"classical_proposed", "svm_classical_proposed"}:
        coeff_method = "proposed_lsa_glqie"
    if method == "lesion_size_only":
        coeff = size_only
    elif method == "shuffled_size_allocation":
        coeff = build_coefficients(g, l, coeff_method, shuffle_categories(categories, seed), seed, total_coefficients)
    else:
        coeff = build_coefficients(g, l, coeff_method, categories, seed, total_coefficients)
    scaler = fit_robust_scaler(coeff[train_idx])
    x_train = transform_robust(coeff[train_idx], scaler, float(cfg["features"]["robust_clip"]))
    x_val = transform_robust(coeff[val_idx], scaler, float(cfg["features"]["robust_clip"]))
    x_test = transform_robust(coeff[test_idx], scaler, float(cfg["features"]["robust_clip"]))
    joblib.dump(scaler, model_dir / "scaler.joblib")
    classical_methods = {"classical_proposed", "classical_fixed_4g_4l", "lesion_size_only", "svm_classical_proposed", "svm_classical_fixed_4g_4l"}
    is_quantum = method not in classical_methods
    if is_quantum:
        batch_size = int(cfg.get("runtime", {}).get("batch_size", 256))
        quantum_args = {
            "shots": int(cfg["quantum"].get("shots", 0)),
            "noise_level": float(cfg["quantum"].get("noise_level", 0.0)),
            "rounds": int(cfg["quantum"]["reuploading_rounds"]),
            "num_qubits": int(cfg["quantum"]["num_qubits"]),
            "gate": str(cfg["quantum"]["encoding_gate"]),
        }
        x_train_model = quantum_features(x_train, device, batch_size, seed=seed, **quantum_args)
        x_val_model = quantum_features(x_val, device, batch_size, seed=seed + 1, **quantum_args)
        x_test_model = quantum_features(x_test, device, batch_size, seed=seed + 2, **quantum_args)
    else:
        x_train_model, x_val_model, x_test_model = x_train, x_val, x_test
    if method in {"svm_classical_proposed", "svm_classical_fixed_4g_4l"}:
        model, test_probs = fit_predict_svm(np.vstack([x_train_model, x_val_model]), np.concatenate([y[train_idx], y[val_idx]]), x_test_model, seed)
        joblib.dump(model, model_dir / "svm.joblib")
        val_probs = model.predict_proba(x_val_model)
        train_probs = model.predict_proba(x_train_model)
    else:
        result = fit_torch_logreg(x_train_model, y[train_idx], x_val_model, y[val_idx], device, cfg["classifier"], seed)
        torch.save(result.model.state_dict(), model_dir / "model_state.pt")
        pd.DataFrame(result.history).to_csv(model_dir / "training_history.csv", index=False)
        train_probs = predict_proba_torch(result.model, x_train_model, device)
        val_probs = predict_proba_torch(result.model, x_val_model, device)
        test_probs = predict_proba_torch(result.model, x_test_model, device)
    test_meta = features.iloc[test_idx].copy().reset_index(drop=True)
    slice_pred = pd.DataFrame(
        {
            "seed": seed,
            "fold": fold,
            "method": method,
            "file_number": test_meta["file_number"],
            "patient_id": test_meta["patient_id"],
            "true_label": test_meta["class_label"],
            "pred_label": test_probs.argmax(axis=1) + 1,
            "prob_class_1": test_probs[:, 0],
            "prob_class_2": test_probs[:, 1],
            "prob_class_3": test_probs[:, 2],
            "lesion_size_category": test_meta["lesion_size_category"],
        }
    )
    patient_pred = aggregate_patient_predictions(slice_pred)
    patient_pred.insert(0, "method", method)
    patient_pred.insert(0, "fold", fold)
    patient_pred.insert(0, "seed", seed)
    slice_metrics = compute_metrics(slice_pred["true_label"].to_numpy() - 1, test_probs, "slice_")
    patient_probs = patient_pred[["prob_class_1", "prob_class_2", "prob_class_3"]].to_numpy()
    patient_metrics = compute_metrics(patient_pred["true_label"].to_numpy() - 1, patient_probs, "patient_")
    metrics = {"seed": seed, "fold": fold, "method": method, **slice_metrics, **patient_metrics}
    for cat in ["small", "medium", "large"]:
        sub = slice_pred[slice_pred["lesion_size_category"] == cat]
        metrics[f"{cat}_slice_support"] = float(len(sub))
        if len(sub) and sub["true_label"].nunique() > 1:
            sub_probs = sub[["prob_class_1", "prob_class_2", "prob_class_3"]].to_numpy()
            metrics[f"{cat}_slice_macro_f1"] = compute_metrics(sub["true_label"].to_numpy() - 1, sub_probs, "")["macro_f1"]
        else:
            metrics[f"{cat}_slice_macro_f1"] = np.nan
    return metrics, slice_pred, patient_pred


def _summarize_methods(fold_metrics: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    metric_cols = [c for c in fold_metrics.columns if c.startswith("patient_") or c.startswith("slice_") or c.endswith("_slice_macro_f1")]
    for method in methods:
        sub = fold_metrics[fold_metrics["method"] == method]
        for metric in metric_cols:
            vals = sub[metric].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            lo, hi = confidence_interval(vals)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(finite.mean()) if finite.size else np.nan,
                    "std": float(finite.std(ddof=1)) if finite.size > 1 else np.nan,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_folds": int(finite.size),
                }
            )
    return pd.DataFrame(rows)


def _write_manifests(data_dir: Path, output_dir: Path) -> None:
    info = discover_dataset(data_dir)
    rows = []
    for name, path in sorted((info["zips"] or {}).items()):  # type: ignore[union-attr]
        rows.append({"path": str(path), "name": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    if info["cvind"]:
        path = info["cvind"]  # type: ignore[assignment]
        rows.append({"path": str(path), "name": "cvind.mat", "sha256": sha256_file(path), "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(output_dir / "reproducibility" / "file_hashes.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "reproducibility" / "data_manifest.csv", index=False)


def run_experiment(data_dir: str | Path, output_dir: str | Path, cfg: dict[str, Any], args: Any, logger: Any) -> dict[str, object]:
    """Run the complete research experiment."""
    start_time = time.time()
    output = Path(output_dir)
    ensure_result_dirs(output)
    device = select_device(str(args.device))
    save_environment(output, device)
    logger.info("Using device: %s", device)
    expected_coefficients = int(cfg["quantum"]["num_qubits"]) * int(cfg["quantum"]["reuploading_rounds"])
    if int(cfg["features"]["total_coefficients"]) != expected_coefficients:
        raise ValueError(
            "features.total_coefficients must equal quantum.num_qubits * quantum.reuploading_rounds "
            f"({expected_coefficients}) for this angle reuploading circuit"
        )
    data_path = Path(data_dir)
    _write_manifests(data_path, output)
    metadata = build_metadata(data_path)
    if cfg["experiment"].get("quick_max_samples"):
        metadata = _quick_balanced_metadata(metadata, int(cfg["experiment"]["quick_max_samples"]), int(cfg["experiment"]["seeds"][0]))
    dataset = BrainTumorDataset(data_path, metadata)
    validated, exclusions = validate_dataset(dataset, data_path, full=bool(cfg["data"]["validate_every_sample"]))
    selected = validated[validated["valid"].fillna(False)].copy().sort_values("file_number").reset_index(drop=True)
    selected = selected[selected["file_number"].isin(metadata["file_number"])].reset_index(drop=True)
    _save_dataframe(selected, output / "cache" / "metadata.parquet")
    exclusions.to_csv(output / "cache" / "validation_exclusions.csv", index=False)
    official_leak = official_fold_patient_leakage(validated)
    official_leak.to_csv(output / "cache" / "official_fold_patient_leakage.csv", index=False)
    features = compute_base_features(dataset, selected, cfg, output, bool(getattr(args, "resume", False)))
    folds = patient_wise_folds(features, int(cfg["experiment"]["n_splits"]), int(cfg["experiment"]["seeds"][0]))
    split_rows = []
    all_metrics = []
    all_slice_preds = []
    all_patient_preds = []
    methods = METHODS_PRIMARY if getattr(args, "quick", False) else [m for m in METHODS_ALL if not getattr(args, "skip_ablations", False) or m in METHODS_PRIMARY]
    for seed in cfg["experiment"]["seeds"]:
        for fold, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
            fold_features, thresholds = add_size_categories(features, np.isin(np.arange(len(features)), train_idx), cfg["lesion_size"]["quantiles"])
            fold_features.to_csv(output / "cache" / "fold_cache" / f"fold_{fold}_features.csv", index=False)
            for split_name, idxs in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
                for idx in idxs:
                    split_rows.append({"seed": seed, "fold": fold, "split": split_name, "file_number": int(features.iloc[idx]["file_number"]), "patient_id": features.iloc[idx]["patient_id"], "class_label": int(features.iloc[idx]["class_label"]), "threshold_small": thresholds[0], "threshold_large": thresholds[1]})
            categories = fold_features["lesion_size_category"].to_numpy()
            for method in methods:
                metrics, slice_pred, patient_pred = _train_eval_method(method, fold_features, train_idx, val_idx, test_idx, cfg, device, int(seed), fold, output, categories)
                metrics["threshold_small"] = thresholds[0]
                metrics["threshold_large"] = thresholds[1]
                all_metrics.append(metrics)
                all_slice_preds.append(slice_pred)
                all_patient_preds.append(patient_pred)
                logger.info("Finished seed=%s fold=%s method=%s patient_macro_f1=%.3f", seed, fold, method, metrics.get("patient_macro_f1", np.nan))
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(output / "cache" / "fold_cache" / "split_membership.csv", index=False)
    fold_metrics = pd.DataFrame(all_metrics)
    fold_metrics.to_csv(output / "cache" / "fold_metrics.csv", index=False)
    slice_predictions = pd.concat(all_slice_preds, ignore_index=True)
    patient_predictions = pd.concat(all_patient_preds, ignore_index=True)
    slice_predictions.to_csv(output / "predictions" / "slice_predictions.csv", index=False)
    patient_predictions.to_csv(output / "predictions" / "patient_predictions.csv", index=False)
    selected_with_sizes = features.merge(fold_features[["file_number", "lesion_size_category"]], on="file_number", how="left")
    table1 = pd.concat(
        [
            selected.groupby("class_label").agg(slice_count=("file_number", "count"), patient_count=("patient_id", "nunique")).reset_index(),
            pd.DataFrame([{"class_label": "official_fold_patient_leakage", "slice_count": len(official_leak), "patient_count": len(official_leak)}]),
        ],
        ignore_index=True,
    )
    table2 = _summarize_methods(fold_metrics, METHODS_PRIMARY)
    table3 = _summarize_methods(fold_metrics, methods)[lambda d: d["metric"].str.contains("class_|small_|medium_|large_")]
    table4 = _summarize_methods(fold_metrics, methods)
    table5 = pd.DataFrame(
        [
            {"analysis": "Shot robustness", "status": "not_run_in_primary_saved_outputs", "available_command": "python main.py --data-dir Dataset --output-dir results_shots --shots 128 --device cuda"},
            {"analysis": "Effective noise", "status": "not_run_in_primary_saved_outputs", "available_command": "python main.py --data-dir Dataset --output-dir results_noise --noise-level 0.005 --device cuda"},
            {"analysis": "ROI perturbation", "status": "not_run_in_primary_saved_outputs", "available_command": "requires rerun with ROI perturbation configuration"},
            {"analysis": "Context margin", "status": "not_run_in_primary_saved_outputs", "available_command": "edit data.local_margin and rerun"},
            {"analysis": "Coefficient budget", "status": "not_run_in_primary_saved_outputs", "available_command": "edit features.total_coefficients and quantum.reuploading_rounds and rerun"},
        ]
    )
    stats = compare_methods(
        fold_metrics,
        "proposed_lsa_glqie",
        ["global_only", "local_only", "fixed_4g_4l", "random_allocation", "classical_proposed", "classical_fixed_4g_4l", "svm_classical_proposed"],
        "patient_macro_f1",
        int(cfg["experiment"]["seeds"][0]),
        int(cfg["statistics"]["permutation_resamples"]),
    )
    resources = pd.DataFrame(
        [
            {
                "comparison": "logical_resources",
                "metric": "primary_circuit",
                "status": "ok",
                **logical_resource_counts(
                    int(cfg["quantum"]["reuploading_rounds"]),
                    int(cfg["quantum"]["num_qubits"]),
                    str(cfg["quantum"]["encoding_gate"]),
                ),
            }
        ]
    )
    table6 = pd.concat([stats, resources], ignore_index=True)
    for i, (name, table) in enumerate(zip(TABLE_NAMES, [table1, table2, table3, table4, table5, table6]), start=1):
        save_table(table, output, i, name)
    plot_validation = create_all_figures(output, cfg, selected_with_sizes, fold_metrics, patient_predictions, data_path)
    validation_report = validate_results(output)
    runtime = {"total_seconds": time.time() - start_time, "device": device, "validation": validation_report}
    with (output / "reproducibility" / "runtime_summary.json").open("w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)
    logger.info("Final validation: %s", validation_report)
    return {"output_dir": str(output), "device": device, "tables": 6, "figures": 8, "plot_validation_rows": len(plot_validation), "validation": validation_report}
