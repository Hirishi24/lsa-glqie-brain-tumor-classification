"""Dataset integrity checks."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset_io import BrainTumorDataset, load_cvind


def validate_dataset(dataset: BrainTumorDataset, data_dir: str | Path, full: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate samples and return enriched metadata plus exclusion records."""
    md = dataset.metadata.copy()
    exclusions: list[dict[str, object]] = []
    image_hashes: Counter[str] = Counter()
    pair_hashes: Counter[str] = Counter()
    rows = []
    expected = set(range(1, 3065))
    present = set(md["file_number"].astype(int))
    for missing in sorted(expected - present):
        exclusions.append({"file_number": missing, "reason": "missing_file"})
    if md["file_number"].duplicated().any():
        for num in md.loc[md["file_number"].duplicated(), "file_number"]:
            exclusions.append({"file_number": int(num), "reason": "duplicate_file_number"})
    for _, rec in md.iterrows():
        reasons = []
        try:
            sample = dataset.read_sample(int(rec["file_number"]))
            image = np.asarray(sample["image"])
            mask = np.asarray(sample["tumorMask"])
            border = np.asarray(sample["tumorBorder"]).ravel()
            if int(sample["label"]) not in {1, 2, 3}:
                reasons.append("invalid_label")
            if not str(sample["patient_id"]).strip():
                reasons.append("missing_pid")
            if image.ndim != 2:
                reasons.append("invalid_image_dimensionality")
            if mask.ndim != 2:
                reasons.append("invalid_mask_dimensionality")
            if image.shape != mask.shape:
                reasons.append("image_mask_shape_mismatch")
            if not np.isfinite(image).all() or not np.isfinite(mask).all():
                reasons.append("nan_or_inf")
            if np.nanmax(image) == np.nanmin(image):
                reasons.append("constant_image")
            mask_values = set(np.unique(mask).astype(int).tolist())
            if not mask_values.issubset({0, 1}):
                reasons.append("non_binary_mask")
            if int(mask.sum()) <= 0:
                reasons.append("empty_tumor_mask")
            if border.size < 4 or border.size % 2:
                reasons.append("invalid_border_vector")
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()
            pair_hash = hashlib.sha256(image.tobytes() + mask.tobytes()).hexdigest()
            image_hashes[image_hash] += 1
            pair_hashes[pair_hash] += 1
            rows.append({"file_number": int(rec["file_number"]), "valid": not reasons, "reason": ";".join(reasons), "image_hash": image_hash, "pair_hash": pair_hash})
            for reason in reasons:
                exclusions.append({"file_number": int(rec["file_number"]), "reason": reason})
        except Exception as exc:
            rows.append({"file_number": int(rec["file_number"]), "valid": False, "reason": f"read_error:{exc}", "image_hash": "", "pair_hash": ""})
            exclusions.append({"file_number": int(rec["file_number"]), "reason": f"read_error:{exc}"})
        if not full and len(rows) >= len(md):
            break
    val = pd.DataFrame(rows)
    md = md.merge(val, on="file_number", how="left")
    for h, count in image_hashes.items():
        if count > 1:
            md.loc[md["image_hash"] == h, "duplicate_image_hash_count"] = count
    for h, count in pair_hashes.items():
        if count > 1:
            md.loc[md["pair_hash"] == h, "duplicate_pair_hash_count"] = count
    class_per_patient = md.groupby("patient_id")["class_label"].nunique()
    for pid in class_per_patient[class_per_patient > 1].index:
        exclusions.append({"file_number": "", "patient_id": pid, "reason": "patient_has_multiple_classes"})
    try:
        cvind = load_cvind(data_dir)
        if len(cvind) >= int(md["file_number"].max()):
            md["official_fold"] = [int(cvind[int(n) - 1]) for n in md["file_number"]]
    except Exception:
        md["official_fold"] = np.nan
    return md, pd.DataFrame(exclusions)


def official_fold_patient_leakage(metadata: pd.DataFrame) -> pd.DataFrame:
    """Report patients appearing in more than one official fold."""
    if "official_fold" not in metadata:
        return pd.DataFrame(columns=["patient_id", "folds", "n_folds"])
    rows = []
    for pid, group in metadata.groupby("patient_id"):
        folds = sorted(set(int(x) for x in group["official_fold"].dropna()))
        if len(folds) > 1:
            rows.append({"patient_id": pid, "folds": ",".join(map(str, folds)), "n_folds": len(folds)})
    return pd.DataFrame(rows)

