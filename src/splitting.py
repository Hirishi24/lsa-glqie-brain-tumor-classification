"""Patient-disjoint grouped splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


def patient_wise_folds(metadata: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create train, validation and test indices with no patient overlap."""
    y = metadata["class_label"].to_numpy()
    groups = metadata["patient_id"].astype(str).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for trainval_idx, test_idx in sgkf.split(np.zeros(len(y)), y, groups):
        trainval = metadata.iloc[trainval_idx].reset_index()
        patient_df = trainval.groupby("patient_id", as_index=False)["class_label"].first()
        if patient_df["class_label"].nunique() > 1 and len(patient_df) >= 6:
            val_frac = min(0.25, max(1.0 / max(n_splits, 2), 0.15))
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            p_train, p_val = next(sss.split(patient_df[["patient_id"]], patient_df["class_label"]))
            val_patients = set(patient_df.iloc[p_val]["patient_id"].astype(str))
        else:
            unique_patients = sorted(patient_df["patient_id"].astype(str))
            val_patients = set(unique_patients[: max(1, len(unique_patients) // max(n_splits, 2))])
        val_idx = trainval.loc[trainval["patient_id"].astype(str).isin(val_patients), "index"].to_numpy()
        train_idx = np.setdiff1d(trainval_idx, val_idx)
        assert_no_patient_overlap(metadata, train_idx, val_idx, test_idx)
        folds.append((train_idx, val_idx, test_idx))
    return folds


def assert_no_patient_overlap(metadata: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Assert train/validation/test patient sets are pairwise disjoint."""
    train_p = set(metadata.iloc[train_idx]["patient_id"].astype(str))
    val_p = set(metadata.iloc[val_idx]["patient_id"].astype(str))
    test_p = set(metadata.iloc[test_idx]["patient_id"].astype(str))
    if train_p & val_p or train_p & test_p or val_p & test_p:
        raise AssertionError("Patient leakage detected among train/validation/test")

