"""Metrics and patient aggregation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute multiclass metrics while preserving undefined metrics as NaN."""
    labels = np.array([0, 1, 2])
    pred = np.argmax(probs, axis=1)
    out: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out[prefix + "accuracy"] = float(accuracy_score(y_true, pred))
        out[prefix + "balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
        out[prefix + "macro_precision"] = float(precision_score(y_true, pred, labels=labels, average="macro", zero_division=np.nan))
        out[prefix + "macro_recall"] = float(recall_score(y_true, pred, labels=labels, average="macro", zero_division=np.nan))
        out[prefix + "macro_f1"] = float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=np.nan))
        out[prefix + "weighted_f1"] = float(f1_score(y_true, pred, labels=labels, average="weighted", zero_division=np.nan))
        out[prefix + "mcc"] = float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) > 1 else np.nan
        try:
            onehot = np.eye(3)[y_true]
            out[prefix + "macro_auroc"] = float(roc_auc_score(onehot, probs, average="macro", multi_class="ovr"))
        except Exception:
            out[prefix + "macro_auroc"] = np.nan
        try:
            out[prefix + "macro_auprc"] = float(average_precision_score(np.eye(3)[y_true], probs, average="macro"))
        except Exception:
            out[prefix + "macro_auprc"] = np.nan
    cm = confusion_matrix(y_true, pred, labels=labels)
    for c in labels:
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        out[prefix + f"class_{c + 1}_sensitivity"] = float(tp / (tp + fn)) if tp + fn else np.nan
        out[prefix + f"class_{c + 1}_specificity"] = float(tn / (tn + fp)) if tn + fp else np.nan
    prec, rec, f1, support = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=np.nan)
    for i, c in enumerate(labels):
        out[prefix + f"class_{c + 1}_precision"] = float(prec[i])
        out[prefix + f"class_{c + 1}_recall"] = float(rec[i])
        out[prefix + f"class_{c + 1}_f1"] = float(f1[i])
        out[prefix + f"class_{c + 1}_support"] = float(support[i])
    return out


def aggregate_patient_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Average slice probabilities per patient."""
    prob_cols = ["prob_class_1", "prob_class_2", "prob_class_3"]
    grouped = df.groupby(["patient_id", "true_label"], as_index=False)[prob_cols].mean()
    grouped["pred_label"] = grouped[prob_cols].to_numpy().argmax(axis=1) + 1
    return grouped

