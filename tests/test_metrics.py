"""Metrics tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import aggregate_patient_predictions, compute_metrics


def test_metrics_handle_missing_classes():
    y = np.array([0, 0, 1])
    probs = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1]])
    out = compute_metrics(y, probs)
    assert "macro_f1" in out
    assert "macro_auroc" in out


def test_patient_probability_aggregation():
    df = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b"],
            "true_label": [1, 1, 2],
            "prob_class_1": [0.7, 0.5, 0.2],
            "prob_class_2": [0.2, 0.3, 0.7],
            "prob_class_3": [0.1, 0.2, 0.1],
        }
    )
    out = aggregate_patient_predictions(df)
    assert len(out) == 2
    assert np.allclose(out[["prob_class_1", "prob_class_2", "prob_class_3"]].sum(axis=1), 1.0)

