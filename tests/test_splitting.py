"""Split tests."""

from __future__ import annotations

import numpy as np

from src.dataset_io import build_metadata
from src.splitting import assert_no_patient_overlap, patient_wise_folds


def test_patient_disjoint_split(synthetic_dataset_dir):
    md = build_metadata(synthetic_dataset_dir)
    folds = patient_wise_folds(md, 2, 11)
    assert len(folds) == 2
    for train_idx, val_idx, test_idx in folds:
        assert_no_patient_overlap(md, train_idx, val_idx, test_idx)
        assert len(np.intersect1d(train_idx, test_idx)) == 0

