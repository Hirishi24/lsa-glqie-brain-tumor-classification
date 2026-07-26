"""Shared fixtures."""

from __future__ import annotations

import pytest

from src.synthetic_data import create_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_dataset_dir(tmp_path_factory):
    """Create one synthetic dataset for the test session."""
    root = tmp_path_factory.mktemp("synthetic_dataset")
    return create_synthetic_dataset(root, seed=123, patients_per_class=6, slices_per_patient=3)

