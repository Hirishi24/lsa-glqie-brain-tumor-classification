"""Dataset I/O tests."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from src.dataset_io import BrainTumorDataset, archive_for_file_number, build_metadata, load_cvind


def test_read_mat_file_from_zip(synthetic_dataset_dir):
    md = build_metadata(synthetic_dataset_dir)
    ds = BrainTumorDataset(synthetic_dataset_dir, md)
    sample = ds.read_sample(int(md.iloc[0]["file_number"]))
    assert sample["image"].ndim == 2
    assert sample["tumorMask"].shape == sample["image"].shape
    assert sample["label"] in {1, 2, 3}


def test_read_extracted_mat_file(synthetic_dataset_dir, tmp_path):
    for item in Path(synthetic_dataset_dir).iterdir():
        if item.is_file():
            shutil.copy(item, tmp_path / item.name)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(tmp_path / "brainTumorDataPublic_1-766.zip") as zf:
        zf.extract("1.mat", extracted)
    md = build_metadata(tmp_path)
    ds = BrainTumorDataset(tmp_path, md)
    sample = ds.read_sample(1)
    assert sample["patient_id"].startswith("SYN")


def test_official_file_number_mapping():
    assert archive_for_file_number(1).endswith("1-766.zip")
    assert archive_for_file_number(767).endswith("767-1532.zip")
    assert archive_for_file_number(1533).endswith("1533-2298.zip")
    assert archive_for_file_number(3064).endswith("2299-3064.zip")


def test_cvind_alignment(synthetic_dataset_dir):
    cvind = load_cvind(synthetic_dataset_dir)
    assert cvind[0] == 1
    assert cvind[4] == 5

