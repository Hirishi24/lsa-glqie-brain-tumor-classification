"""Lazy readers for the Figshare brain tumour MATLAB v7.3 dataset."""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

EXPECTED_ARCHIVES = [
    "brainTumorDataPublic_1-766.zip",
    "brainTumorDataPublic_767-1532.zip",
    "brainTumorDataPublic_1533-2298.zip",
    "brainTumorDataPublic_2299-3064.zip",
]


@dataclass(frozen=True)
class SampleRecord:
    """Metadata for a single image slice."""

    sample_id: str
    file_number: int
    archive_name: str
    member_name: str
    extracted_path: str
    patient_id: str
    class_label: int
    original_height: int
    original_width: int


def decode_matlab_char(value: h5py.Dataset | np.ndarray | str | bytes) -> str:
    """Decode MATLAB char arrays stored as uint16, bytes, object arrays or strings."""
    if isinstance(value, str):
        return value.strip().replace("\x00", "")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip().replace("\x00", "")
    arr = np.asarray(value[()] if isinstance(value, h5py.Dataset) else value)
    if arr.dtype.kind in {"U", "S"}:
        text = "".join(x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x) for x in arr.ravel(order="F"))
        return text.strip().replace("\x00", "")
    chars = []
    for x in arr.ravel(order="F"):
        code = int(x)
        if code:
            chars.append(chr(code))
    return "".join(chars).strip()


def resolve_dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    """Resolve a struct field that may be direct or stored as an HDF5 object reference."""
    obj = group[name]
    if isinstance(obj, h5py.Dataset) and obj.dtype == h5py.ref_dtype:
        ref = obj[()].ravel()[0]
        return group.file[ref]
    if isinstance(obj, h5py.Dataset):
        return obj
    raise ValueError(f"Field {name!r} is not a dataset")


def _read_h5_sample(handle: h5py.File) -> tuple[int, str, np.ndarray, np.ndarray, np.ndarray]:
    if "cjdata" not in handle:
        raise ValueError("Missing cjdata group")
    cj = handle["cjdata"]
    label = int(np.asarray(resolve_dataset(cj, "label"))[()].ravel()[0])
    patient_id = decode_matlab_char(resolve_dataset(cj, "PID"))
    image = np.asarray(resolve_dataset(cj, "image")[()])
    mask = np.asarray(resolve_dataset(cj, "tumorMask")[()])
    border = np.asarray(resolve_dataset(cj, "tumorBorder")[()]).ravel()
    return label, patient_id, image, mask, border


def archive_for_file_number(file_number: int) -> str:
    """Return expected archive name for a numeric MATLAB file."""
    if 1 <= file_number <= 766:
        return EXPECTED_ARCHIVES[0]
    if 767 <= file_number <= 1532:
        return EXPECTED_ARCHIVES[1]
    if 1533 <= file_number <= 2298:
        return EXPECTED_ARCHIVES[2]
    if 2299 <= file_number <= 3064:
        return EXPECTED_ARCHIVES[3]
    raise ValueError(f"Unexpected file number: {file_number}")


def discover_dataset(data_dir: str | Path) -> dict[str, object]:
    """Discover README, cvind, expected ZIPs and extracted MATLAB files."""
    root = Path(data_dir)
    zips = {p.name: p for p in root.glob("*.zip") if p.name in EXPECTED_ARCHIVES}
    mats = {}
    for p in root.rglob("*.mat"):
        if p.name == "cvind.mat":
            continue
        try:
            mats[int(p.stem)] = p
        except ValueError:
            continue
    readmes = list(root.glob("README*"))
    cvind = root / "cvind.mat" if (root / "cvind.mat").exists() else None
    source = "extracted" if len(mats) == 3064 else "zip"
    return {"root": root, "zips": zips, "mats": mats, "readme": readmes[0] if readmes else None, "cvind": cvind, "source": source}


def _iter_zip_members(path: Path) -> Iterable[tuple[int, str]]:
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".mat"):
                yield int(Path(name).stem), name


def build_metadata(data_dir: str | Path, max_samples: int | None = None) -> pd.DataFrame:
    """Build deterministic sample metadata without keeping image arrays in memory."""
    info = discover_dataset(data_dir)
    rows: list[SampleRecord] = []
    if info["source"] == "extracted":
        items = sorted(info["mats"].items())  # type: ignore[union-attr]
        for num, path in items[:max_samples]:
            with h5py.File(path, "r") as h5:
                label, pid, image, _, _ = _read_h5_sample(h5)
            rows.append(SampleRecord(str(num), num, "", "", str(path), pid, label, int(image.shape[0]), int(image.shape[1])))
    else:
        zips: dict[str, Path] = info["zips"]  # type: ignore[assignment]
        missing_archives = [name for name in EXPECTED_ARCHIVES if name not in zips]
        if missing_archives:
            raise FileNotFoundError(f"Missing expected archives: {missing_archives}")
        members: list[tuple[int, Path, str]] = []
        for archive_name in EXPECTED_ARCHIVES:
            archive_path = zips[archive_name]
            for num, member in _iter_zip_members(archive_path):
                members.append((num, archive_path, member))
        for num, archive_path, member in sorted(members)[:max_samples]:
            with zipfile.ZipFile(archive_path) as zf:
                payload = zf.read(member)
            with h5py.File(io.BytesIO(payload), "r") as h5:
                label, pid, image, _, _ = _read_h5_sample(h5)
            rows.append(SampleRecord(str(num), num, archive_path.name, member, "", pid, label, int(image.shape[0]), int(image.shape[1])))
    return pd.DataFrame([r.__dict__ for r in rows]).sort_values("file_number").reset_index(drop=True)


class BrainTumorDataset:
    """Lazy dataset reader supporting flat ZIP archives and extracted .mat files."""

    def __init__(self, data_dir: str | Path, metadata: pd.DataFrame | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.info = discover_dataset(data_dir)
        self.metadata = metadata if metadata is not None else build_metadata(data_dir)
        self._zip_paths: dict[str, Path] = self.info["zips"]  # type: ignore[assignment]

    def read_sample(self, file_number: int) -> dict[str, object]:
        """Read a sample by its numeric file id."""
        row = self.metadata.loc[self.metadata["file_number"] == file_number]
        if row.empty:
            raise KeyError(f"Unknown file_number {file_number}")
        rec = row.iloc[0]
        if rec.get("extracted_path", ""):
            with h5py.File(rec["extracted_path"], "r") as h5:
                label, pid, image, mask, border = _read_h5_sample(h5)
        else:
            archive = self._zip_paths[rec["archive_name"]]
            with zipfile.ZipFile(archive) as zf:
                payload = zf.read(rec["member_name"])
            with h5py.File(io.BytesIO(payload), "r") as h5:
                label, pid, image, mask, border = _read_h5_sample(h5)
        return {"file_number": file_number, "label": label, "patient_id": pid, "image": image, "tumorMask": mask, "tumorBorder": border}


def load_cvind(data_dir: str | Path) -> np.ndarray:
    """Load official cvind from MATLAB v7.3 HDF5 file."""
    path = Path(data_dir) / "cvind.mat"
    with h5py.File(path, "r") as h5:
        if "cvind" not in h5:
            raise ValueError("cvind.mat does not contain cvind")
        return np.asarray(h5["cvind"][()]).astype(int).ravel()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file for reproducibility manifests."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

