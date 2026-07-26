"""Synthetic HDF5 MATLAB-style smoke-test dataset."""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import h5py
import numpy as np

from .dataset_io import EXPECTED_ARCHIVES


def _matlab_char_array(text: str) -> np.ndarray:
    return np.asarray([ord(c) for c in text], dtype=np.uint16).reshape(-1, 1)


def _ellipse_mask(size: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    return (((y - cy) / max(ry, 1)) ** 2 + ((x - cx) / max(rx, 1)) ** 2 <= 1).astype(np.uint8)


def _border_from_mask(mask: np.ndarray, max_points: int = 32) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    cy, cx = ys.mean(), xs.mean()
    angles = np.linspace(0, 2 * math.pi, max_points // 2, endpoint=False)
    coords = []
    for a in angles:
        proj = (xs - cx) * math.cos(a) + (ys - cy) * math.sin(a)
        idx = int(np.argmax(proj))
        coords.extend([float(xs[idx]), float(ys[idx])])
    return np.asarray(coords, dtype=np.float64).reshape(1, -1)


def _write_mat_bytes(label: int, pid: str, image: np.ndarray, mask: np.ndarray, border: np.ndarray) -> bytes:
    bio = io.BytesIO()
    with h5py.File(bio, "w") as h5:
        g = h5.create_group("cjdata")
        g.attrs["MATLAB_class"] = np.bytes_("struct")
        d = g.create_dataset("label", data=np.asarray([[label]], dtype=np.float64))
        d.attrs["MATLAB_class"] = np.bytes_("double")
        p = g.create_dataset("PID", data=_matlab_char_array(pid))
        p.attrs["MATLAB_class"] = np.bytes_("char")
        p.attrs["MATLAB_int_decode"] = np.int32(2)
        im = g.create_dataset("image", data=image.astype(np.int16))
        im.attrs["MATLAB_class"] = np.bytes_("int16")
        mk = g.create_dataset("tumorMask", data=mask.astype(np.uint8))
        mk.attrs["MATLAB_class"] = np.bytes_("logical")
        tb = g.create_dataset("tumorBorder", data=border.astype(np.float64))
        tb.attrs["MATLAB_class"] = np.bytes_("double")
    return bio.getvalue()


def create_synthetic_dataset(output_dir: str | Path, seed: int = 11, patients_per_class: int = 6, slices_per_patient: int = 4) -> Path:
    """Create a synthetic Figshare-like dataset with four ZIP archives and cvind.mat."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples: list[tuple[int, bytes]] = []
    file_number = 1
    for label in [1, 2, 3]:
        for patient in range(patients_per_class):
            pid = f"SYN{label}{patient:03d}"
            for slice_id in range(slices_per_patient):
                size = 128 if (patient + slice_id) % 7 else 96
                yy, xx = np.mgrid[:size, :size]
                image = 80 + 12 * rng.normal(size=(size, size)) + 30 * np.sin(xx / 9.0) + 15 * np.cos(yy / 11.0)
                lesion_scale = [0.055, 0.105, 0.17][(patient + slice_id) % 3]
                ry = size * lesion_scale * (1.0 + 0.15 * label)
                rx = size * lesion_scale * (1.2 + 0.05 * slice_id)
                cy = size * (0.35 + 0.1 * label) + rng.normal(0, 2)
                cx = size * (0.32 + 0.08 * patient) % (size * 0.7) + size * 0.15
                mask = _ellipse_mask(size, cy, cx, ry, rx)
                image += mask * (80 + label * 45)
                image = np.clip(image, 0, 1200)
                border = _border_from_mask(mask)
                samples.append((file_number, _write_mat_bytes(label, pid, image, mask, border)))
                file_number += 1
    chunk_size = int(math.ceil(len(samples) / 4))
    chunks = [samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)]
    for archive_name, chunk in zip(EXPECTED_ARCHIVES, chunks):
        with zipfile.ZipFile(root / archive_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for num, payload in chunk:
                zf.writestr(f"{num}.mat", payload)
    cvind = (np.arange(len(samples)) % 5 + 1).astype(np.float64).reshape(1, -1)
    with h5py.File(root / "cvind.mat", "w") as h5:
        d = h5.create_dataset("cvind", data=cvind)
        d.attrs["MATLAB_class"] = np.bytes_("double")
    (root / "README 2024.txt").write_text("Synthetic smoke-test dataset for LSA-GLQIE.\n", encoding="utf-8")
    return root
