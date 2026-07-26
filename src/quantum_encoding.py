"""Scaling and quantum feature generation."""

from __future__ import annotations

import numpy as np
import torch

from .quantum_simulator import add_effective_noise, add_shot_noise, encode_angles


def fit_robust_scaler(x: np.ndarray) -> dict[str, np.ndarray]:
    """Fit median and IQR on training data."""
    median = np.median(x, axis=0)
    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    return {"median": median, "iqr": q3 - q1}


def transform_robust(x: np.ndarray, scaler: dict[str, np.ndarray], clip: float = 3.0) -> np.ndarray:
    """Robust-scale, clip and map coefficients to rotation angles."""
    scaled = (x - scaler["median"]) / (scaler["iqr"] + 1e-8)
    scaled = np.clip(scaled, -clip, clip)
    return (np.pi * np.tanh(scaled)).astype(np.float64)


def quantum_features(angles: np.ndarray, device: str, batch_size: int, shots: int = 0, noise_level: float = 0.0, seed: int = 0, rounds: int = 2) -> np.ndarray:
    """Generate quantum expectation features in batches."""
    outputs = []
    with torch.no_grad():
        for start in range(0, len(angles), batch_size):
            batch = torch.as_tensor(angles[start : start + batch_size], dtype=torch.float64, device=device)
            exp = encode_angles(batch, rounds=rounds)
            exp = add_shot_noise(exp, shots, seed + start)
            exp = add_effective_noise(exp, noise_level, seed + start + 7919)
            outputs.append(exp.cpu().numpy())
    return np.vstack(outputs).astype(np.float64)

