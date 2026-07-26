"""Quantum encoding tests."""

from __future__ import annotations

import numpy as np
import torch

from src.quantum_encoding import quantum_features
from src.quantum_simulator import encode_angles, initial_state, validate_state


def test_four_qubit_state_shape():
    state = initial_state(5)
    assert state.shape == (5, 16)
    validate_state(state)


def test_batched_simulator_output_shape_and_finite():
    angles = torch.zeros((7, 8), dtype=torch.float64)
    out = encode_angles(angles)
    assert out.shape == (7, 8)
    assert torch.isfinite(out).all()


def test_exact_repeated_runs_are_deterministic():
    angles = np.ones((4, 8), dtype=np.float64) * 0.2
    a = quantum_features(angles, "cpu", 2, shots=0, seed=99)
    b = quantum_features(angles, "cpu", 2, shots=0, seed=99)
    assert np.allclose(a, b)


def test_shot_simulation_is_deterministic():
    angles = np.ones((4, 8), dtype=np.float64) * 0.2
    a = quantum_features(angles, "cpu", 2, shots=128, seed=99)
    b = quantum_features(angles, "cpu", 2, shots=128, seed=99)
    assert np.allclose(a, b)

