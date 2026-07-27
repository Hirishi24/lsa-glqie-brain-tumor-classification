"""Quantum encoding tests."""

from __future__ import annotations

import numpy as np
import torch

from src.quantum_encoding import quantum_features
from src.quantum_simulator import encode_angles, initial_state, validate_state


def test_six_qubit_state_shape():
    state = initial_state(5)
    assert state.shape == (5, 64)
    validate_state(state)


def test_batched_simulator_output_shape_and_finite():
    angles = torch.zeros((7, 12), dtype=torch.float64)
    out = encode_angles(angles)
    assert out.shape == (7, 21)
    assert torch.isfinite(out).all()


def test_exact_repeated_runs_are_deterministic():
    angles = np.ones((4, 12), dtype=np.float64) * 0.2
    a = quantum_features(angles, "cpu", 2, shots=0, seed=99)
    b = quantum_features(angles, "cpu", 2, shots=0, seed=99)
    assert np.allclose(a, b)


def test_shot_simulation_is_deterministic():
    angles = np.ones((4, 12), dtype=np.float64) * 0.2
    a = quantum_features(angles, "cpu", 2, shots=128, seed=99)
    b = quantum_features(angles, "cpu", 2, shots=128, seed=99)
    assert np.allclose(a, b)


def test_four_qubit_non_default_shape():
    state = initial_state(3, num_qubits=4)
    assert state.shape == (3, 16)
    out = encode_angles(torch.zeros((3, 8), dtype=torch.float64), num_qubits=4, gate="RY")
    assert out.shape == (3, 10)
