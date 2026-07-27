"""Quantum gate tests."""

from __future__ import annotations

import torch

from src.quantum_simulator import apply_cnot, initial_state, ry_matrix, validate_state


def test_ry_gate_unitarity():
    theta = torch.tensor([0.7], dtype=torch.float64)
    ry = ry_matrix(theta)[0]
    eye = ry.conj().T @ ry
    assert torch.allclose(eye, torch.eye(2, dtype=torch.complex128), atol=1e-10)


def test_cnot_norm_preservation():
    state = initial_state(2)
    state = apply_cnot(state, 0, 1, num_qubits=6)
    validate_state(state)
