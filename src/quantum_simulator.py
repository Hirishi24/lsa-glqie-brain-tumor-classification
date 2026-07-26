"""Batched four-qubit PyTorch state-vector simulator."""

from __future__ import annotations

import math

import torch


def initial_state(batch: int, device: str = "cpu", dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Create batched |0000> states."""
    state = torch.zeros((batch, 16), dtype=dtype, device=device)
    state[:, 0] = 1.0 + 0.0j
    return state


def ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Return batched RY matrices."""
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    mat = torch.zeros((theta.shape[0], 2, 2), dtype=torch.complex128, device=theta.device)
    mat[:, 0, 0] = c
    mat[:, 0, 1] = -s
    mat[:, 1, 0] = s
    mat[:, 1, 1] = c
    return mat


def rz_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Return batched RZ matrices."""
    mat = torch.zeros((theta.shape[0], 2, 2), dtype=torch.complex128, device=theta.device)
    mat[:, 0, 0] = torch.exp(-0.5j * theta)
    mat[:, 1, 1] = torch.exp(0.5j * theta)
    return mat


def apply_one_qubit_gate(state: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
    """Apply a batched one-qubit gate to a four-qubit state."""
    batch = state.shape[0]
    tensor = state.reshape(batch, 2, 2, 2, 2)
    tensor = tensor.movedim(qubit + 1, 1)
    flat = tensor.reshape(batch, 2, 8)
    updated = torch.einsum("bij,bjk->bik", gate, flat)
    updated = updated.reshape(batch, 2, *tensor.shape[2:]).movedim(1, qubit + 1)
    return updated.reshape(batch, 16)


def apply_cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    """Apply CNOT by permuting basis amplitudes."""
    idx = torch.arange(16, device=state.device)
    control_bit = 3 - control
    target_bit = 3 - target
    flipped = idx ^ (1 << target_bit)
    mask = ((idx >> control_bit) & 1).bool()
    perm = torch.where(mask, flipped, idx)
    return state[:, perm]


def ring_cnot(state: torch.Tensor) -> torch.Tensor:
    """Apply ring entanglement 0->1->2->3->0."""
    for c, t in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        state = apply_cnot(state, c, t)
    return state


def expectations(state: torch.Tensor) -> torch.Tensor:
    """Compute Z and nearest-neighbour ZZ expectations."""
    probs = (state.abs() ** 2).real
    idx = torch.arange(16, device=state.device)
    z_cols = []
    for q in range(4):
        bit = 3 - q
        z = torch.where(((idx >> bit) & 1).bool(), -1.0, 1.0).to(probs.dtype)
        z_cols.append((probs * z).sum(dim=1))
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        za = torch.where(((idx >> (3 - a)) & 1).bool(), -1.0, 1.0).to(probs.dtype)
        zb = torch.where(((idx >> (3 - b)) & 1).bool(), -1.0, 1.0).to(probs.dtype)
        z_cols.append((probs * za * zb).sum(dim=1))
    return torch.stack(z_cols, dim=1)


def validate_state(state: torch.Tensor, atol: float = 1e-8) -> None:
    """Validate shape, finite values and state norms."""
    if state.shape[1] != 16:
        raise ValueError(f"Expected state dimension 16, got {state.shape}")
    if not torch.isfinite(state.real).all() or not torch.isfinite(state.imag).all():
        raise ValueError("State contains NaN or infinity")
    norms = (state.abs() ** 2).sum(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=atol):
        raise ValueError("State norm drifted from one")


def encode_angles(angles: torch.Tensor, gate: str = "RY", rounds: int = 2) -> torch.Tensor:
    """Run the fixed four-qubit reuploading circuit and return expectations."""
    if angles.ndim != 2 or angles.shape[1] != 4 * rounds:
        raise ValueError(f"Expected angles shape (batch, {4 * rounds}), got {tuple(angles.shape)}")
    angles = angles.to(dtype=torch.float64)
    state = initial_state(angles.shape[0], str(angles.device), torch.complex128)
    for r in range(rounds):
        for q in range(4):
            theta = angles[:, r * 4 + q]
            mat = ry_matrix(theta) if gate.upper() == "RY" else rz_matrix(theta)
            state = apply_one_qubit_gate(state, mat, q)
        state = ring_cnot(state)
        validate_state(state)
    exp = expectations(state)
    if not torch.isfinite(exp).all():
        raise ValueError("Expectation features are non-finite")
    return exp


def add_shot_noise(expectation: torch.Tensor, shots: int, seed: int) -> torch.Tensor:
    """Convert exact expectations to deterministic finite-shot estimates."""
    if shots <= 0:
        return expectation
    gen = torch.Generator(device=expectation.device).manual_seed(seed)
    p = torch.clamp((expectation + 1.0) / 2.0, 0.0, 1.0)
    counts = torch.binomial(torch.full_like(p, float(shots)), p, generator=gen)
    return 2.0 * counts / float(shots) - 1.0


def add_effective_noise(expectation: torch.Tensor, noise_level: float, seed: int) -> torch.Tensor:
    """Apply lightweight expectation-space noise."""
    if noise_level <= 0:
        return expectation
    gen = torch.Generator(device=expectation.device).manual_seed(seed)
    perturb = torch.randn(expectation.shape, generator=gen, device=expectation.device, dtype=expectation.dtype) * noise_level
    return torch.clamp((1.0 - noise_level) * expectation + perturb, -1.0, 1.0)


def logical_resource_counts(rounds: int = 2) -> dict[str, int]:
    """Return fixed logical circuit resource counts."""
    return {"num_qubits": 4, "ry_gates": 4 * rounds, "cnot_gates": 4 * rounds, "measurements": 8, "nominal_depth": 2 * rounds}

