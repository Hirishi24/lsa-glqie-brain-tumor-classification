"""Batched configurable-qubit PyTorch state-vector simulator."""

from __future__ import annotations

import math
from itertools import combinations

import torch


def initial_state(batch: int, device: str = "cpu", dtype: torch.dtype = torch.complex128, num_qubits: int = 6) -> torch.Tensor:
    """Create batched all-zero computational basis states."""
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be positive, got {num_qubits}")
    state = torch.zeros((batch, 2**num_qubits), dtype=dtype, device=device)
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


def _infer_num_qubits(state: torch.Tensor) -> int:
    dim = int(state.shape[1])
    num_qubits = int(math.log2(dim))
    if 2**num_qubits != dim:
        raise ValueError(f"State dimension must be a power of two, got {dim}")
    return num_qubits


def apply_one_qubit_gate(state: torch.Tensor, gate: torch.Tensor, qubit: int, num_qubits: int | None = None) -> torch.Tensor:
    """Apply a batched one-qubit gate to a state vector."""
    batch = state.shape[0]
    num_qubits = _infer_num_qubits(state) if num_qubits is None else num_qubits
    if not 0 <= qubit < num_qubits:
        raise ValueError(f"qubit index {qubit} out of range for {num_qubits} qubits")
    tensor = state.reshape(batch, *([2] * num_qubits))
    tensor = tensor.movedim(qubit + 1, 1)
    flat = tensor.reshape(batch, 2, 2 ** (num_qubits - 1))
    updated = torch.einsum("bij,bjk->bik", gate, flat)
    updated = updated.reshape(batch, 2, *tensor.shape[2:]).movedim(1, qubit + 1)
    return updated.reshape(batch, 2**num_qubits)


def apply_cnot(state: torch.Tensor, control: int, target: int, num_qubits: int | None = None) -> torch.Tensor:
    """Apply CNOT by permuting basis amplitudes."""
    num_qubits = _infer_num_qubits(state) if num_qubits is None else num_qubits
    if control == target:
        raise ValueError("control and target qubits must differ")
    if not 0 <= control < num_qubits or not 0 <= target < num_qubits:
        raise ValueError(f"CNOT indices {(control, target)} out of range for {num_qubits} qubits")
    idx = torch.arange(2**num_qubits, device=state.device)
    control_bit = num_qubits - 1 - control
    target_bit = num_qubits - 1 - target
    flipped = idx ^ (1 << target_bit)
    mask = ((idx >> control_bit) & 1).bool()
    perm = torch.where(mask, flipped, idx)
    return state[:, perm]


def ring_cnot(state: torch.Tensor, num_qubits: int | None = None) -> torch.Tensor:
    """Apply ring entanglement q -> q+1, with the final qubit targeting q0."""
    num_qubits = _infer_num_qubits(state) if num_qubits is None else num_qubits
    for c in range(num_qubits):
        state = apply_cnot(state, c, (c + 1) % num_qubits, num_qubits)
    return state


def expectations(state: torch.Tensor, num_qubits: int | None = None) -> torch.Tensor:
    """Compute all single-Z and pairwise-ZZ expectations."""
    num_qubits = _infer_num_qubits(state) if num_qubits is None else num_qubits
    probs = (state.abs() ** 2).real
    idx = torch.arange(2**num_qubits, device=state.device)
    signs = []
    for q in range(num_qubits):
        bit = num_qubits - 1 - q
        signs.append(torch.where(((idx >> bit) & 1).bool(), -1.0, 1.0).to(probs.dtype))
    z_cols = []
    for z in signs:
        z_cols.append((probs * z).sum(dim=1))
    for a, b in combinations(range(num_qubits), 2):
        z_cols.append((probs * signs[a] * signs[b]).sum(dim=1))
    return torch.stack(z_cols, dim=1)


def validate_state(state: torch.Tensor, num_qubits: int | None = None, atol: float = 1e-8) -> None:
    """Validate shape, finite values and state norms."""
    num_qubits = _infer_num_qubits(state) if num_qubits is None else num_qubits
    expected_dim = 2**num_qubits
    if state.shape[1] != expected_dim:
        raise ValueError(f"Expected state dimension {expected_dim}, got {state.shape}")
    if not torch.isfinite(state.real).all() or not torch.isfinite(state.imag).all():
        raise ValueError("State contains NaN or infinity")
    norms = (state.abs() ** 2).sum(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=atol):
        raise ValueError("State norm drifted from one")


def encode_angles(angles: torch.Tensor, gate: str = "RY_RZ", rounds: int = 2, num_qubits: int = 6) -> torch.Tensor:
    """Run the reuploading circuit and return expectation features."""
    expected_angles = num_qubits * rounds
    if angles.ndim != 2 or angles.shape[1] != expected_angles:
        raise ValueError(f"Expected angles shape (batch, {expected_angles}), got {tuple(angles.shape)}")
    angles = angles.to(dtype=torch.float64)
    gate_name = gate.upper().replace("-", "_")
    if gate_name not in {"RY", "RZ", "RY_RZ", "RYRZ"}:
        raise ValueError(f"Unsupported encoding gate {gate}")
    state = initial_state(angles.shape[0], str(angles.device), torch.complex128, num_qubits)
    for r in range(rounds):
        for q in range(num_qubits):
            theta = angles[:, r * num_qubits + q]
            if gate_name in {"RY", "RY_RZ", "RYRZ"}:
                state = apply_one_qubit_gate(state, ry_matrix(theta), q, num_qubits)
            if gate_name in {"RZ", "RY_RZ", "RYRZ"}:
                state = apply_one_qubit_gate(state, rz_matrix(theta), q, num_qubits)
        state = ring_cnot(state, num_qubits)
        validate_state(state, num_qubits)
    exp = expectations(state, num_qubits)
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


def logical_resource_counts(rounds: int = 2, num_qubits: int = 6, gate: str = "RY_RZ") -> dict[str, int]:
    """Return logical circuit resource counts for the configured feature map."""
    gate_name = gate.upper().replace("-", "_")
    uses_ry = gate_name in {"RY", "RY_RZ", "RYRZ"}
    uses_rz = gate_name in {"RZ", "RY_RZ", "RYRZ"}
    rotation_layers = int(uses_ry) + int(uses_rz)
    return {
        "num_qubits": int(num_qubits),
        "ry_gates": int(num_qubits * rounds * int(uses_ry)),
        "rz_gates": int(num_qubits * rounds * int(uses_rz)),
        "cnot_gates": int(num_qubits * rounds),
        "measurements": int(num_qubits + (num_qubits * (num_qubits - 1)) // 2),
        "nominal_depth": int(rounds * (rotation_layers + 1)),
    }
