"""Deep quantum fusion tests."""

from __future__ import annotations

import torch

from src.deep_quantum_fusion import DeepQuantumConfig, QuantumFusionHead


def test_quantum_fusion_head_forward_backward():
    cfg = DeepQuantumConfig(num_qubits=3, reuploading_rounds=1, vqc_layers=1, hidden_dim=8, batch_size=2)
    model = QuantumFusionHead(input_dim=12, cfg=cfg)
    x = torch.randn(4, 12)
    y = torch.tensor([0, 1, 2, 1])
    logits = model(x)
    assert logits.shape == (4, 3)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    assert model.theta.grad is not None
    assert torch.isfinite(logits).all()
