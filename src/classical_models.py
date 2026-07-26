"""Classical classifiers used by all encodings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.svm import SVC


@dataclass
class TorchLogisticResult:
    """Trained logistic regression result."""

    model: torch.nn.Module
    history: list[dict[str, float]]
    weight_decay: float


def _class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float64, device=device)


def fit_torch_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: str,
    cfg: dict,
    seed: int,
) -> TorchLogisticResult:
    """Fit multinomial logistic regression with validation-based L2 selection."""
    torch.manual_seed(seed)
    xtr = torch.as_tensor(x_train, dtype=torch.float64, device=device)
    ytr = torch.as_tensor(y_train.astype(int), dtype=torch.long, device=device)
    xva = torch.as_tensor(x_val, dtype=torch.float64, device=device)
    yva = torch.as_tensor(y_val.astype(int), dtype=torch.long, device=device)
    best_result: TorchLogisticResult | None = None
    best_loss = float("inf")
    for wd in cfg["weight_decay_candidates"]:
        model = torch.nn.Linear(x_train.shape[1], 3, dtype=torch.float64).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(wd))
        weights = _class_weights(y_train, device) if cfg.get("class_weighting", True) else None
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        history = []
        patience = int(cfg["early_stopping_patience"])
        wait = 0
        local_best = float("inf")
        local_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        max_epochs = int(cfg["max_epochs"])
        for epoch in range(max_epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xtr), ytr)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(xva), yva).detach().cpu()) if len(y_val) else float(loss.detach().cpu())
            history.append({"epoch": float(epoch), "train_loss": float(loss.detach().cpu()), "val_loss": val_loss})
            if val_loss + 1e-9 < local_best:
                local_best = val_loss
                local_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break
        model.load_state_dict(local_state)
        if local_best < best_loss:
            best_loss = local_best
            best_result = TorchLogisticResult(model=model, history=history, weight_decay=float(wd))
    if best_result is None:
        raise RuntimeError("No logistic regression model was trained")
    return best_result


def predict_proba_torch(model: torch.nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    """Predict class probabilities with a trained PyTorch model."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float64, device=device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def fit_predict_svm(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> tuple[SVC, np.ndarray]:
    """Fit an RBF SVM and return model plus probabilities."""
    model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    model.fit(x_train, y_train)
    return model, model.predict_proba(x_test)

