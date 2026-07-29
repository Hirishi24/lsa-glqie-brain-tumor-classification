"""Pretrained deep-feature extraction with a trainable quantum fusion head."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from .dataset_io import BrainTumorDataset, build_metadata
from .environment import save_environment, select_device
from .lesion_size import add_size_categories, lesion_features
from .metrics import aggregate_patient_predictions, compute_metrics
from .plotting import create_all_figures
from .preprocessing import preprocess_image_mask, resize_array
from .quantum_simulator import apply_one_qubit_gate, expectations, ring_cnot, ry_matrix, rz_matrix, validate_state
from .reproducibility import set_global_seed
from .roi_processing import expand_bbox, local_crop, tumor_bbox
from .splitting import patient_wise_folds


@dataclass(frozen=True)
class DeepQuantumConfig:
    """Configuration for the deep-feature plus quantum-head experiment."""

    backbone: str = "efficientnet_b0"
    weights: str = "imagenet"
    image_size: int = 224
    num_qubits: int = 6
    reuploading_rounds: int = 2
    vqc_layers: int = 2
    hidden_dim: int = 128
    dropout: float = 0.25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 120
    patience: int = 18
    batch_size: int = 128
    feature_batch_size: int = 64
    folds: int = 5
    seed: int = 11
    quick_max_samples: int = 0
    lesion_quantiles: tuple[float, float] = (0.333333, 0.666667)
    plot_dpi: int = 300
    plot_png_min_width: int = 1600
    plot_png_min_height: int = 1000
    plot_title_font_size: int = 18
    plot_axis_font_size: int = 15
    plot_tick_font_size: int = 12
    plot_legend_font_size: int = 12

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DeepQuantumConfig":
        """Load config from YAML while requiring every key used by the runner."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        required = set(cls.__dataclass_fields__)
        missing = sorted(required - set(data))
        extra = sorted(set(data) - required)
        if missing:
            raise ValueError(f"Missing required deep quantum config keys: {missing}")
        if extra:
            raise ValueError(f"Unknown deep quantum config keys: {extra}")
        data["lesion_quantiles"] = tuple(float(x) for x in data["lesion_quantiles"])
        return cls(**data)


def _plot_config(cfg: DeepQuantumConfig) -> dict[str, Any]:
    return {
        "features": {"total_coefficients": int(cfg.num_qubits * cfg.reuploading_rounds)},
        "quantum": {
            "num_qubits": int(cfg.num_qubits),
            "reuploading_rounds": int(cfg.reuploading_rounds),
            "encoding_gate": "RY_RZ",
        },
        "plots": {
            "dpi": int(cfg.plot_dpi),
            "png_min_width": int(cfg.plot_png_min_width),
            "png_min_height": int(cfg.plot_png_min_height),
            "title_font_size": int(cfg.plot_title_font_size),
            "axis_font_size": int(cfg.plot_axis_font_size),
            "tick_font_size": int(cfg.plot_tick_font_size),
            "legend_font_size": int(cfg.plot_legend_font_size),
            "primary_method": "deep_quantum_fusion",
            "workflow_feature_node": "Pretrained multi-view CNN features",
            "workflow_feature_top": "Full / ROI\nCNN",
            "workflow_feature_bottom": "Context +\ngeometry",
        },
    }


def _load_torchvision_backbone(backbone: str, weights: str, device: str) -> tuple[torch.nn.Module, int]:
    try:
        from torchvision import models
    except Exception as exc:
        raise ImportError("torchvision is required for pretrained feature extraction. Install it with `pip install torchvision`.") from exc

    use_weights = weights.lower() not in {"none", "random", "false", "0"}
    name = backbone.lower()
    if name == "efficientnet_b0":
        enum = models.EfficientNet_B0_Weights.DEFAULT if use_weights else None
        model = models.efficientnet_b0(weights=enum)
        extractor = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten())
        feature_dim = 1280
    elif name == "resnet50":
        enum = models.ResNet50_Weights.DEFAULT if use_weights else None
        model = models.resnet50(weights=enum)
        extractor = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
        feature_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone {backbone!r}; use efficientnet_b0 or resnet50")
    extractor.eval().to(device)
    for param in extractor.parameters():
        param.requires_grad = False
    return extractor, feature_dim


def _imagenet_normalize(batch: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(batch, dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _as_three_channel(image: np.ndarray) -> np.ndarray:
    x = np.clip(image.astype(np.float32), 0.0, 1.0)
    return np.repeat(x[None, :, :], 3, axis=0)


def _crop_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], size: int) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return resize_array(image[y0:y1, x0:x1], size, order=1).astype(np.float32)


def _peritumoral_crop(image: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    bbox = tumor_bbox(mask)
    outer = expand_bbox(bbox, image.shape, margin=0.65)
    dilated = binary_dilation(mask > 0, iterations=max(3, image.shape[0] // 48))
    ring = np.logical_and(dilated, ~(mask > 0))
    emphasized = image.copy()
    emphasized[ring] = np.maximum(emphasized[ring], 0.85)
    return _crop_bbox(emphasized, outer, size)


def _sample_views(sample: dict[str, object], cfg: DeepQuantumConfig) -> tuple[list[np.ndarray], dict[str, float]]:
    image, mask = preprocess_image_mask(
        np.asarray(sample["image"]),
        np.asarray(sample["tumorMask"]),
        cfg.image_size,
        1.0,
        99.0,
    )
    full = image
    roi = local_crop(image, mask, cfg.image_size, margin=0.08)
    context = _peritumoral_crop(image, mask, cfg.image_size)
    geo = lesion_features(mask)
    return [full, roi, context], geo


def _metadata_subset(metadata: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(metadata):
        return metadata.reset_index(drop=True)
    per_class = max(1, max_samples // int(metadata["class_label"].nunique()))
    chunks = []
    for _, group in metadata.groupby("class_label"):
        chunks.append(group.sample(n=min(per_class, len(group)), random_state=seed))
    return pd.concat(chunks).sort_values("file_number").reset_index(drop=True)


def extract_deep_features(data_dir: str | Path, output_dir: Path, cfg: DeepQuantumConfig, device: str, resume: bool = True) -> dict[str, Any]:
    """Extract and cache full/ROI/context pretrained CNN features."""
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"deep_features_{cfg.backbone}_{cfg.weights}_{cfg.image_size}_{cfg.quick_max_samples or 'full'}.joblib"
    if resume and cache_path.exists():
        return joblib.load(cache_path)

    metadata = _metadata_subset(build_metadata(data_dir), cfg.quick_max_samples, cfg.seed)
    dataset = BrainTumorDataset(data_dir, metadata)
    extractor, feature_dim = _load_torchvision_backbone(cfg.backbone, cfg.weights, device)
    features = np.zeros((len(metadata), 3, feature_dim), dtype=np.float32)
    geometry_names = ["tumor_area_ratio", "bounding_box_ratio", "equivalent_diameter", "compactness"]
    geometry = np.zeros((len(metadata), len(geometry_names)), dtype=np.float32)
    labels = metadata["class_label"].to_numpy(dtype=int) - 1
    file_numbers = metadata["file_number"].to_numpy(dtype=int)
    patient_ids = metadata["patient_id"].astype(str).to_numpy()

    pending: list[np.ndarray] = []
    pending_refs: list[tuple[int, int]] = []
    with torch.no_grad():
        for i, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting pretrained views"):
            sample = dataset.read_sample(int(row["file_number"]))
            views, geo = _sample_views(sample, cfg)
            geometry[int(i)] = np.nan_to_num(np.asarray([float(geo[name]) for name in geometry_names], dtype=np.float32))
            for view_idx, view in enumerate(views):
                pending.append(_as_three_channel(view))
                pending_refs.append((int(i), view_idx))
                if len(pending) >= cfg.feature_batch_size:
                    batch = _imagenet_normalize(np.stack(pending)).to(device)
                    emb = extractor(batch).detach().cpu().numpy().astype(np.float32)
                    for (sample_idx, sample_view), vec in zip(pending_refs, emb):
                        features[sample_idx, sample_view] = vec
                    pending.clear()
                    pending_refs.clear()
        if pending:
            batch = _imagenet_normalize(np.stack(pending)).to(device)
            emb = extractor(batch).detach().cpu().numpy().astype(np.float32)
            for (sample_idx, sample_view), vec in zip(pending_refs, emb):
                features[sample_idx, sample_view] = vec

    payload = {
        "features": features,
        "geometry": geometry,
        "geometry_names": geometry_names,
        "labels": labels,
        "file_numbers": file_numbers,
        "patient_ids": patient_ids,
        "metadata": metadata,
        "feature_dim": feature_dim,
        "config": asdict(cfg),
    }
    for col_idx, name in enumerate(geometry_names):
        metadata[name] = geometry[:, col_idx]
    payload["metadata"] = metadata
    joblib.dump(payload, cache_path)
    return payload


class QuantumFusionHead(torch.nn.Module):
    """Trainable VQC head over compact pretrained image embeddings."""

    def __init__(self, input_dim: int, cfg: DeepQuantumConfig) -> None:
        super().__init__()
        self.num_qubits = int(cfg.num_qubits)
        self.rounds = int(cfg.reuploading_rounds)
        self.vqc_layers = int(cfg.vqc_layers)
        self.projector = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Dropout(float(cfg.dropout)),
            torch.nn.Linear(input_dim, int(cfg.hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(cfg.hidden_dim), self.num_qubits * self.rounds),
        )
        self.theta = torch.nn.Parameter(0.02 * torch.randn(self.vqc_layers, self.num_qubits, 2, dtype=torch.float64))
        measurement_dim = self.num_qubits + (self.num_qubits * (self.num_qubits - 1)) // 2
        self.temperature = torch.nn.Parameter(torch.tensor(4.0, dtype=torch.float64))
        self.classifier = torch.nn.Linear(measurement_dim, 3, dtype=torch.float64)

    def _initial_state(self, batch: int, device: torch.device) -> torch.Tensor:
        state = torch.zeros((batch, 2**self.num_qubits), dtype=torch.complex128, device=device)
        state[:, 0] = 1.0 + 0.0j
        return state

    def _vqc(self, angles: torch.Tensor) -> torch.Tensor:
        state = self._initial_state(angles.shape[0], angles.device)
        for r in range(self.rounds):
            for q in range(self.num_qubits):
                angle = angles[:, r * self.num_qubits + q]
                state = apply_one_qubit_gate(state, ry_matrix(angle), q, self.num_qubits)
                state = apply_one_qubit_gate(state, rz_matrix(angle), q, self.num_qubits)
            state = ring_cnot(state, self.num_qubits)
            layer = min(r, self.vqc_layers - 1)
            for q in range(self.num_qubits):
                ry = self.theta[layer, q, 0].expand(angles.shape[0])
                rz = self.theta[layer, q, 1].expand(angles.shape[0])
                state = apply_one_qubit_gate(state, ry_matrix(ry), q, self.num_qubits)
                state = apply_one_qubit_gate(state, rz_matrix(rz), q, self.num_qubits)
            state = ring_cnot(state, self.num_qubits)
            validate_state(state, self.num_qubits, atol=1e-6)
        return expectations(state, self.num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = torch.pi * torch.tanh(self.projector(x.float()).to(torch.float64))
        measurements = self._vqc(angles)
        return self.classifier(measurements * torch.clamp(self.temperature, 0.5, 20.0))


def _class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float64, device=device)


def _standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return tuple(((x - mean) / std).astype(np.float32) for x in (train, *others))


def _predict(model: torch.nn.Module, x: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.vstack(out)


def _fit_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DeepQuantumConfig,
    device: str,
) -> tuple[QuantumFusionHead, list[dict[str, float]]]:
    model = QuantumFusionHead(x_train.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=_class_weights(y_train, device))
    xtr = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_train, dtype=torch.long, device=device)
    xva = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    yva = torch.as_tensor(y_val, dtype=torch.long, device=device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_loss = float("inf")
    wait = 0
    history = []
    for epoch in range(cfg.max_epochs):
        model.train()
        order = torch.randperm(len(xtr), device=device)
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xva), yva).detach().cpu()) if len(xva) else float(np.mean(losses))
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, history


def run_deep_quantum_experiment(data_dir: str | Path, output_dir: str | Path, cfg: DeepQuantumConfig, device_request: str = "auto", resume: bool = True) -> dict[str, Any]:
    """Run patient-wise pretrained-feature + quantum-head classification."""
    start = time.time()
    output = Path(output_dir)
    for sub in ["cache", "configs", "models", "predictions", "tables", "reproducibility"]:
        (output / sub).mkdir(parents=True, exist_ok=True)
    device = select_device(device_request)
    set_global_seed(cfg.seed)
    save_environment(output, device)
    (output / "configs" / "deep_quantum_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    payload = extract_deep_features(data_dir, output, cfg, device, resume=resume)
    cnn_features = payload["features"].reshape(len(payload["labels"]), -1)
    geometry = payload["geometry"].astype(np.float32)
    features = np.concatenate([cnn_features, geometry], axis=1)
    y = payload["labels"].astype(int)
    metadata = payload["metadata"].copy().reset_index(drop=True)
    folds = patient_wise_folds(metadata, cfg.folds, cfg.seed)

    fold_rows = []
    slice_predictions = []
    patient_predictions = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
        x_train, x_val, x_test = _standardize(features[train_idx], features[val_idx], features[test_idx])
        model, history = _fit_fold(x_train, y[train_idx], x_val, y[val_idx], cfg, device)
        fold_dir = output / "models" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fold_dir / "quantum_head.pt")
        pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
        probs = _predict(model, x_test, device, cfg.batch_size)
        test_meta = metadata.iloc[test_idx].copy().reset_index(drop=True)
        slice_df = pd.DataFrame(
            {
                "seed": cfg.seed,
                "fold": fold,
                "method": "deep_quantum_fusion",
                "file_number": test_meta["file_number"],
                "patient_id": test_meta["patient_id"],
                "true_label": test_meta["class_label"],
                "pred_label": probs.argmax(axis=1) + 1,
                "prob_class_1": probs[:, 0],
                "prob_class_2": probs[:, 1],
                "prob_class_3": probs[:, 2],
            }
        )
        patient_df = aggregate_patient_predictions(slice_df)
        patient_df.insert(0, "method", "deep_quantum_fusion")
        patient_df.insert(0, "fold", fold)
        patient_df.insert(0, "seed", cfg.seed)
        slice_metrics = compute_metrics(slice_df["true_label"].to_numpy() - 1, probs, "slice_")
        patient_probs = patient_df[["prob_class_1", "prob_class_2", "prob_class_3"]].to_numpy()
        patient_metrics = compute_metrics(patient_df["true_label"].to_numpy() - 1, patient_probs, "patient_")
        fold_rows.append({"seed": cfg.seed, "fold": fold, "method": "deep_quantum_fusion", **slice_metrics, **patient_metrics})
        slice_predictions.append(slice_df)
        patient_predictions.append(patient_df)

    fold_metrics = pd.DataFrame(fold_rows)
    pd.concat(slice_predictions, ignore_index=True).to_csv(output / "predictions" / "slice_predictions.csv", index=False)
    pd.concat(patient_predictions, ignore_index=True).to_csv(output / "predictions" / "patient_predictions.csv", index=False)
    fold_metrics.to_csv(output / "tables" / "deep_quantum_fold_metrics.csv", index=False)
    summary_rows = []
    metric_cols = [c for c in fold_metrics.columns if c.startswith("patient_") or c.startswith("slice_")]
    for metric in metric_cols:
        values = fold_metrics[metric].dropna().to_numpy(dtype=float)
        summary_rows.append(
            {
                "method": "deep_quantum_fusion",
                "metric": metric,
                "mean": float(values.mean()) if values.size else np.nan,
                "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "n_folds": int(values.size),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "tables" / "deep_quantum_summary.csv", index=False)
    metadata_with_sizes, _ = add_size_categories(metadata, np.ones(len(metadata), dtype=bool), list(cfg.lesion_quantiles))
    plot_validation = create_all_figures(
        output,
        _plot_config(cfg),
        metadata_with_sizes,
        fold_metrics,
        pd.concat(patient_predictions, ignore_index=True),
        data_dir,
    )
    report = {
        "output_dir": str(output),
        "device": device,
        "feature_shape": list(payload["features"].shape),
        "summary_table": str(output / "tables" / "deep_quantum_summary.csv"),
        "figure_count": int(len(plot_validation)),
        "total_seconds": time.time() - start,
    }
    (output / "reproducibility" / "runtime_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
