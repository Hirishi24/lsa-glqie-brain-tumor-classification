"""Configuration loading and CLI override helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"name": "LSA-GLQIE", "seeds": [11, 22, 33], "n_splits": 5, "repeats": 1, "primary_metric": "macro_f1"},
    "data": {
        "image_size": 128,
        "global_view_size": 64,
        "local_view_size": 64,
        "local_margin": 0.15,
        "percentile_clip_low": 1.0,
        "percentile_clip_high": 99.0,
        "cache_processed_data": True,
        "validate_every_sample": True,
    },
    "lesion_size": {
        "quantiles": [0.333333, 0.666667],
        "allocation_small": [3, 9],
        "allocation_medium": [6, 6],
        "allocation_large": [9, 3],
    },
    "features": {"method": "dct", "total_coefficients": 12, "exclude_dc": True, "dct_zigzag": True, "robust_clip": 3.0},
    "quantum": {
        "num_qubits": 6,
        "reuploading_rounds": 2,
        "encoding_gate": "RY_RZ",
        "entanglement": "ring_cnot",
        "expectation_dtype": "float64",
        "state_dtype": "complex128",
        "shots": 0,
        "shot_options": [0, 128, 256, 512, 1024],
        "noise_options": [0.0, 0.001, 0.005, 0.01],
    },
    "classifier": {
        "name": "multinomial_logistic_regression",
        "learning_rate": 0.01,
        "max_epochs": 1000,
        "weight_decay_candidates": [0.0, 0.0001, 0.001, 0.01],
        "early_stopping_patience": 50,
        "class_weighting": True,
    },
    "statistics": {"bootstrap_resamples": 1000, "permutation_resamples": 5000, "confidence_level": 0.95, "correction": "holm"},
    "plots": {"dpi": 300, "png_min_width": 1600, "png_min_height": 1000, "title_font_size": 18, "axis_font_size": 15, "tick_font_size": 12, "legend_font_size": 12},
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of two dictionaries."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load YAML config and fill missing values with defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and Path(path).exists():
        with Path(path).open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, user_cfg)
    return cfg


def apply_cli_overrides(cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    """Apply command-line values to a resolved configuration."""
    out = copy.deepcopy(cfg)
    if getattr(args, "seed", None) is not None:
        out["experiment"]["seeds"] = [int(args.seed)]
    if getattr(args, "folds", None) is not None:
        out["experiment"]["n_splits"] = int(args.folds)
    if getattr(args, "repeats", None) is not None:
        out["experiment"]["repeats"] = int(args.repeats)
    if getattr(args, "shots", None) is not None:
        out["quantum"]["shots"] = int(args.shots)
    if getattr(args, "noise_level", None) is not None:
        out["quantum"]["noise_level"] = float(args.noise_level)
    if getattr(args, "batch_size", None) is not None:
        out["runtime"] = {**out.get("runtime", {}), "batch_size": int(args.batch_size)}
    if getattr(args, "num_workers", None) is not None:
        out["runtime"] = {**out.get("runtime", {}), "num_workers": int(args.num_workers)}
    if getattr(args, "quick", False):
        out["experiment"]["seeds"] = out["experiment"]["seeds"][:1]
        out["experiment"]["n_splits"] = min(2, int(out["experiment"]["n_splits"]))
        out["experiment"]["quick_max_samples"] = 300
        out["classifier"]["max_epochs"] = min(180, int(out["classifier"]["max_epochs"]))
        out["statistics"]["bootstrap_resamples"] = min(100, int(out["statistics"]["bootstrap_resamples"]))
        out["statistics"]["permutation_resamples"] = min(200, int(out["statistics"]["permutation_resamples"]))
        out["runtime"] = {**out.get("runtime", {}), "quick": True}
    return out


def save_resolved_config(cfg: dict[str, Any], out_dir: Path) -> None:
    """Save resolved configuration as YAML and JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
