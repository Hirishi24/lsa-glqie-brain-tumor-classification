"""Environment reporting."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def select_device(requested: str) -> str:
    """Return cpu or cuda, warning through text when CUDA is unavailable."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if requested == "auto":
        return "cuda" if has_cuda else "cpu"
    if requested == "cuda" and not has_cuda:
        print("WARNING: CUDA was requested but is unavailable; using CPU.")
        return "cpu"
    return requested


def collect_environment(device: str) -> dict[str, Any]:
    """Collect reproducibility environment facts."""
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": device,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def save_environment(output_dir: Path, device: str) -> None:
    """Save environment, pip freeze, git commit and runtime metadata."""
    rep = output_dir / "reproducibility"
    rep.mkdir(parents=True, exist_ok=True)
    info = collect_environment(device)
    with (rep / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True)
        (rep / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    except Exception as exc:
        (rep / "pip_freeze.txt").write_text(f"pip freeze failed: {exc}\n", encoding="utf-8")
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
        (rep / "git_commit.txt").write_text(git.stdout.strip() + "\n", encoding="utf-8")
    except Exception as exc:
        (rep / "git_commit.txt").write_text(f"git unavailable: {exc}\n", encoding="utf-8")

