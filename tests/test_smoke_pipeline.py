"""Synthetic pipeline smoke test."""

from __future__ import annotations

import subprocess
import sys


def test_one_command_synthetic_smoke_test(tmp_path):
    output = tmp_path / "results_smoke"
    cmd = [
        sys.executable,
        "main.py",
        "--synthetic-smoke-test",
        "--quick",
        "--output-dir",
        str(output),
        "--folds",
        "2",
        "--seed",
        "11",
        "--batch-size",
        "64",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((output / "tables").glob("table_*.csv"))) == 6
    assert len(list((output / "plots").glob("figure_*.png"))) == 8
    assert (output / "reproducibility" / "plot_validation.csv").exists()

