"""Run pretrained CNN feature extraction followed by a trainable quantum head."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deep_quantum_fusion import DeepQuantumConfig, run_deep_quantum_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lesion-size-aware deep quantum fusion on the Figshare brain tumor dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], required=True)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = DeepQuantumConfig.from_yaml(args.config)
    report = run_deep_quantum_experiment(args.data_dir, args.output_dir, cfg, args.device, resume=not args.no_resume)
    print("Deep quantum fusion complete")
    print(f"Output directory: {report['output_dir']}")
    print(f"Device: {report['device']}")
    print(f"Feature tensor shape: {report['feature_shape']}")
    print(f"Summary table: {report['summary_table']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
