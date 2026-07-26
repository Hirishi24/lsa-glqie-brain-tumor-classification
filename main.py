"""Command-line entrypoint for LSA-GLQIE experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import apply_cli_overrides, load_config, save_resolved_config
from src.experiments import ensure_result_dirs, run_experiment
from src.logging_utils import setup_logging
from src.reproducibility import set_global_seed
from src.synthetic_data import create_synthetic_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="LSA-GLQIE brain tumour classification experiment")
    parser.add_argument("--data-dir", type=str, default="Dataset")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--noise-level", type=float)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run CLI."""
    args = parse_args()
    output = Path(args.output_dir)
    ensure_result_dirs(output)
    logger = setup_logging(output)
    cfg = apply_cli_overrides(load_config(args.config), args)
    set_global_seed(int(cfg["experiment"]["seeds"][0]))
    save_resolved_config(cfg, output / "configs")
    data_dir = Path(args.data_dir)
    if args.synthetic_smoke_test:
        data_dir = create_synthetic_dataset(output / "synthetic_dataset", seed=int(cfg["experiment"]["seeds"][0]))
        logger.info("Created synthetic smoke dataset at %s", data_dir)
    try:
        report = run_experiment(data_dir, output, cfg, args, logger)
    except Exception:
        logger.exception("Experiment failed")
        return 1
    print("Experiment complete")
    print(f"Output directory: {report['output_dir']}")
    print(f"Device: {report['device']}")
    print(f"Final tables: {report['tables']}")
    print(f"Final figure groups: {report['figures']}")
    print(f"Validation: {report['validation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

