#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-Dataset}"
OUTPUT_DIR="${2:-results}"
mkdir -p "${OUTPUT_DIR}/reproducibility"

if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  echo "Using active conda environment: ${CONDA_DEFAULT_ENV}"
fi

python - <<'PY'
import platform, sys
print("Python:", sys.version)
print("Platform:", platform.platform())
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Torch check failed:", exc)
PY

python -m pytest -q | tee "${OUTPUT_DIR}/reproducibility/test_results.txt"
python main.py --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" --config config.yaml
echo "Results saved to ${OUTPUT_DIR}"
