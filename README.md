# LSA-GLQIE

Research code for **LSA-GLQIE: Lesion-Size-Aware Global-Local Quantum Image Encoding for Brain Tumour Classification Under a Fixed Encoding Budget**.

This repository evaluates a fixed six-qubit feature-map strategy for three-class brain tumour classification. The contribution is the encoding strategy: a fixed budget of 12 DCT coefficients is allocated between a global MRI view and a tumour-centred local view according to lesion size. The work does not claim quantum advantage. The circuit is a fixed feature map; only the downstream classical classifier is trained.

## Dataset

Use the official Figshare Brain Tumor Dataset: https://doi.org/10.6084/m9.figshare.1512427

Download page:

```text
https://doi.org/10.6084/m9.figshare.1512427
```

Do not commit the dataset files to GitHub. Download them on the machine where you will run the experiment, then place the dataset folder in the repository root or pass its absolute path with `--data-dir`.

Expected structure:

```text
Dataset/
├── README 2024.txt
├── cvind.mat
├── brainTumorDataPublic_1-766.zip
├── brainTumorDataPublic_767-1532.zip
├── brainTumorDataPublic_1533-2298.zip
└── brainTumorDataPublic_2299-3064.zip
```

If you place the dataset in the repo root, the final layout should be:

```text
lsa-glqie-brain-tumor-classification/
├── main.py
├── config.yaml
├── src/
├── tests/
└── Dataset/
    ├── README 2024.txt
    ├── cvind.mat
    ├── brainTumorDataPublic_1-766.zip
    ├── brainTumorDataPublic_767-1532.zip
    ├── brainTumorDataPublic_1533-2298.zip
    └── brainTumorDataPublic_2299-3064.zip
```

The loader reads MATLAB v7.3 HDF5 `.mat` files directly from ZIP archives using `zipfile`, `io.BytesIO`, and `h5py`. Extracted `.mat` files are also supported if all files are present. Each file contains `cjdata.label`, `cjdata.PID`, `cjdata.image`, `cjdata.tumorBorder`, and `cjdata.tumorMask`. PID is decoded from MATLAB char/uint16 representation.

Classes are `1` meningioma, `2` glioma, and `3` pituitary tumour. Manual tumour masks are used as oracle localisation.

## Method

Each image is preprocessed from the original numerical MRI array, not JPEG. The global branch resizes the full slice to `64 x 64` and extracts low-frequency orthonormal DCT coefficients. The local branch crops the tumour bounding box with configurable context margin, resizes it to `64 x 64`, and extracts the same type of coefficients.

Lesion size is measured by tumour-area ratio. Training-fold quantiles define small, medium and large lesions:

```text
small:  3 global + 9 local
medium: 6 global + 6 local
large:  9 global + 3 local
```

Every quantum method supplies exactly 12 coefficients to the same circuit: six qubits, two RY+RZ reuploading rounds, ring CNOT entanglement after each round, and 21 measurements: six single-qubit `<Zq>` expectations plus all pairwise `<ZaZb>` correlations. Classical LR/SVM controls are trained on the same lesion-aware or fixed 12-coefficient inputs, without the quantum feature map.

Primary evaluation uses patient-disjoint grouped cross-validation. Slice-level samples are not statistically independent, so patient-level metrics are emphasized. The official `cvind.mat` folds are inspected for patient overlap and treated only as sensitivity metadata.

## Installation

Conda:

```bash
conda env create -f environment.yml
conda activate lsa-glqie
```

Pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU acceleration is optional. Six-qubit state-vector simulation is still modest, but CUDA helps batched simulation and PyTorch logistic regression.

On a Linux GPU server, use a CUDA-enabled PyTorch build. With conda, install PyTorch from the official `pytorch`/`nvidia` channels if your server CUDA stack requires a specific version. The code accepts `--device auto`, `--device cuda`, or `--device cpu`; `auto` uses CUDA when `torch.cuda.is_available()` is true.

GPU check:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
```

## Running

Full real-data run:

```bash
python main.py --data-dir "/path/to/Dataset" --output-dir "results" --config "config.yaml"
```

GPU real-data run after placing the dataset at `./Dataset`:

```bash
python main.py \
  --data-dir "Dataset" \
  --output-dir "results_gpu" \
  --config "config.yaml" \
  --device cuda \
  --batch-size 512
```

If CUDA is unavailable, use:

```bash
python main.py --data-dir "Dataset" --output-dir "results_cpu" --config "config.yaml" --device cpu
```

One-command script:

```bash
bash run_experiment.sh "/path/to/Dataset"
```

Quick run:

```bash
python main.py --data-dir "Dataset" --output-dir "results_quick" --config "config.yaml" --quick
```

Synthetic smoke test:

```bash
python main.py --synthetic-smoke-test --quick --output-dir results_smoke
```

Resume:

```bash
python main.py --data-dir "Dataset" --output-dir "results" --config "config.yaml" --resume
```

Tests:

```bash
python -m pytest -q
```

## Outputs

The run creates `results/logs`, `results/configs`, `results/cache`, `results/predictions`, `results/models`, `results/tables`, `results/plots`, and `results/reproducibility`.

Exactly six final tables are saved as CSV, Markdown and LaTeX:

1. Dataset and split summary
2. Main method results
3. Per-class and lesion-size results
4. Allocation and encoding ablations
5. Robustness results
6. Statistical and resource summary

Exactly eight final figure groups are saved as PNG and PDF:

1. Dataset overview
2. Method workflow
3. Lesion size and allocation analysis
4. Main performance comparison
5. Confusion and per-class results
6. Lesion-size subgroup and allocation ablation
7. Robustness and resource tradeoffs
8. ROC, PR and calibration summary

The machine-readable source data used for each final figure is also saved under:

```text
results/plots/data/
```

These CSV files let you recreate the plots in a different style or tool without rerunning the experiment.

To regenerate the final figures from an existing results folder:

```bash
python scripts/replot_saved_results.py \
  --results-dir results \
  --data-dir Dataset
```

The `--data-dir` argument is optional, but providing it lets Figure 1 use real MRI examples with mask overlays.

## Hybrid Deep-Quantum Fusion

For the stronger quantum direction, use pretrained CNN features first and apply quantum processing only to a compact learned embedding. This keeps the quantum component as a trainable fusion/decision head instead of using it directly on raw DCT coefficients.

GPU run:

```bash
python scripts/train_deep_quantum_fusion.py \
  --config config_deep_quantum.yaml \
  --data-dir Dataset \
  --output-dir results_deep_quantum \
  --device cuda
```

The script extracts three frozen pretrained views per slice: full MRI, tumour-centred ROI, and peritumoral context. Their embeddings are concatenated with compact lesion-size/shape features, projected to quantum rotation angles, processed by a trainable variational quantum circuit, and evaluated with patient-disjoint folds. Outputs are saved under `results_deep_quantum/predictions`, `results_deep_quantum/tables`, `results_deep_quantum/plots`, `results_deep_quantum/models`, and `results_deep_quantum/reproducibility`.

If the GPU server cannot download ImageNet weights, either preload the TorchVision cache or run with `--weights none` for code testing only.

The effective noise model is a feature-space perturbation and is not a hardware-accurate quantum noise channel. Lesion size may correlate with tumour class, so size-only and shuffled-allocation controls are included.

## Troubleshooting

If loading `.mat` files fails, confirm `h5py` is installed. `scipy.io.loadmat` alone cannot read the MATLAB v7.3 image files. If CUDA is requested but unavailable, the code falls back to CPU and logs a warning. If `pyarrow` is unavailable, metadata is also saved as CSV and a small marker file explains the fallback.
