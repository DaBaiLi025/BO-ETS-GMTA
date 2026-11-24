# BO-ETS-GMTA source code release

This is a **code/configuration-only** package. It contains no throughput data, forecast outputs, metric tables, bootstrap results, model checkpoints, manuscripts, or reviewer correspondence.

## Contents

- `src/p1_core.py`: P1 ETS and GMTA implementation.
- `src/variants.py`: fixed-gate, no-gate, without-attention, and single-scale-kernel-5 model variants.
- `scripts/run_selected.py`: selected-configuration rolling-origin runner; no hyperparameter search. Neural training requires the explicit `--execute-training` flag and writes only to a new output directory.
- `scripts/recompute_metrics.py` and `scripts/bootstrap_paired.py`: analysis utilities. They require forecast CSVs supplied by the user under `results/certified/` and are not runnable from this code-only package alone.
- `configs/`: selected P1 and PatchTST configurations. These are model settings, not observed throughput or result tables.
- `third_party/PatchTST/`: upstream PatchTST source and Apache-2.0 license.

## Input data expected by the runner

The runner expects a user-provided file at `data/monthly_throughput.csv` with two columns:

```csv
period,y
YYYY-MM,numeric_throughput_value
```

Provide a contiguous monthly series from February 2012 through August 2026 (175 rows) for the configured windows. The values are monthly Shanghai Port container throughput in units of 10^4 TEU. Obtain and verify data and applicable reuse terms independently; this repository intentionally does not distribute them.

## Environment

The audited environment was Python 3.11.9, PyTorch 2.11.0+cu128, NumPy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, and statsmodels 0.15.0. Install compatible PyTorch for your own platform separately.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Install a compatible PyTorch 2.11 build using the official PyTorch installation selector.
```

## Run a selected configuration

After providing `data/monthly_throughput.csv`, a single-task CPU smoke run is:

```powershell
python scripts/run_selected.py --model full --window main --horizons 1 --seeds 42 --targets 2023-01 --device cpu --execute-training --output outputs/smoke_h1_202301
```

This creates a **new fit**, not an exact replay of the historical paper run. The historical full-model checkpoints, complete scaler tensors, and RNG states were not retained. CPU results are not claimed to be numerically identical to the audited CUDA results. Use a fresh output directory for every run.

The P1 evaluation is retrospective, not a blind prospective test. The 2023-2025 outcomes were examined during model development; the 2026 auxiliary outcomes were also inspected. The internal chronological 80:20 validation split is not horizon-purged, and historical data-release vintages were not reconstructed. These limitations must accompany any description of the protocol.

## PatchTST and licensing

PatchTST upstream commit: `204c21efe0b39603ad6e2ca640ef5896646ab1a9` (Apache-2.0). This package includes the upstream source and selected architecture configuration, but not the historical 128-fit selection and 432-fit orchestration as a one-command experiment. Do not imply otherwise.

The manuscript-specific code currently has no license assigned. The corresponding author/rightsholder should select one before public release. Do not assume the upstream PatchTST license applies to this project.
