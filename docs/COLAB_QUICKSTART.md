# Colab Quickstart for CLEF 2026 Humour Project

Use this for GPU-heavy steps only.

## Recommended split

- Local CPU: baseline retrieval, preprocessing, result comparison
- Colab GPU: dense retrieval, cross-encoder training, cross-encoder reranking

## Colab setup

1. Open a new Colab notebook and enable GPU runtime.
2. Run the notebook in `notebooks/colab_clef_pipeline.ipynb` from this repository.

## Run Dense Experiments In Colab (with Drive export)

Use this when `src/retrieval/run_dense_experiments.py` is too slow locally.

```python
# In Colab cell
!git clone https://github.com/Badshah1508/Humour-Aware-Information-Retrieval-Pun-Translation-CLEF-2026-.git
%cd Humour-Aware-Information-Retrieval-Pun-Translation-CLEF-2026-
!pip install -r requirements.txt
```

```python
# Optional tuning knobs for Colab GPU
import os
os.environ["RETRIEVAL_QUERY_SPLIT"] = "all"
os.environ["DENSE_BATCH_SIZE"] = "128"  # try 192/256 if memory allows
```

```python
# Runs dense experiments + exports output files to Google Drive
!python scripts/colab_run_dense_experiments.py
```

Outputs are exported to:

- `MyDrive/CLEF_2026_Humour_Project/dense_experiments_<timestamp>/`

Export includes:

- `dense_experiment_metrics.csv`
- `dense_results_minilm_cosine.json`
- `dense_results_mpnet_cosine.json`
- `dense_results_bge_dot.json`
- `dense_results_e5_dot.json`

## One-command profile runner

From repo root:

```powershell
# Laptop-safe local profile (best for low RAM + integrated graphics)
./scripts/run_profile.ps1 -Profile laptop -Step all

# Fast local profile
./scripts/run_profile.ps1 -Profile fast -Step all

# Full profile (recommended in Colab GPU)
./scripts/run_profile.ps1 -Profile full -Step all

# Run only training
./scripts/run_profile.ps1 -Profile full -Step train

# Benchmark and print per-stage runtime summary
./scripts/run_profile.ps1 -Profile laptop -Step all -Benchmark

# Benchmark and append results to CSV (logs/runtime_benchmark.csv)
./scripts/run_profile.ps1 -Profile laptop -Step all -Benchmark -WriteBenchmarkCsv

# Compare benchmark history and print fastest profile per stage
./scripts/run_profile.ps1 -CompareBenchmark

# Compare benchmark history and print top 3 (or custom N) fastest runs per stage
./scripts/run_profile.ps1 -CompareBenchmark -TopN 3
```

## Fast vs full profiles

Fast profile is designed to reduce runtime for iteration:

- CE_EPOCHS=2
- CE_BATCH_SIZE=6
- CE_MAX_HARD_NEGS=8
- CE_MAX_RANDOM_NEGS=3
- DENSE_BATCH_SIZE=32
- DENSE_TOP_K=40
- CROSS_ENCODER_CANDIDATES=15
- DENSE model: all-MiniLM-L6-v2

Laptop profile is tuned for low-memory CPU systems:

- CE_EPOCHS=1
- CE_BATCH_SIZE=4
- CE_MAX_HARD_NEGS=5
- CE_MAX_RANDOM_NEGS=2
- DENSE_BATCH_SIZE=16
- DENSE_TOP_K=30
- CROSS_ENCODER_CANDIDATES=10
- TOKENIZERS_PARALLELISM=false

Full profile is for final-quality runs:

- CE_EPOCHS=8
- CE_MAX_HARD_NEGS=40
- CE_MAX_RANDOM_NEGS=10
- CROSS_ENCODER_CANDIDATES=100
- DENSE model: all-mpnet-base-v2

## Notes

- On Colab T4, if memory allows, increase `DENSE_BATCH_SIZE` to 192 or 256.
- If training OOM happens, lower `CE_BATCH_SIZE` to 8.
- Save outputs to Drive or push result files to your repository branch.
