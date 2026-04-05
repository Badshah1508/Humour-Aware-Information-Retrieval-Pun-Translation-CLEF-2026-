import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.retrieval.hybrid_retrieval import HybridRetriever
from src.logger import logging


def load_json(path):
    logging.info(f"Loading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_best_tuned_config(result_dir):
    """Load best hybrid tuning config if report exists."""
    report_path = os.path.join(result_dir, os.getenv("HYBRID_TUNE_REPORT", "hybrid_tuning_metrics.csv"))
    if not os.path.exists(report_path):
        return None

    try:
        df = pd.read_csv(report_path)
        if df.empty:
            return None

        metric = os.getenv("HYBRID_TUNE_METRIC", "nDCG@5")
        sort_cols = [c for c in [metric, "MAP", "MRR"] if c in df.columns]
        if not sort_cols:
            return None

        best = df.sort_values(by=sort_cols, ascending=False).iloc[0].to_dict()
        return {
            "fusion_method": str(best.get("fusion_method", "rrf")).strip().lower(),
            "alpha": float(best.get("alpha", 0.5)),
            "rrf_k": int(best.get("rrf_k", 60)),
        }
    except Exception as exc:
        logging.warning(f"Could not read tuned hybrid config from {report_path}: {exc}")
        return None


# Load results
bm25_path = "results/task1_retrieval/english/bm25_results.json"
dense_path = "results/task1_retrieval/english/dense_results.json"
result_dir = "results/task1_retrieval/english"

bm25_results = load_json(bm25_path)
dense_results = load_json(dense_path)

# Init hybrid
alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
fusion_method = os.getenv("HYBRID_FUSION_METHOD", "rrf")
rrf_k = int(os.getenv("HYBRID_RRF_K", "60"))
top_k = int(os.getenv("HYBRID_TOP_K", "100"))

if os.getenv("HYBRID_USE_TUNED_CONFIG", "1").strip().lower() in {"1", "true", "yes"}:
    best_cfg = load_best_tuned_config(result_dir)
    if best_cfg:
        fusion_method = best_cfg["fusion_method"]
        alpha = best_cfg["alpha"]
        rrf_k = best_cfg["rrf_k"]
        logging.info(f"Loaded tuned hybrid config: {best_cfg}")
    else:
        logging.info("No tuned config found. Using env/default HYBRID_* values.")

hybrid = HybridRetriever(alpha=alpha, fusion_method=fusion_method, rrf_k=rrf_k)

# Fuse
final_results = hybrid.fuse(bm25_results, dense_results, top_k=top_k)

# Save
output_path = "results/task1_retrieval/english/hybrid_results.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4)

logging.info(f"Hybrid results saved at: {output_path}")