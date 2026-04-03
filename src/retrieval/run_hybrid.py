import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.retrieval.hybrid_retrieval import HybridRetriever
from src.logger import logging


def load_json(path):
    logging.info(f"Loading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load results
bm25_path = "results/task1_retrieval/english/bm25_results.json"
dense_path = "results/task1_retrieval/english/dense_results.json"

bm25_results = load_json(bm25_path)
dense_results = load_json(dense_path)

# Init hybrid
alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
fusion_method = os.getenv("HYBRID_FUSION_METHOD", "rrf")
rrf_k = int(os.getenv("HYBRID_RRF_K", "60"))
top_k = int(os.getenv("HYBRID_TOP_K", "100"))

hybrid = HybridRetriever(alpha=alpha, fusion_method=fusion_method, rrf_k=rrf_k)

# Fuse
final_results = hybrid.fuse(bm25_results, dense_results, top_k=top_k)

# Save
output_path = "results/task1_retrieval/english/hybrid_results.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4)

logging.info(f"Hybrid results saved at: {output_path}")