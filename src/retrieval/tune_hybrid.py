import json
import os
from itertools import product

import pandas as pd

from src.data.data_loader import DataLoader
from src.evaluation.retrieval_eval import AdvancedEvaluator
from src.retrieval.hybrid_retrieval import HybridRetriever


TASK = "task1_retrieval"
LANGUAGE = "english"
RESULT_DIR = "results/task1_retrieval/english"
TOP_K = int(os.getenv("HYBRID_TOP_K", "100"))

BM25_FILE = os.getenv("HYBRID_TUNE_BM25_FILE", "bm25_results.json")
DENSE_FILE = os.getenv("HYBRID_TUNE_DENSE_FILE", "dense_results.json")

ALPHAS = [float(x) for x in os.getenv("HYBRID_TUNE_ALPHAS", "0.2,0.3,0.4,0.5,0.6,0.7,0.8").split(",")]
RRF_KS = [int(x) for x in os.getenv("HYBRID_TUNE_RRF_KS", "10,30,60,90").split(",")]
METHODS = [x.strip().lower() for x in os.getenv("HYBRID_TUNE_METHODS", "rrf,score").split(",")]

METRIC = os.getenv("HYBRID_TUNE_METRIC", "nDCG@5")
OUT_CSV = os.getenv("HYBRID_TUNE_REPORT", "hybrid_tuning_metrics.csv")
OUT_JSON = os.getenv("HYBRID_TUNE_OUTPUT", "hybrid_results_tuned.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    loader = DataLoader(task=TASK, language=LANGUAGE)
    qrels_df = loader.load_qrels()

    bm25_results = load_json(os.path.join(RESULT_DIR, BM25_FILE))
    dense_results = load_json(os.path.join(RESULT_DIR, DENSE_FILE))

    rows = []
    best_score = float("-inf")
    best_results = None
    best_cfg = None

    for method, alpha, rrf_k in product(METHODS, ALPHAS, RRF_KS):
        if method == "score" and rrf_k != RRF_KS[0]:
            # score fusion does not use rrf_k; evaluate once per alpha.
            continue

        retriever = HybridRetriever(alpha=alpha, fusion_method=method, rrf_k=rrf_k)
        fused = retriever.fuse(bm25_results, dense_results, top_k=TOP_K)

        metrics = AdvancedEvaluator(qrels_df, fused).evaluate()

        row = {
            "fusion_method": method,
            "alpha": alpha,
            "rrf_k": rrf_k,
        }
        row.update(metrics)
        rows.append(row)

        score = float(metrics.get(METRIC, 0.0))
        if score > best_score:
            best_score = score
            best_results = fused
            best_cfg = row

    if not rows:
        raise RuntimeError("No tuning runs were executed.")

    os.makedirs(RESULT_DIR, exist_ok=True)

    df = pd.DataFrame(rows).sort_values(by=[METRIC, "MAP", "MRR"], ascending=False)
    df.to_csv(os.path.join(RESULT_DIR, OUT_CSV), index=False)

    with open(os.path.join(RESULT_DIR, OUT_JSON), "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=4)

    print("Best hybrid config:")
    print(best_cfg)
    print(f"Saved tuning report: {os.path.join(RESULT_DIR, OUT_CSV)}")
    print(f"Saved tuned results: {os.path.join(RESULT_DIR, OUT_JSON)}")


if __name__ == "__main__":
    main()
