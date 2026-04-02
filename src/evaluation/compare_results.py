import json
import pandas as pd
from src.logger import logging
from src.evaluation.retrieval_eval import AdvancedEvaluator
from src.data.data_loader import DataLoader


def load_json(path):
    logging.info(f"Loading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_model(results_path):
    results = load_json(results_path)

    loader = DataLoader(task="task1_retrieval", language="english")
    qrels_df = loader.load_qrels()

    evaluator = AdvancedEvaluator(qrels_df, results)
    metrics = evaluator.evaluate()

    return metrics


if __name__ == "__main__":
    qrels_path = "data/task1_retrieval/qrels.json"

    models = {
        "BM25": "results/task1_retrieval/english/bm25_results.json",
        "RM3": "results/task1_retrieval/english/rm3_results.json",
        "Dense": "results/task1_retrieval/english/dense_results.json",
        "Hybrid": "results/task1_retrieval/english/hybrid_results.json",
        "Cross-Encoder": "results/task1_retrieval/english/cross_encoder_results.json"
    }

    all_results = {}

    for model_name, path in models.items():
        logging.info(f"Evaluating {model_name}...")
        metrics = evaluate_model(path)
        all_results[model_name] = metrics

    # Convert to DataFrame
    df = pd.DataFrame(all_results).T  # transpose

    print("\n Final Comparison Table:\n")
    print(df.to_string())

    # Save
    df.to_csv("results/final_metrics_comparison.csv")
    logging.info("Saved comparison table.")