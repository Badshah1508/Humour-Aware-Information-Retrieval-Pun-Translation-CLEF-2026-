import json
import os
from typing import List, Dict

import pandas as pd

from src.data.data_loader import DataLoader
from src.evaluation.retrieval_eval import AdvancedEvaluator
from src.logger import logging
from src.retrieval.dense_retrieval import DenseRetriever
from src.retrieval.embedding_model import EmbeddingModel


TASK = "task1_retrieval"
LANGUAGE = "english"
TOP_K = int(os.getenv("DENSE_EXPERIMENT_TOP_K", "100"))
OUT_DIR = "results/task1_retrieval/english"
DENSE_BATCH_SIZE = int(os.getenv("DENSE_BATCH_SIZE", "64"))


def run_dense_variant(
    corpus,
    query_texts,
    query_ids,
    model_name: str,
    normalize_embeddings: bool,
    similarity: str,
    output_file: str,
    query_prefix: str = "",
    doc_prefix: str = "",
):
    logging.info(
        f"Running dense variant model={model_name}, normalize={normalize_embeddings}, similarity={similarity}"
    )

    embedding_model = EmbeddingModel(
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        batch_size=DENSE_BATCH_SIZE,
        query_prefix=query_prefix,
        doc_prefix=doc_prefix,
    )
    retriever = DenseRetriever(embedding_model, similarity=similarity)

    retriever.fit(corpus)
    results = retriever.search(query_texts, top_k=TOP_K, query_ids=query_ids)

    os.makedirs(OUT_DIR, exist_ok=True)
    output_path = os.path.join(OUT_DIR, output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return output_path, results


def evaluate_results(qrels_df, results) -> Dict[str, float]:
    evaluator = AdvancedEvaluator(qrels_df, results)
    return evaluator.evaluate()


if __name__ == "__main__":
    loader = DataLoader(task=TASK, language=LANGUAGE)
    data = loader.load_all()

    corpus = data["corpus"].to_dict(orient="records")
    queries_df = data["queries"]
    qrels_df = data["qrels"]

    query_texts = queries_df["query"].astype(str).tolist()
    query_ids = queries_df["query_id"].astype(str).tolist()

    experiments: List[Dict] = [
        {
            "name": "MiniLM-cosine",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "normalize_embeddings": True,
            "similarity": "cosine",
            "output_file": "dense_results_minilm_cosine.json",
        },
        {
            "name": "MPNet-cosine",
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "normalize_embeddings": True,
            "similarity": "cosine",
            "output_file": "dense_results_mpnet_cosine.json",
        },
        {
            "name": "BGE-base-dot",
            "model_name": "BAAI/bge-base-en-v1.5",
            "normalize_embeddings": True,
            "similarity": "dot",
            "output_file": "dense_results_bge_dot.json",
        },
        {
            "name": "E5-base-dot",
            "model_name": "intfloat/e5-base-v2",
            "normalize_embeddings": True,
            "similarity": "dot",
            "output_file": "dense_results_e5_dot.json",
            "query_prefix": "query: ",
            "doc_prefix": "passage: ",
        },
    ]

    rows = []

    for exp in experiments:
        output_path, results = run_dense_variant(
            corpus=corpus,
            query_texts=query_texts,
            query_ids=query_ids,
            model_name=exp["model_name"],
            normalize_embeddings=exp["normalize_embeddings"],
            similarity=exp["similarity"],
            output_file=exp["output_file"],
            query_prefix=exp.get("query_prefix", ""),
            doc_prefix=exp.get("doc_prefix", ""),
        )

        metrics = evaluate_results(qrels_df, results)
        row = {
            "Experiment": exp["name"],
            "Model": exp["model_name"],
            "Similarity": exp["similarity"],
            "Normalize": exp["normalize_embeddings"],
            "OutputFile": output_path,
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["MAP", "nDCG@5", "MRR"], ascending=False)

    report_path = os.path.join(OUT_DIR, "dense_experiment_metrics.csv")
    df.to_csv(report_path, index=False)

    print("\nDense Experiment Comparison:\n")
    print(df.to_string(index=False))
    print(f"\nSaved experiment report: {report_path}")
