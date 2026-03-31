import os
import sys
import json
import pandas as pd
import math

from src.exception import CustomException
from src.logger import logging


class AdvancedEvaluator:
    def __init__(self, qrels_df, results_list):
        try:
            self.qrels_df = qrels_df
            self.results_list = results_list
        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Prepare Qrels
    # -----------------------------
    def prepare_qrels(self):
        try:
            qrels = {}

            for _, row in self.qrels_df.iterrows():
                qid = str(row["query_id"])
                doc_id = str(row["doc_id"])

                rel = row["relevance"]
                if pd.isna(rel):
                    rel = 0
                else:
                    rel = int(rel)

                if qid not in qrels:
                    qrels[qid] = {}

                qrels[qid][doc_id] = rel

            return qrels

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Prepare Results
    # -----------------------------
    def prepare_results(self):
        try:
            results = {}

            for item in self.results_list:
                qid = str(item["query_id"])
                doc_id = str(item["doc_id"])
                score = float(item["score"])

                if qid not in results:
                    results[qid] = []

                results[qid].append((doc_id, score))

            # sort descending
            for qid in results:
                results[qid] = sorted(
                    results[qid],
                    key=lambda x: x[1],
                    reverse=True
                )

            return results

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Metrics
    # -----------------------------
    def precision_at_k(self, relevant, retrieved, k):
        retrieved_k = [doc for doc, _ in retrieved[:k]]
        return sum(1 for doc in retrieved_k if doc in relevant) / k

    def recall_at_k(self, relevant, retrieved, k):
        retrieved_k = [doc for doc, _ in retrieved[:k]]
        return sum(1 for doc in retrieved_k if doc in relevant) / max(len(relevant), 1)

    def average_precision(self, relevant, retrieved):
        score = 0
        hits = 0

        for i, (doc, _) in enumerate(retrieved):
            if doc in relevant:
                hits += 1
                score += hits / (i + 1)

        return score / max(len(relevant), 1)

    def dcg(self, rel_scores):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(rel_scores))

    def ndcg_at_k(self, qrels, retrieved, k):
        rel_scores = [qrels.get(doc, 0) for doc, _ in retrieved[:k]]
        dcg_val = self.dcg(rel_scores)

        ideal = sorted(qrels.values(), reverse=True)[:k]
        idcg = self.dcg(ideal)

        return dcg_val / idcg if idcg > 0 else 0

    def reciprocal_rank(self, relevant, retrieved):
        for i, (doc, _) in enumerate(retrieved):
            if doc in relevant:
                return 1 / (i + 1)
        return 0

    def bpref(self, relevant, retrieved):
        non_rel = 0
        score = 0

        for doc, _ in retrieved:
            if doc in relevant:
                score += 1 - (non_rel / max(len(relevant), 1))
            else:
                non_rel += 1

        return score / max(len(relevant), 1)

    # -----------------------------
    # Evaluate
    # -----------------------------
    def evaluate(self):
        try:
            qrels = self.prepare_qrels()
            results = self.prepare_results()
            
            #  PUT HERE
            sample_qid = list(qrels.keys())[0]

            print("\n DEBUG CHECK:")
            print("Qrels docs:", list(qrels[sample_qid].keys())[:5])
            print("Results docs:", [d for d, _ in results.get(sample_qid, [])])

            print("Total queries in qrels:", len(qrels))

            if len(qrels) == 0:
                raise ValueError("No valid queries found in qrels!")

            metrics = {
                "MAP": 0,
                "nDCG@5": 0,
                "P@1": 0,
                "P@5": 0,
                "P@10": 0,
                "Recall@5": 0,
                "Recall@10": 0,
                "MRR": 0,
                "bpref": 0
            }

            n = len(qrels)

            for qid in qrels:
                relevant = {doc for doc, rel in qrels[qid].items() if rel > 0}
                retrieved = results.get(qid, [])

                metrics["MAP"] += self.average_precision(relevant, retrieved)
                metrics["nDCG@5"] += self.ndcg_at_k(qrels[qid], retrieved, 5)
                metrics["P@1"] += self.precision_at_k(relevant, retrieved, 1)
                metrics["P@5"] += self.precision_at_k(relevant, retrieved, 5)
                metrics["P@10"] += self.precision_at_k(relevant, retrieved, 10)
                metrics["Recall@5"] += self.recall_at_k(relevant, retrieved, 5)
                metrics["Recall@10"] += self.recall_at_k(relevant, retrieved, 10)
                metrics["MRR"] += self.reciprocal_rank(relevant, retrieved)
                metrics["bpref"] += self.bpref(relevant, retrieved)

            for key in metrics:
                metrics[key] /= n

            logging.info(f"Evaluation Results: {metrics}")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    try:
        TASK = "task1_retrieval"
        LANGUAGE = "english"

        print("\n Starting Evaluation...\n")

        from src.data.data_loader import DataLoader

        loader = DataLoader(task=TASK, language=LANGUAGE)
        data = loader.load_all()

        qrels_df = data["qrels"]

        # Fix NaN
        qrels_df["relevance"] = qrels_df["relevance"].fillna(0)

        # Choose which result to evaluate
        RESULT_FILE = "rm3_results.json"  #  change to rm3_results.json if needed

        results_path = os.path.join(
            "results", TASK, LANGUAGE, RESULT_FILE
        )

        print("Looking for:", results_path)

        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(results_path, "r", encoding="utf-8") as f:
            results_list = json.load(f)

        print("Loaded results:", len(results_list))

        evaluator = AdvancedEvaluator(qrels_df, results_list)
        scores = evaluator.evaluate()

        print("\n FINAL Evaluation Results:")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

    except Exception as e:
        raise CustomException(e, sys)