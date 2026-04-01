import json
from collections import defaultdict
from src.logger import logging


class HybridRetriever:
    def __init__(self, alpha=0.5):
        self.alpha = alpha  # weight for BM25

    def normalize(self, results):
        """
        Min-max normalize scores per query.
        Accepts either:
          - dict mapping qid -> list of docs
          - list of score entries with query_id/doc_id/score
        """
        if isinstance(results, list):
            grouped = defaultdict(list)
            for item in results:
                qid = item.get("query_id") or item.get("qid")
                grouped[qid].append({
                    "doc_id": item["doc_id"],
                    "score": item["score"]
                })
            results = grouped

        normalized = {}

        for qid, docs in results.items():
            scores = [doc["score"] for doc in docs]

            if not scores:
                normalized[qid] = docs
                continue

            min_s = min(scores)
            max_s = max(scores)

            norm_docs = []
            for doc in docs:
                if max_s - min_s == 0:
                    norm_score = 0
                else:
                    norm_score = (doc["score"] - min_s) / (max_s - min_s)

                norm_docs.append({
                    "doc_id": doc["doc_id"],
                    "score": norm_score
                })

            normalized[qid] = norm_docs

        return normalized

    def fuse(self, bm25_results, dense_results, top_k=10):
        logging.info("Starting Hybrid Retrieval...")

        bm25_norm = self.normalize(bm25_results)
        dense_norm = self.normalize(dense_results)

        final_results = {}
        all_qids = set(bm25_norm) | set(dense_norm)

        for qid in all_qids:
            score_dict = defaultdict(float)

            # BM25 contribution
            for doc in bm25_norm.get(qid, []):
                score_dict[doc["doc_id"]] += self.alpha * doc["score"]

            # Dense contribution
            for doc in dense_norm.get(qid, []):
                score_dict[doc["doc_id"]] += (1 - self.alpha) * doc["score"]

            # Sort
            ranked_docs = sorted(
                score_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            final_results[qid] = [
                {"doc_id": doc_id, "score": float(score)}
                for doc_id, score in ranked_docs
            ]

        logging.info("Hybrid retrieval completed.")
        return final_results