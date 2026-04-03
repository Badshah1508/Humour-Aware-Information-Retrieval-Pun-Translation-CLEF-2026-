import json
from collections import defaultdict
from src.logger import logging


class HybridRetriever:
    def __init__(self, alpha=0.5, fusion_method="rrf", rrf_k=60):
        self.alpha = alpha  # BM25 weight when using score fusion
        self.fusion_method = fusion_method.lower()
        self.rrf_k = rrf_k

        if self.fusion_method not in {"rrf", "score"}:
            raise ValueError(f"Unsupported fusion_method: {self.fusion_method}")

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

    def _to_rank_dict(self, results):
        if isinstance(results, list):
            grouped = defaultdict(list)
            for item in results:
                qid = str(item.get("query_id") or item.get("qid"))
                grouped[qid].append(
                    {
                        "doc_id": str(item["doc_id"]),
                        "score": float(item["score"]),
                    }
                )
            results = grouped

        rank_dict = {}
        for qid, docs in results.items():
            sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)
            rank_dict[str(qid)] = {
                str(doc["doc_id"]): rank + 1 for rank, doc in enumerate(sorted_docs)
            }
        return rank_dict

    def _fuse_rrf(self, bm25_results, dense_results, top_k):
        bm25_ranks = self._to_rank_dict(bm25_results)
        dense_ranks = self._to_rank_dict(dense_results)

        final_results = {}
        all_qids = set(bm25_ranks) | set(dense_ranks)

        for qid in all_qids:
            score_dict = defaultdict(float)
            doc_ids = set(bm25_ranks.get(qid, {})) | set(dense_ranks.get(qid, {}))

            for doc_id in doc_ids:
                bm_rank = bm25_ranks.get(qid, {}).get(doc_id)
                de_rank = dense_ranks.get(qid, {}).get(doc_id)

                if bm_rank is not None:
                    score_dict[doc_id] += self.alpha * (1.0 / (self.rrf_k + bm_rank))
                if de_rank is not None:
                    score_dict[doc_id] += (1.0 - self.alpha) * (1.0 / (self.rrf_k + de_rank))

            ranked_docs = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            final_results[qid] = [{"doc_id": doc_id, "score": float(score)} for doc_id, score in ranked_docs]

        return final_results

    def _fuse_score(self, bm25_results, dense_results, top_k):
        bm25_norm = self.normalize(bm25_results)
        dense_norm = self.normalize(dense_results)

        final_results = {}
        all_qids = set(bm25_norm) | set(dense_norm)

        for qid in all_qids:
            score_dict = defaultdict(float)

            for doc in bm25_norm.get(qid, []):
                score_dict[doc["doc_id"]] += self.alpha * doc["score"]

            for doc in dense_norm.get(qid, []):
                score_dict[doc["doc_id"]] += (1 - self.alpha) * doc["score"]

            ranked_docs = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            final_results[qid] = [{"doc_id": doc_id, "score": float(score)} for doc_id, score in ranked_docs]

        return final_results

    def fuse(self, bm25_results, dense_results, top_k=10):
        logging.info("Starting Hybrid Retrieval...")

        if self.fusion_method == "rrf":
            final_results = self._fuse_rrf(bm25_results, dense_results, top_k)
        else:
            final_results = self._fuse_score(bm25_results, dense_results, top_k)

        logging.info("Hybrid retrieval completed.")
        return final_results