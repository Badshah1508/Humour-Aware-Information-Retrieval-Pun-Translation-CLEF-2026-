from sentence_transformers import CrossEncoder
from src.logger import logging


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logging.info(f"Loading Cross-Encoder model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs):
        """
        query: string
        docs: list of dicts [{"doc_id":..., "text":...}]
        """
        pairs = [(query, doc["text"]) for doc in docs]

        scores = self.model.predict(pairs)

        reranked = []

        for doc, score in zip(docs, scores):
            reranked.append({
                "doc_id": doc["doc_id"],
                "score": float(score)
            })

        # Sort by score
        reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

        return reranked