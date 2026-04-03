from sentence_transformers import CrossEncoder
from src.logger import logging


class CrossEncoderReranker:
    def __init__(self, model_name="models/reranker/cross_encoder_finetuned"):
        logging.info(f"Loading Fine-tuned Cross-Encoder model from: {model_name}")
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