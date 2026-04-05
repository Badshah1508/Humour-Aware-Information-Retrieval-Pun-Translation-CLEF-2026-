import os

from sentence_transformers import CrossEncoder
from src.logger import logging


class CrossEncoderReranker:
    def __init__(self, model_name="models/reranker/cross_encoder_finetuned"):
        logging.info(f"Loading Fine-tuned Cross-Encoder model from: {model_name}")
        self.model = CrossEncoder(model_name)

    @staticmethod
    def _env_int(name, default):
        raw = os.getenv(name, str(default)).strip()
        try:
            value = int(raw)
            return value if value > 0 else default
        except (TypeError, ValueError):
            logging.warning(f"Invalid {name}={raw!r}; using default {default}")
            return default

    @staticmethod
    def _env_bool(name, default=False):
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def rerank(self, query, docs):
        """
        query: string
        docs: list of dicts [{"doc_id":..., "text":...}]
        """
        if not docs:
            return []

        pairs = [(query, doc["text"]) for doc in docs]

        batch_size = self._env_int("CROSS_ENCODER_BATCH_SIZE", 32)
        max_length = self._env_int("CROSS_ENCODER_MAX_LENGTH", 512)
        show_progress_bar = self._env_bool("CROSS_ENCODER_SHOW_PROGRESS", False)

        try:
            scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                max_length=max_length,
            )
        except TypeError:
            # Older sentence-transformers versions do not accept max_length.
            scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            )

        reranked = []

        for doc, score in zip(docs, scores):
            reranked.append({
                "doc_id": doc["doc_id"],
                "score": float(score)
            })

        # Sort by score
        reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

        return reranked