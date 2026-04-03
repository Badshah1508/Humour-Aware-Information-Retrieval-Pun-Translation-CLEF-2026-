import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.logger import logging  


class DenseRetriever:
    def __init__(self, embedding_model, similarity="cosine"):
        self.embedding_model = embedding_model
        self.similarity = similarity.lower()
        self.corpus_embeddings = None
        self.doc_ids = []

        if self.similarity not in {"cosine", "dot"}:
            raise ValueError(f"Unsupported similarity: {self.similarity}")

    def fit(self, corpus):
        """
        corpus: list of dicts [{"doc_id":..., "text":...}]
        """
        logging.info("Starting corpus encoding for dense retrieval...")

        texts = []
        self.doc_ids = []

        for idx, doc in enumerate(corpus):
            if doc is None or not isinstance(doc, dict):
                logging.warning(f"Skipping invalid corpus document at index {idx}: {doc}")
                continue

            text = doc.get("text")
            doc_id = doc.get("doc_id")

            if text is None:
                logging.warning(f"Skipping corpus doc_id={doc_id} with missing text")
                continue

            text = str(text).strip()
            if text == "":
                logging.warning(f"Skipping corpus doc_id={doc_id} with empty text")
                continue

            if doc_id is None:
                doc_id = str(idx)
                logging.warning(f"Missing doc_id found. Assigning temporary id: {doc_id}")

            texts.append(text)
            self.doc_ids.append(doc_id)

        logging.info(f"Total documents to encode: {len(texts)}")

        self.corpus_embeddings = self.embedding_model.encode_corpus(texts)

        logging.info("Corpus encoding completed successfully.")

    def _score_query(self, q_emb):
        if self.similarity == "dot":
            return np.dot(self.corpus_embeddings, q_emb)

        return cosine_similarity([q_emb], self.corpus_embeddings)[0]

    def search(self, queries, top_k=10, query_ids=None):
        logging.info("Starting query encoding for dense retrieval...")

        if self.corpus_embeddings is None or len(self.doc_ids) == 0:
            raise ValueError("DenseRetriever must be fit on corpus before search().")

        query_embeddings = self.embedding_model.encode_queries(queries)

        logging.info(f"Total queries: {len(queries)}")

        results = {}

        effective_top_k = min(top_k, len(self.doc_ids))

        for i, q_emb in enumerate(query_embeddings):
            similarities = self._score_query(q_emb)

            top_indices = np.argsort(similarities)[::-1][:effective_top_k]
            qid = query_ids[i] if query_ids is not None and i < len(query_ids) else f"q{i+1}"

            results[qid] = [
                {
                    "doc_id": self.doc_ids[idx],
                    "score": float(similarities[idx])
                }
                for idx in top_indices
            ]

            if i < 2:
                logging.info(f"Top results for query {qid}: {results[qid][:2]}")

        logging.info("Dense retrieval search completed.")

        return results