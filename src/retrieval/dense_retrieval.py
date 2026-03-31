import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.logger import logging   # ✅ logging added


class DenseRetriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.corpus_embeddings = None
        self.doc_ids = []

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

    def search(self, queries, top_k=10):
        logging.info("Starting query encoding for dense retrieval...")

        query_embeddings = self.embedding_model.encode_queries(queries)

        logging.info(f"Total queries: {len(queries)}")

        results = {}

        for i, q_emb in enumerate(query_embeddings):
            similarities = cosine_similarity(
                [q_emb], self.corpus_embeddings
            )[0]

            top_indices = np.argsort(similarities)[::-1][:top_k]

            results[str(i)] = [
                {
                    "doc_id": self.doc_ids[idx],
                    "score": float(similarities[idx])
                }
                for idx in top_indices
            ]

            # ✅ Optional debug (only for first few queries)
            if i < 2:
                logging.info(f"Top results for query {i}: {results[str(i)][:2]}")

        logging.info("Dense retrieval search completed.")

        return results