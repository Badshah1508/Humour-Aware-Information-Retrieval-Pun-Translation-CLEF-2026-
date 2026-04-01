try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: sentence-transformers is required for dense retrieval. "
        "Install it with `pip install sentence-transformers` and retry."
    ) from e


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_corpus(self, corpus_texts):
        return self.model.encode(corpus_texts, show_progress_bar=True)

    def encode_queries(self, queries):
        return self.model.encode(queries, show_progress_bar=True)