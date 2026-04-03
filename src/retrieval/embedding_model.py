from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        normalize_embeddings=True,
        batch_size=64,
    ):
        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

    def encode_corpus(self, corpus_texts):
        return self.model.encode(
            corpus_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=self.batch_size,
        )

    def encode_queries(self, queries):
        return self.model.encode(
            queries,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=self.batch_size,
        )