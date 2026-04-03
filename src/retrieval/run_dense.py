import json
import os
from src.retrieval.embedding_model import EmbeddingModel
from src.retrieval.dense_retrieval import DenseRetriever
from src.data.data_loader import DataLoader
from src.logger import logging

TASK = "task1_retrieval"
LANGUAGE = "english"
TOP_K = int(os.getenv("DENSE_TOP_K", "100"))
MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
SIMILARITY = os.getenv("DENSE_SIMILARITY", "cosine")
NORMALIZE_EMB = os.getenv("DENSE_NORMALIZE", "true").strip().lower() in {"1", "true", "yes"}
DENSE_BATCH_SIZE = int(os.getenv("DENSE_BATCH_SIZE", "64"))
OUTPUT_FILE = os.getenv("DENSE_OUTPUT_FILE", "dense_results.json")
QUERY_PREFIX = os.getenv("DENSE_QUERY_PREFIX", "")
DOC_PREFIX = os.getenv("DENSE_DOC_PREFIX", "")

if __name__ == "__main__":
    try:
        logging.info("Loading corpus and query data via DataLoader...")
        loader = DataLoader(task=TASK, language=LANGUAGE)
        data = loader.load_all()

        corpus = data["corpus"].to_dict(orient="records")
        queries_df = data["queries"]
        query_texts = queries_df["query"].astype(str).tolist()
        query_ids = queries_df["query_id"].astype(str).tolist()

        logging.info(f"Corpus size: {len(corpus)}")
        logging.info(f"Total queries: {len(query_texts)}")

        logging.info(
            f"Initializing embedding model={MODEL_NAME}, normalize={NORMALIZE_EMB}, similarity={SIMILARITY}"
        )
        embedding_model = EmbeddingModel(
            model_name=MODEL_NAME,
            normalize_embeddings=NORMALIZE_EMB,
            batch_size=DENSE_BATCH_SIZE,
            query_prefix=QUERY_PREFIX,
            doc_prefix=DOC_PREFIX,
        )

        logging.info("Initializing dense retriever...")
        retriever = DenseRetriever(embedding_model, similarity=SIMILARITY)

        logging.info("Fitting retriever on corpus...")
        retriever.fit(corpus)

        logging.info("Running dense retrieval...")
        results = retriever.search(query_texts, top_k=TOP_K, query_ids=query_ids)

        output_dir = "results/task1_retrieval/english"
        output_path = f"{output_dir}/{OUTPUT_FILE}"
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Saving results to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        logging.info("Dense retrieval pipeline completed successfully")

    except Exception as e:
        logging.exception("Failed to run dense retrieval")
        raise
