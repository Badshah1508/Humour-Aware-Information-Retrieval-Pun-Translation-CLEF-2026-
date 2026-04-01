import json
import os
from models.retrieval.embedding_model import EmbeddingModel
from src.retrieval.dense_retrieval import DenseRetriever
from src.data.data_loader import DataLoader
from src.logger import logging

TASK = "task1_retrieval"
LANGUAGE = "english"
TOP_K = 10

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

        logging.info("Initializing embedding model...")
        embedding_model = EmbeddingModel()

        logging.info("Initializing dense retriever...")
        retriever = DenseRetriever(embedding_model)

        logging.info("Fitting retriever on corpus...")
        retriever.fit(corpus)

        logging.info("Running dense retrieval...")
        results = retriever.search(query_texts, top_k=TOP_K, query_ids=query_ids)

        output_dir = "results/task1_retrieval/english"
        output_path = f"{output_dir}/dense_results.json"
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Saving results to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        logging.info("Dense retrieval pipeline completed successfully")

    except Exception as e:
        logging.exception("Failed to run dense retrieval")
        raise
