import os
import random
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from src.data.data_loader import DataLoader as ProjectDataLoader
from src.logger import logging


def build_qrels_dict(qrels_df):
    qrels_dict = {}

    for _, row in qrels_df.iterrows():
        qid = str(row["query_id"])
        docid = str(row["doc_id"])
        relevance = int(row.get("relevance", 0))

        if relevance <= 0:
            continue

        if qid not in qrels_dict:
            qrels_dict[qid] = set()

        qrels_dict[qid].add(docid)

    return qrels_dict


if __name__ == "__main__":
    # Ensure relative paths resolve from project root even when run from elsewhere.
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)

    logging.info("Loading data...")

    loader = ProjectDataLoader(task="task1_retrieval", language="english")
    data = loader.load_all()

    # Build lookup
    corpus_dict = {
        str(row["doc_id"]): str(row["text"])
        for _, row in data["corpus"].iterrows()
    }
    query_dict = {
        str(row["query_id"]): str(row["query"])
        for _, row in data["queries"].iterrows()
    }

    qrels_dict = build_qrels_dict(data["qrels"])

    training_samples = []

    logging.info("Creating training samples...")

    for qid, relevant_docs in qrels_dict.items():
        query_text = query_dict.get(qid)

        if not query_text:
            continue

        # Positive samples
        for doc_id in relevant_docs:
            doc_text = corpus_dict.get(doc_id)
            if doc_text:
                training_samples.append(
                    InputExample(texts=[query_text, doc_text], label=1.0)
                )

        # Negative samples (VERY IMPORTANT)
        all_doc_ids = list(corpus_dict.keys())
        random.shuffle(all_doc_ids)
        for doc_id in all_doc_ids[:5]:
            if doc_id not in relevant_docs:
                training_samples.append(
                    InputExample(
                        texts=[query_text, corpus_dict[doc_id]],
                        label=0.0
                    )
                )

    logging.info(f"Total training samples: {len(training_samples)}")

    if not training_samples:
        raise ValueError("No training samples were created. Check qrels/query alignment.")

    # Load model
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

    train_dataloader = DataLoader(training_samples, shuffle=True, batch_size=16)

    logging.info("Starting training...")

    model.fit(
        train_dataloader=train_dataloader,
        epochs=1,
        warmup_steps=100,
        show_progress_bar=True
    )

    # 🚨 SAVE MODEL (IMPORTANT)
    save_path = str(project_root / "models" / "reranker" / "cross_encoder_finetuned")

    os.makedirs(save_path, exist_ok=True)

    model.save(save_path)

    logging.info(f"Model saved at: {save_path}")