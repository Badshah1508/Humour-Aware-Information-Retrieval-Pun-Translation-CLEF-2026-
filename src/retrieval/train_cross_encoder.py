import os
from pathlib import Path
import random
import json
import torch

from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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


def load_hybrid_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_result_file(path):
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}

    grouped = {}
    for item in data:
        qid = str(item.get("query_id") or item.get("qid"))
        grouped.setdefault(qid, []).append(
            {"doc_id": str(item["doc_id"]), "score": float(item.get("score", 0.0))}
        )
    return grouped


def load_candidate_pool(result_paths):
    merged = {}

    for path in result_paths:
        current = _load_result_file(path)
        if not current:
            continue

        logging.info(f"Loaded candidate file for negatives: {path}")

        for qid, docs in current.items():
            merged.setdefault(str(qid), [])
            seen = {str(d["doc_id"]) for d in merged[str(qid)]}

            for doc in docs:
                doc_id = str(doc["doc_id"])
                if doc_id in seen:
                    continue
                merged[str(qid)].append({"doc_id": doc_id, "score": float(doc.get("score", 0.0))})
                seen.add(doc_id)

    # Prioritize harder negatives by score.
    for qid in merged:
        merged[qid] = sorted(merged[qid], key=lambda x: float(x.get("score", 0.0)), reverse=True)

    return merged


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)
    random.seed(42)

    ce_model_name = os.getenv("CE_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    ce_epochs = int(os.getenv("CE_EPOCHS", "8"))
    ce_batch_size = int(os.getenv("CE_BATCH_SIZE", "16"))
    ce_eval_steps = int(os.getenv("CE_EVAL_STEPS", "200"))
    ce_warmup_ratio = float(os.getenv("CE_WARMUP_RATIO", "0.1"))
    max_hard_negs = int(os.getenv("CE_MAX_HARD_NEGS", "40"))
    max_random_negs = int(os.getenv("CE_MAX_RANDOM_NEGS", "10"))
    ce_num_workers = int(os.getenv("CE_NUM_WORKERS", "0"))
    ce_pin_memory = os.getenv("CE_PIN_MEMORY", "auto").strip().lower()
    query_split = os.getenv("RETRIEVAL_QUERY_SPLIT", "all").strip().lower()
    val_ratio = float(os.getenv("CE_VAL_RATIO", "0.1"))

    logging.info("Loading data...")

    loader = ProjectDataLoader(task="task1_retrieval", language="english")
    data = loader.load_all(query_split=query_split)

    corpus_dict = {
        str(row["doc_id"]): str(row["text"])
        for _, row in data["corpus"].iterrows()
    }

    query_dict = {
        str(row["query_id"]): str(row["query"])
        for _, row in data["queries"].iterrows()
    }

    qrels_dict = build_qrels_dict(data["qrels"])
    candidate_files = [
        "results/task1_retrieval/english/hybrid_results_tuned.json",
        "results/task1_retrieval/english/hybrid_results.json",
        "results/task1_retrieval/english/dense_results.json",
        "results/task1_retrieval/english/bm25_results.json",
        "results/task1_retrieval/english/rm3_results.json",
    ]
    merged_candidates = load_candidate_pool(candidate_files)
    all_doc_ids = list(corpus_dict.keys())

    training_samples = []
    positive_count = 0
    hard_negative_count = 0
    random_negative_count = 0

    logging.info("Creating samples with multi-source hard negatives + random negatives...")
    logging.info(f"Using query split: {query_split}")

    for qid, relevant_docs in qrels_dict.items():
        query_text = query_dict.get(qid)
        if not query_text:
            continue

        # ✅ Positive
        for doc_id in relevant_docs:
            doc_text = corpus_dict.get(doc_id)
            if doc_text:
                training_samples.append(
                    InputExample(texts=[query_text, doc_text], label=1.0)
                )
                positive_count += 1

        # Hard negatives from merged retrieval candidates
        retrieved_docs = merged_candidates.get(qid, [])
        hard_neg_count = 0
        sampled_neg_ids = set()

        for doc in retrieved_docs:
            doc_id = str(doc["doc_id"])

            if doc_id not in relevant_docs:
                doc_text = corpus_dict.get(doc_id)

                if doc_text:
                    training_samples.append(
                        InputExample(
                            texts=[query_text, doc_text],
                            label=0.0
                        )
                    )
                    hard_neg_count += 1
                    hard_negative_count += 1
                    sampled_neg_ids.add(doc_id)

            if hard_neg_count >= max_hard_negs:
                break

        # Random negatives for better diversity
        random_neg_count = 0
        if max_random_negs > 0:
            random.shuffle(all_doc_ids)
            for doc_id in all_doc_ids:
                if doc_id in relevant_docs or doc_id in sampled_neg_ids:
                    continue

                doc_text = corpus_dict.get(doc_id)
                if not doc_text:
                    continue

                training_samples.append(
                    InputExample(texts=[query_text, doc_text], label=0.0)
                )
                random_neg_count += 1
                random_negative_count += 1

                if random_neg_count >= max_random_negs:
                    break

    logging.info(f"Total samples: {len(training_samples)}")
    logging.info(
        f"Positive={positive_count}, hard_negatives={hard_negative_count}, random_negatives={random_negative_count}"
    )

    if not training_samples:
        raise ValueError("No training samples created.")

    # =========================
    # 🔥 TRAIN / VALID SPLIT
    # =========================
    train_samples, val_samples = train_test_split(
        training_samples,
        test_size=val_ratio,
        random_state=42
    )

    logging.info(f"Train size: {len(train_samples)}")
    logging.info(f"Validation size: {len(val_samples)}")

    # =========================
    # 🔥 MODEL
    # =========================
    model = CrossEncoder(ce_model_name, num_labels=1)

    if ce_pin_memory == "auto":
        pin_memory = torch.cuda.is_available()
    else:
        pin_memory = ce_pin_memory in {"1", "true", "yes"}

    train_dataloader = DataLoader(
        train_samples,
        shuffle=True,
        batch_size=ce_batch_size,
        num_workers=ce_num_workers,
        pin_memory=pin_memory,
    )

    # =========================
    # 🔥 VALIDATION EVALUATOR
    # =========================
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        val_samples,
        name="validation"
    )

    logging.info("Starting training with validation...")

    total_steps = max(len(train_dataloader) * ce_epochs, 1)
    warmup_steps = int(total_steps * ce_warmup_ratio)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        evaluation_steps=ce_eval_steps,
        epochs=ce_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True
    )

    # =========================
    # 🚨 SAVE MODEL
    # =========================
    save_path = str(project_root / "models" / "reranker" / "cross_encoder_finetuned")

    os.makedirs(save_path, exist_ok=True)

    model.save(save_path)

    logging.info(f"Model saved at: {save_path}")