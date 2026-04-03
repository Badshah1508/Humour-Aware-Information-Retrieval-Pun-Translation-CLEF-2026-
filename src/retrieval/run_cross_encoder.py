import json
import os
import sys
from collections import defaultdict
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.data.data_loader import DataLoader
from src.logger import logging


def load_json(path):
    """Load JSON with error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")


def load_hybrid_results_with_fallback(base_path="results/task1_retrieval/english"):
    """Load hybrid results with fallback to other retrieval methods."""
    candidates = [
        os.path.join(base_path, "hybrid_results.json"),
        os.path.join(base_path, "dense_results.json"),
        os.path.join(base_path, "bm25_results.json"),
        os.path.join(base_path, "rm3_results.json"),
    ]
    
    for path in candidates:
        if os.path.exists(path):
            try:
                results = load_json(path)
                method = os.path.basename(path).replace("_results.json", "")
                print(f"[run_cross_encoder] Loaded {method} results: {len(results)} queries")
                logging.info(f"Using {method} results from {path}")
                return results
            except Exception as e:
                print(f"[run_cross_encoder] Failed to load {path}: {e}")
                logging.warning(f"Failed to load {path}: {e}")
                continue
    
    raise FileNotFoundError(f"No retrieval results found in {base_path}")


def resolve_model_path(model_path):
    """Resolve common nested save structure after fine-tuning."""
    if os.path.isdir(model_path):
        nested = os.path.join(model_path, "cross_encoder_finetuned")
        if os.path.isdir(nested) and os.path.exists(os.path.join(nested, "config.json")):
            return nested

    return model_path


def _to_dict_results(results_obj):
    if isinstance(results_obj, dict):
        return {str(k): v for k, v in results_obj.items()}

    grouped = defaultdict(list)
    for item in results_obj:
        grouped[str(item["query_id"])].append(
            {"doc_id": str(item["doc_id"]), "score": float(item["score"])}
        )
    return dict(grouped)


def merge_candidate_results(result_sets, max_candidates):
    """Merge result sets per query while keeping best score seen for each doc."""
    merged = {}

    for result_set in result_sets:
        current = _to_dict_results(result_set)
        for qid, docs in current.items():
            if qid not in merged:
                merged[qid] = {}

            for rank, doc in enumerate(docs, start=1):
                doc_id = str(doc["doc_id"])
                score = float(doc.get("score", 0.0))

                # Blend score and reciprocal-rank hint to stabilize mixed sources.
                fused_score = score + (1.0 / (60.0 + rank))

                if doc_id not in merged[qid] or fused_score > merged[qid][doc_id]:
                    merged[qid][doc_id] = fused_score

    merged_lists = {}
    for qid, doc_scores in merged.items():
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:max_candidates]
        merged_lists[qid] = [{"doc_id": doc_id, "score": score} for doc_id, score in ranked]

    return merged_lists


if __name__ == "__main__":
    try:
        logging.info("Loading data...")
        print("[run_cross_encoder] Starting cross-encoder rerank pipeline...")

        max_candidates = int(os.getenv("CROSS_ENCODER_CANDIDATES", "100"))
        output_file = os.getenv("CROSS_ENCODER_OUTPUT_FILE", "cross_encoder_finetuned_results.json")
        model_path = os.getenv("CROSS_ENCODER_MODEL_PATH", "models/reranker/cross_encoder_finetuned")

        # Load retrieval results with fallback
        try:
            hybrid_results = load_hybrid_results_with_fallback()
            dense_results = load_json("results/task1_retrieval/english/dense_results.json") if os.path.exists("results/task1_retrieval/english/dense_results.json") else {}
            bm25_results = load_json("results/task1_retrieval/english/bm25_results.json") if os.path.exists("results/task1_retrieval/english/bm25_results.json") else {}
            rm3_results = load_json("results/task1_retrieval/english/rm3_results.json") if os.path.exists("results/task1_retrieval/english/rm3_results.json") else {}

            hybrid_results = merge_candidate_results(
                [hybrid_results, dense_results, bm25_results, rm3_results],
                max_candidates=max_candidates,
            )
        except FileNotFoundError as e:
            print(f"[run_cross_encoder] ERROR: {e}")
            logging.error(str(e))
            sys.exit(1)

        # Load corpus and queries
        try:
            loader = DataLoader(task="task1_retrieval", language="english")
            corpus_df = loader.load_corpus()
            queries_df = loader.load_queries()
            print(f"[run_cross_encoder] Corpus loaded: {len(corpus_df)} docs, queries loaded: {len(queries_df)}")
        except Exception as e:
            print(f"[run_cross_encoder] ERROR loading data: {e}")
            logging.error(f"Failed to load corpus/queries: {e}")
            sys.exit(1)

        corpus = corpus_df.to_dict(orient="records")
        queries = queries_df.to_dict(orient="records")

        # Build doc lookup
        corpus_dict = {str(doc["doc_id"]): doc["text"] for doc in corpus}

        resolved_model_path = resolve_model_path(model_path)
        print(f"[run_cross_encoder] Using model path: {resolved_model_path}")
        reranker = CrossEncoderReranker(model_name=resolved_model_path)

        final_results = {}

        for i, query in enumerate(queries, 1):
            qid = str(query["query_id"])
            query_text = query["query"]

            docs = hybrid_results.get(qid, [])[:max_candidates]

            # Attach text
            docs_with_text = [
                {
                    "doc_id": doc["doc_id"],
                    "text": corpus_dict.get(str(doc["doc_id"]), "")
                }
                for doc in docs
                if corpus_dict.get(str(doc["doc_id"]), "")
            ]

            if not docs_with_text:
                final_results[qid] = []
                continue

            reranked = reranker.rerank(query_text, docs_with_text)

            final_results[qid] = reranked[:10]

            if i <= 2:
                logging.info(f"Sample reranked for query {qid}: {final_results[qid][:2]}")

        # Save results
        output_path = f"results/task1_retrieval/english/{output_file}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)

        logging.info("Cross-Encoder reranking completed.")
        print(f"[run_cross_encoder] Saved reranked results to: {output_path}")
        print(f"[run_cross_encoder] Pipeline complete. Total queries reranked: {len(final_results)}")

    except Exception as e:
        print(f"[run_cross_encoder] FATAL ERROR: {e}")
        logging.exception("Cross-encoder pipeline failed")
        sys.exit(1)
