import os
import sys
import json
from collections import defaultdict

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.data.data_loader import DataLoader
from src.evaluation.retrieval_eval import AdvancedEvaluator
from src.RAG.rag_reranker import RAGReranker


def load_results(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_rag_output_name(result_file):
    if result_file.endswith("_results.json"):
        return result_file.replace("_results.json", "_rag_results.json")
    if result_file.endswith(".json"):
        return result_file.replace(".json", "_rag.json")
    return f"{result_file}_rag_results.json"


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_metrics(scores, path):
    pd.DataFrame([scores]).to_csv(path, index=False)


def group_by_query(results_list):
    from collections import defaultdict

    query_group = defaultdict(list)

    # CASE 1: already dict
    if isinstance(results_list, dict):
        return results_list

    # CASE 2: list of dicts
    for item in results_list:
        if isinstance(item, dict):
            query_group[item["query_id"]].append(item)
        else:
            print("Unexpected item:", item)

    return query_group


def build_text_lookup(corpus_df):
    if corpus_df is None or corpus_df.empty:
        return {}

    return {
        str(row["doc_id"]): str(row["text"])
        for _, row in corpus_df.iterrows()
    }


def apply_rag(query_group, rag, doc_text_lookup, query_text_lookup, strict_llm=False):
    rag_results_list = []
    llm_used_queries = 0
    total_queries = 0

    print("\n Applying RAG reranking...\n")

    for qid, docs in query_group.items():
        total_queries += 1
        try:
            formatted_docs = [
                {
                    "doc_id": doc["doc_id"],
                    "text": doc.get("text") or doc_text_lookup.get(str(doc["doc_id"]), ""),
                    "score": doc["score"]
                }
                for doc in docs
            ]

            # Skip only if all docs are missing text.
            has_any_text = any(d["text"].strip() for d in formatted_docs) if formatted_docs else False
            if len(formatted_docs) == 0 or not has_any_text:
                for doc in docs:
                    rag_results_list.append({
                        "query_id": qid,
                        "doc_id": doc["doc_id"],
                        "score": doc["score"]
                    })
                continue

            query_text = query_text_lookup.get(str(qid), str(qid))
            rag_output = rag.rerank(query_text, formatted_docs)

            if not rag_output:
                if strict_llm:
                    raise RuntimeError(
                        f"RAG returned empty output for query {qid} in strict LLM mode."
                    )
                # If LLM failed/parsing failed, keep original ranking for this query.
                for doc in docs:
                    rag_results_list.append({
                        "query_id": qid,
                        "doc_id": doc["doc_id"],
                        "score": doc["score"]
                    })
                continue

            llm_used_queries += 1
            print(f"\nQuery: {qid} | query_text: {query_text}")
            for r in rag_output:
                print(r["doc_id"], r["llm_score"])

            doc_map = {doc["doc_id"]: doc for doc in formatted_docs}

            for item in rag_output:
                doc_id = item["doc_id"]

                cross_score = float(doc_map[doc_id]["score"])
                llm_score = float(item["llm_score"])

                final_score = 0.7 * cross_score + 0.3 * llm_score

                rag_results_list.append({
                    "query_id": qid,
                    "doc_id": doc_id,
                    "score": final_score
                })

        except Exception as e:
            if strict_llm:
                raise
            print(f"RAG failed for query {qid}: {e}")
            for doc in docs:
                rag_results_list.append({
                    "query_id": qid,
                    "doc_id": doc["doc_id"],
                    "score": doc["score"]
                })

    print(f"\nLLM reranking applied on {llm_used_queries}/{total_queries} queries")
    return rag_results_list


if __name__ == "__main__":
    try:
        TASK = "task1_retrieval"
        LANGUAGE = "english"

        print("\n Starting RAG Evaluation...\n")

        loader = DataLoader(task=TASK, language=LANGUAGE)
        data = loader.load_all()

        corpus_df = data["corpus"]
        queries_df = data["queries"]
        qrels_df = data["qrels"]
        qrels_df["relevance"] = qrels_df["relevance"].fillna(0)

        RESULT_FILE = "rm3_results.json"
        # RESULT_FILE = "cross_encoder_finetuned_results.json"
        output_result_file = build_rag_output_name(RESULT_FILE)
        output_metrics_file = output_result_file.replace(".json", "_metrics.csv")

        results_path = os.path.join(
            "results", TASK, LANGUAGE, RESULT_FILE
        )

        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")

        results_list = load_results(results_path)

        #  Initialize RAG
        STRICT_LLM_RERANK = True

        rag = RAGReranker(
            model_name="phi3:mini",
            stage1_pool_size=10,
            stage1_top_k=5,
            stage2_top_k=3,
            require_backend=STRICT_LLM_RERANK,
        )

        if STRICT_LLM_RERANK:
            print("Strict LLM mode enabled: run will fail if Ollama/model is unavailable.")
            rag.ensure_backend()

        #  Group by query
        query_group = group_by_query(results_list)

        doc_text_lookup = build_text_lookup(corpus_df)
        query_text_lookup = {
            str(row["query_id"]): str(row["query"])
            for _, row in queries_df.iterrows()
        }

        #  Apply RAG
        rag_results = apply_rag(
            query_group=query_group,
            rag=rag,
            doc_text_lookup=doc_text_lookup,
            query_text_lookup=query_text_lookup,
            strict_llm=STRICT_LLM_RERANK,
        )

        output_results_path = os.path.join(
            "results", TASK, LANGUAGE, output_result_file
        )
        output_metrics_path = os.path.join(
            "results", TASK, LANGUAGE, output_metrics_file
        )

        save_json(rag_results, output_results_path)
        print(f"\nSaved RAG reranked results to: {output_results_path}")

        #  Evaluate
        evaluator = AdvancedEvaluator(qrels_df, rag_results)
        scores = evaluator.evaluate()

        save_metrics(scores, output_metrics_path)
        print(f"Saved RAG metrics to: {output_metrics_path}")

        print("\n RAG Evaluation Results:")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

    except Exception as e:
        raise CustomException(e, sys)