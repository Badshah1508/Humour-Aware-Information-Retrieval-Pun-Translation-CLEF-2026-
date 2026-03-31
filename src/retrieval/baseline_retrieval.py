import os
import sys
import json
from collections import Counter
from rank_bm25 import BM25Okapi

from src.exception import CustomException
from src.logger import logging


# =============================
# BM25 RETRIEVER
# =============================
class BM25Retriever:
    def __init__(self, corpus_df):
        try:
            self.corpus_df = corpus_df
            self.texts = corpus_df["text"].tolist()
            self.tokenized_corpus = [doc.split() for doc in self.texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        except Exception as e:
            raise CustomException(e, sys)

    def retrieve(self, query, top_k=5):
        try:
            scores = self.bm25.get_scores(query.split())

            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    "doc_id": self.corpus_df.iloc[idx]["doc_id"],
                    "score": float(scores[idx])
                })

            return results

        except Exception as e:
            raise CustomException(e, sys)


# =============================
# RM3 RETRIEVER
# =============================
class RM3Retriever:
    def __init__(self, corpus_df):
        try:
            self.corpus_df = corpus_df
            self.texts = corpus_df["text"].tolist()
            self.tokenized_corpus = [doc.split() for doc in self.texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        except Exception as e:
            raise CustomException(e, sys)

    def expand_query(self, query, top_docs, top_terms=5):
        try:
            counter = Counter()

            for doc in top_docs:
                counter.update(doc.split())

            expansion_terms = [w for w, _ in counter.most_common(top_terms)]

            return query + " " + " ".join(expansion_terms)

        except Exception as e:
            raise CustomException(e, sys)

    def retrieve(self, query, top_k=5):
        try:
            # Step 1: initial retrieval
            scores = self.bm25.get_scores(query.split())

            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:10]

            top_docs = [self.texts[i] for i in top_indices]

            # Step 2: expand query
            expanded_query = self.expand_query(query, top_docs)

            # Step 3: final retrieval
            scores = self.bm25.get_scores(expanded_query.split())

            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    "doc_id": self.corpus_df.iloc[idx]["doc_id"],
                    "score": float(scores[idx])
                })

            return results

        except Exception as e:
            raise CustomException(e, sys)


# =============================
# SAVE FUNCTION
# =============================
def save_results(results, task, language, filename):
    try:
        path = os.path.join("results", task, language)
        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, filename)

        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Saved: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


# =============================
# MAIN PIPELINE
# =============================
if __name__ == "__main__":
    try:
        from src.data.data_loader import DataLoader
        from src.data.preprocessing import TextPreprocessor

        TASK = "task1_retrieval"
        LANGUAGE = "english"
        TOP_K = 5

        loader = DataLoader(task=TASK, language=LANGUAGE)
        data = loader.load_all()

        corpus_df = data["corpus"]
        queries_df = data["queries"]

        #  PASTE HERE
        qrel_doc_ids = set(data["qrels"]["doc_id"].astype(str))

        corpus_df = corpus_df[
            corpus_df["doc_id"].astype(str).isin(qrel_doc_ids)
        ].copy()

        print("Filtered corpus size:", len(corpus_df))

        pre = TextPreprocessor()

        corpus = pre.preprocess_dataframe(corpus_df.copy(), "text")
        queries = pre.preprocess_dataframe(queries_df.copy(), "query")

        # -------------------------
        # BM25
        # -------------------------
        bm25 = BM25Retriever(corpus)
        bm25_results = []

        # -------------------------
        # RM3
        # -------------------------
        rm3 = RM3Retriever(corpus)
        rm3_results = []

        for _, row in queries.iterrows():
            qid = row["query_id"]
            qtext = row["query"]

            # BM25
            bm25_res = bm25.retrieve(qtext, TOP_K)
            for rank, r in enumerate(bm25_res, 1):
                bm25_results.append({
                    "query_id": str(qid),
                    "doc_id": str(r["doc_id"]),
                    "rank": rank,
                    "score": r["score"]
                })

            # RM3
            rm3_res = rm3.retrieve(qtext, TOP_K)
            for rank, r in enumerate(rm3_res, 1):
                rm3_results.append({
                    "query_id": str(qid),
                    "doc_id": str(r["doc_id"]),
                    "rank": rank,
                    "score": r["score"]
                })

        save_results(bm25_results, TASK, LANGUAGE, "bm25_results.json")
        save_results(rm3_results, TASK, LANGUAGE, "rm3_results.json")

        print("BM25 + RM3 completed!")

    except Exception as e:
        raise CustomException(e, sys)