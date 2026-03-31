import os
import sys
import json
import pandas as pd

from src.exception import CustomException
from src.logger import logging


class DataLoader:
    def __init__(self, task="task1_retrieval", language="english"):
        try:
            self.base_path = os.path.join("data", task, language)

            if not os.path.exists(self.base_path):
                raise FileNotFoundError(f"Path not found: {self.base_path}")

            self.files = os.listdir(self.base_path)

            logging.info(f"Base path: {self.base_path}")
            logging.info(f"Available files: {self.files}")

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Load JSON
    # -----------------------------
    def load_json(self, filename):
        try:
            file_path = os.path.join(self.base_path, filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{filename} not found at {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logging.info(f"{filename} loaded successfully")
            return data

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Detect file by keyword
    # -----------------------------
    def find_file(self, keyword):
        try:
            for f in self.files:
                if keyword.lower() in f.lower():
                    return f
            raise FileNotFoundError(f"No file found with keyword: {keyword}")

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Load Corpus
    # -----------------------------
    def load_corpus(self):
        try:
            corpus_file = self.find_file("corpus")
            data = self.load_json(corpus_file)

            corpus = [
                {
                    "doc_id": item.get("docid") or item.get("doc_id"),
                    "text": item.get("text") or item.get("content")
                }
                for item in data
            ]

            df = pd.DataFrame(corpus)
            logging.info(f"Corpus shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Load Queries
    # -----------------------------
    def load_queries(self):
        try:
            # 🔥 ALWAYS use train queries (to match qrels)
            query_file = None

            for f in self.files:
                if "queries_train" in f.lower():
                    query_file = f
                    break

            if query_file is None:
                raise FileNotFoundError("queries_train file not found")

            data = self.load_json(query_file)

            queries = [
                {
                    "query_id": str(item.get("id") or item.get("qid")),
                    "query": item.get("text") or item.get("query")
                }
                for item in data
            ]

            df = pd.DataFrame(queries)

            print(f"\n✅ Using query file: {query_file}")
            print("Total queries:", df.shape[0])

            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    # -----------------------------
    # Load Qrels
    # -----------------------------
    def load_qrels(self):
        try:
            qrels_file = self.find_file("qrels")
            data = self.load_json(qrels_file)

            qrels_list = []

            # 🔥 CASE 1: Dict format (MOST COMMON in CLEF)
            if isinstance(data, dict):
                for qid, docs in data.items():
                    for doc_id, rel in docs.items():
                        qrels_list.append({
                            "query_id": str(qid),
                            "doc_id": str(doc_id),
                            "relevance": int(rel)
                        })

            # 🔥 CASE 2: List format
            elif isinstance(data, list):
                for item in data:
                    qrels_list.append({
                        "query_id": str(item.get("query_id") or item.get("qid")),
                        "doc_id": str(item.get("doc_id") or item.get("docid")),
                        "relevance": int(
                            item.get("relevance")
                            if item.get("relevance") is not None
                            else item.get("label")
                            if item.get("label") is not None
                            else item.get("qrel")
                            if item.get("qrel") is not None
                            else 0
                        )
                    })

            df = pd.DataFrame(qrels_list)

            print("\n✅ Qrels Loaded:")
            print("Total rows:", len(df))
            print("Unique queries:", df["query_id"].nunique())

            logging.info(f"Qrels shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Load Everything
    # -----------------------------

    def load_all(self):
        try:
            return {
                "corpus": self.load_corpus(),
                "queries": self.load_queries(),
                "qrels": self.load_qrels()
            }
        except Exception as e:
            raise CustomException(e, sys)


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    try:
        loader = DataLoader(task="task1_retrieval", language="english")

        data = loader.load_all()

        print("\nCorpus Sample:\n", data["corpus"].head())
        print("\nQueries Sample:\n", data["queries"].head())
        print("\nQrels Sample:\n", data["qrels"].head())

    except Exception as e:
        raise CustomException(e, sys)