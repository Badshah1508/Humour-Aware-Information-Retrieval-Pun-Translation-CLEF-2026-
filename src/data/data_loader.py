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
                    "doc_id": item.get("id"),
                    "text": item.get("text")
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
            # Try multiple possible keywords
            possible_keywords = ["query", "queries", "topic", "question"]

            query_file = None
            for keyword in possible_keywords:
                try:
                    query_file = self.find_file(keyword)
                    break
                except:
                    continue

            if query_file is None:
                raise FileNotFoundError("No query file found")

            data = self.load_json(query_file)

            queries = [
                {
                    "query_id": item.get("id"),
                    "query": item.get("text")
                }
                for item in data
            ]

            df = pd.DataFrame(queries)
            logging.info(f"Queries loaded from {query_file}, shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Load Qrels
    # -----------------------------
    def load_qrels(self):
        try:
            qrels_file = self.find_file("qrels")
            file_path = os.path.join(self.base_path, qrels_file)

            qrels = pd.read_csv(
                file_path,
                sep=r"\s+",
                names=["query_id", "unused", "doc_id", "relevance"]
            )

            qrels = qrels[["query_id", "doc_id", "relevance"]]

            logging.info(f"Qrels shape: {qrels.shape}")
            return qrels

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