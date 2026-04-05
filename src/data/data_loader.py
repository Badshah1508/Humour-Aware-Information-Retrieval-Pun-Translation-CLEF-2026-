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

            corpus = []
            missing_count = 0

            for item in data:
                if item is None:
                    missing_count += 1
                    continue

                doc_id = item.get("docid") or item.get("doc_id")
                text = item.get("text") or item.get("content")

                if doc_id is None or text is None:
                    missing_count += 1
                    continue

                text = str(text).strip()
                if text == "":
                    missing_count += 1
                    continue

                corpus.append({
                    "doc_id": str(doc_id),
                    "text": text
                })

            df = pd.DataFrame(corpus)
            logging.info(f"Corpus shape: {df.shape}")
            if missing_count > 0:
                logging.warning(f"Skipped {missing_count} corpus entries with missing or empty text")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Load Queries
    # -----------------------------
    def _resolve_query_files(self, split):
        split = str(split or "train").strip().lower()

        train_files = [f for f in self.files if "queries_train" in f.lower()]
        test_files = [f for f in self.files if "queries_test" in f.lower()]

        if split == "train":
            if not train_files:
                raise FileNotFoundError("queries_train file not found")
            return train_files

        if split == "test":
            if not test_files:
                raise FileNotFoundError("queries_test file not found")
            return test_files

        if split == "all":
            files = train_files + test_files
            if not files:
                raise FileNotFoundError("No train/test query files found")
            return files

        raise ValueError("Invalid query split. Use one of: train, test, all")

    def load_queries(self, split="train"):
        try:
            query_files = self._resolve_query_files(split=split)

            queries = []
            for query_file in query_files:
                data = self.load_json(query_file)
                for item in data:
                    qid = item.get("id") or item.get("qid")
                    qtext = item.get("text") or item.get("query")

                    if qid is None or qtext is None:
                        continue

                    queries.append({
                        "query_id": str(qid),
                        "query": str(qtext)
                    })

            df = pd.DataFrame(queries).drop_duplicates(subset=["query_id"], keep="first")

            print(f"\n Using query split: {str(split).lower()}")
            print(f"Query files: {query_files}")
            print("Total unique queries:", df.shape[0])

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

            print("\n Qrels Loaded:")
            print("Total rows:", len(df))
            print("Unique queries:", df["query_id"].nunique())

            logging.info(f"Qrels shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Load Everything
    # -----------------------------

    def load_all(self, query_split="train"):
        try:
            return {
                "corpus": self.load_corpus(),
                "queries": self.load_queries(split=query_split),
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