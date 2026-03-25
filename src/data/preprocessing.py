import os
import re
import string
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging


class TextPreprocessor:
    def __init__(self):
        pass

    # Basic Clean Text

    def clean_text(self, text):
        try:
            if not isinstance(text, str):
                return ""

            # Lowercase
            text = text.lower()

            # Remove URLs
            text = re.sub(r"http\S+|www\S+|https\S+", '', text)

            # Remove extra spaces
            text = re.sub(r"\s+", " ", text).strip()

            return text

        except Exception as e:
            raise CustomException(e, sys)

  
    # Light Punctuation Handling

    def handle_punctuation(self, text):
        try:
            # Keep important punctuation for humour
            allowed_punct = ['?', '!', "'", "-"]

            cleaned = ""
            for char in text:
                if char in string.punctuation and char not in allowed_punct:
                    cleaned += " "
                else:
                    cleaned += char

            return cleaned

        except Exception as e:
            raise CustomException(e, sys)

    # Preprocess Single Text

    def preprocess_text(self, text):
        try:
            text = self.clean_text(text)
            text = self.handle_punctuation(text)

            return text

        except Exception as e:
            raise CustomException(e, sys)


    # Apply on DataFrame

    def preprocess_dataframe(self, df, column_name):
        try:
            df[column_name] = df[column_name].apply(self.preprocess_text)

            logging.info(f"Preprocessing done on column: {column_name}")
            return df

        except Exception as e:
            raise CustomException(e, sys)


    # Save DataFrame

    def save_dataframe(self, df, task, language, filename):
        try:
            save_path = os.path.join("data", "processed", task, language)

            # Create directory if not exists
            os.makedirs(save_path, exist_ok=True)

            file_path = os.path.join(save_path, filename)

            df.to_json(file_path, orient="records", indent=4)

            logging.info(f"Saved file at: {file_path}")

        except Exception as e:
            raise CustomException(e, sys)


# MAIN PIPELINE

if __name__ == "__main__":
    try:
        from src.data.data_loader import DataLoader

        TASK = "task1_retrieval"   # change to task2 later
        LANGUAGE = "english"

        # Load Data
        loader = DataLoader(task=TASK, language=LANGUAGE)
        data = loader.load_all()

        # Preprocess
        preprocessor = TextPreprocessor()

        corpus = preprocessor.preprocess_dataframe(data["corpus"], "text")
        queries = preprocessor.preprocess_dataframe(data["queries"], "query")

        # Save Processed Data
        preprocessor.save_dataframe(corpus, TASK, LANGUAGE, "corpus_clean.json")
        preprocessor.save_dataframe(queries, TASK, LANGUAGE, "queries_clean.json")

        print("Preprocessing + Saving Done!")

        print("\nCorpus Sample:\n", corpus.head())
        print("\nQueries Sample:\n", queries.head())

    except Exception as e:
        raise CustomException(e, sys)