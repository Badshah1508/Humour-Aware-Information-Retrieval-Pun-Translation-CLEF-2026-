import json
import re
import time
from typing import List, Dict

try:
    from langchain_community.llms import Ollama as _CommunityOllama
except Exception:
    _CommunityOllama = None

try:
    from langchain_ollama import OllamaLLM as _OllamaLLM
except Exception:
    _OllamaLLM = None

import ollama


class _OllamaClientCompat:
    """Compatibility wrapper that exposes .invoke(prompt)."""

    def __init__(self, model: str, temperature: float = 0):
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str) -> str:
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature}
        )
        return response.get("response", "")


class RAGReranker:
    def __init__(
        self,
        model_name: str = "phi3:mini",
        stage1_pool_size: int = 10,
        stage1_top_k: int = 5,
        stage2_top_k: int = 3,
    ):
        self._backend_available = True
        self._backend_error_logged = False
        self.stage1_pool_size = stage1_pool_size
        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k

        if _CommunityOllama is not None:
            self.llm = _CommunityOllama(
                model=model_name,
                temperature=0  # deterministic
            )
        elif _OllamaLLM is not None:
            self.llm = _OllamaLLM(
                model=model_name,
                temperature=0
            )
        else:
            self.llm = _OllamaClientCompat(
                model=model_name,
                temperature=0
            )

    # -----------------------------
    # Format context
    # -----------------------------
    def format_context(self, docs: List[Dict]) -> str:
        context = ""
        for i, doc in enumerate(docs):
            context += f"[{i}] {doc['text']}\n\n"
        return context

    # -----------------------------
    # Safe LLM call with retry
    # -----------------------------
    def _to_text(self, response) -> str:
        if isinstance(response, str):
            return response

        # LangChain message-like objects often expose `.content`.
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content

        return str(response)

    def _parse_json_response(self, response):
        response_text = self._to_text(response)

        # Try strict JSON first.
        try:
            return json.loads(response_text)
        except Exception:
            pass

        # Remove markdown fences if present.
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Extract the first decodable JSON object/array from mixed text.
        decoder = json.JSONDecoder()
        for i, ch in enumerate(cleaned):
            if ch not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[i:])
                return parsed
            except Exception:
                continue

        raise ValueError("No valid JSON found in LLM response")

    def safe_llm_call(self, prompt, max_retries=3):
        if not self._backend_available:
            return None

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)

                # Try parsing JSON
                parsed = self._parse_json_response(response)
                return parsed

            except Exception as e:
                msg = str(e)
                # If the local Ollama daemon is unavailable, retrying each query is wasteful.
                if "Failed to connect to Ollama" in msg:
                    self._backend_available = False
                    if not self._backend_error_logged:
                        print("Ollama backend is not reachable. Skipping LLM reranking for remaining queries.")
                        self._backend_error_logged = True
                    return None

                print(f"Retry {attempt+1} failed: {e}")
                time.sleep(1)

        # fallback
        return None

    # -----------------------------
    # Stage 1 (fast filtering)
    # -----------------------------
    def stage1_filter(self, query, docs):
        context = self.format_context(docs)

        prompt = f"""
You are a ranking system.

Query: {query}

Documents:
{context}

Select top {self.stage1_top_k} most relevant documents.

Return JSON:
{{"indices": [list of selected indices]}}
"""

        result = self.safe_llm_call(prompt)

        if result and "indices" in result:
            selected = [docs[i] for i in result["indices"] if i < len(docs)]
            return selected[: self.stage1_top_k]

        return docs[: self.stage1_top_k]  # fallback

    # -----------------------------
    # Stage 2 (deep reranking)
    # -----------------------------
    def stage2_rerank(self, query, docs):
        context = self.format_context(docs)

        prompt = f"""
You are an expert in humor ranking.

Query: {query}

Documents:
{context}

Rank top {self.stage2_top_k} documents based on:
- humor quality
- relevance

Return JSON:
{{
  "results": [
    {{"index": 0, "score": 0.9, "explanation": "..."}}
  ]
}}
"""

        result = self.safe_llm_call(prompt)

        if not result or "results" not in result:
            return []

        output = []
        for item in result["results"]:
            idx = item.get("index", 0)

            if idx < len(docs):
                doc = docs[idx]
                output.append({
                    "doc_id": doc["doc_id"],
                    "text": doc["text"],
                    "llm_score": item.get("score", 0),
                    "explanation": item.get("explanation", "")
                })

        return output

    # -----------------------------
    # Main RAG Pipeline
    # -----------------------------
    def rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        if len(docs) == 0:
            return []

        # Stage 1
        stage1_docs = self.stage1_filter(query, docs[: self.stage1_pool_size])

        # Stage 2
        final_results = self.stage2_rerank(query, stage1_docs[: self.stage2_top_k])

        return final_results