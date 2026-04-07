import json
import os
import re
import shutil
import subprocess
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

try:
    import ollama
except Exception:
    ollama = None


class _OllamaClientCompat:
    """Compatibility wrapper that exposes .invoke(prompt)."""

    def __init__(self, model: str, temperature: float = 0):
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str) -> str:
        if ollama is None:
            raise RuntimeError(
                "Python package 'ollama' is not installed in the current environment."
            )
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
        require_backend: bool = False,
    ):
        self.model_name = model_name
        self.require_backend = require_backend
        self._backend_available = True
        self._backend_error_logged = False
        self.stage1_pool_size = stage1_pool_size
        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k
        self._ollama_process = None

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

    def _is_connection_error(self, error_text: str) -> bool:
        msg = (error_text or "").lower()
        return (
            "failed to connect to ollama" in msg
            or "connection refused" in msg
            or "max retries exceeded" in msg
            or "could not connect" in msg
        )

    def ensure_backend(self):
        """Validate Ollama daemon reachability and model availability."""
        if ollama is None:
            raise RuntimeError(
                "Python package 'ollama' is not installed in the current environment. "
                "Install it with: pip install ollama"
            )

        def _name_variants(name: str) -> set:
            base = str(name or "").strip()
            if not base:
                return set()
            variants = {base}
            if ":" not in base:
                variants.add(f"{base}:latest")
            elif base.endswith(":latest"):
                variants.add(base.rsplit(":", 1)[0])
            return variants

        def _extract_model_names(response_obj) -> set:
            # ollama-python may return either dict-like data or typed objects.
            if isinstance(response_obj, dict):
                models_obj = response_obj.get("models", [])
            else:
                models_obj = getattr(response_obj, "models", [])

            names = set()
            for item in models_obj or []:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("model") or ""
                else:
                    name = getattr(item, "name", None) or getattr(item, "model", "")

                for v in _name_variants(str(name)):
                    names.add(v)
            return names

        def _try_list_models():
            response_local = ollama.list()
            model_names_local = _extract_model_names(response_local)
            return response_local, model_names_local

        response = None
        model_names = set()

        try:
            response, model_names = _try_list_models()
        except Exception as first_error:
            # Try to auto-start local Ollama daemon when OLLAMA_HOST points to localhost.
            host = str(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).lower()
            is_local_host = "127.0.0.1" in host or "localhost" in host
            ollama_exe = shutil.which("ollama")

            if is_local_host and ollama_exe:
                try:
                    log_path = os.path.expanduser("~/ollama.log")
                    log_file = open(log_path, "a", encoding="utf-8")
                    self._ollama_process = subprocess.Popen(
                        [ollama_exe, "serve"],
                        stdout=log_file,
                        stderr=log_file,
                        start_new_session=True,
                    )

                    # Give the daemon a few seconds to boot.
                    for _ in range(10):
                        time.sleep(1)
                        try:
                            response, model_names = _try_list_models()
                            break
                        except Exception:
                            continue

                except Exception as start_error:
                    raise RuntimeError(
                        "Ollama backend is required but failed to start automatically. "
                        "See ~/ollama.log for details."
                    ) from start_error

            if response is None:
                if not ollama_exe:
                    raise RuntimeError(
                        "Ollama CLI not found in PATH. Install Ollama and ensure the binary is available in this shell. "
                        "Current PATH does not resolve 'ollama'."
                    ) from first_error

                raise RuntimeError(
                    "Ollama backend is required but not reachable. "
                    "Start Ollama on this node/session and ensure OLLAMA_HOST is correct."
                ) from first_error

        required_variants = _name_variants(self.model_name)
        if not (required_variants & model_names):
            raise RuntimeError(
                f"Ollama model '{self.model_name}' is not available. "
                f"Pull it first with: ollama pull {self.model_name}"
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
            if self.require_backend:
                raise RuntimeError(
                    "Ollama backend is required but unavailable. "
                    "Reranking stopped to prevent fallback-only results."
                )
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
                if self._is_connection_error(msg):
                    self._backend_available = False
                    err = RuntimeError(
                        "Ollama backend is not reachable. "
                        "Start Ollama and retry."
                    )
                    if self.require_backend:
                        raise err from e
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