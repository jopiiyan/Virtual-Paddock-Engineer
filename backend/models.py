"""One place that decides which LLM and embedding model the app uses.

Two independent switches (env vars), because they have different consequences:

  LLM_PROVIDER        = gemini | ollama   — the answer writer. Safe to change any time;
                                            it only reads retrieved text.
  EMBEDDING_PROVIDER  = gemini | ollama   — turns text into vectors. MUST match the model
                                            the Supabase rows were ingested with, or
                                            similarity search silently returns garbage.
                                            Changing it means: clear the table + re-ingest.

Gemini embeddings are truncated to 768 dims so they fit the existing vector(768)
column (schema.sql) and stay under pgvector's 2000-dim index limit.
Gemini needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
"""

import os

try:  # local runs read .env; in Docker, compose injects it via env_file
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

EMBEDDING_DIM = 768

# Flash-lite: fast and within free-tier quota (full 3.8-flash 429s on a free key;
# 2.5-* models are closed to new keys). Override via GEMINI_LLM_MODEL.
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
OLLAMA_LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _provider(var: str) -> str:
    value = os.getenv(var, "gemini").strip().lower()
    if value not in ("gemini", "ollama"):
        raise ValueError(f"{var} must be 'gemini' or 'ollama', got {value!r}")
    return value


def _gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env to use Gemini.")
    return key


def get_llm():
    if _provider("LLM_PROVIDER") == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_LLM_MODEL,
        google_api_key=_gemini_key(),
        temperature=0,
        # No thinking_budget: flash-lite models reject it (400 invalid argument)
        # and already answer without a long reasoning phase.
    )


def build_embedder():
    if _provider("EMBEDDING_PROVIDER") == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    class Gemini768Embeddings(GoogleGenerativeAIEmbeddings):
        """Always request 768-dim vectors with the right task type.

        output_dimensionality is a per-call argument in langchain-google-genai, and
        SupabaseVectorStore.from_documents calls embed_documents() without it, so we
        pin it here. Documents and queries use different task types — Gemini embeds
        them asymmetrically for better retrieval.
        """

        def embed_documents(self, texts, **kwargs):
            kwargs.setdefault("task_type", "RETRIEVAL_DOCUMENT")
            kwargs.setdefault("output_dimensionality", EMBEDDING_DIM)
            return super().embed_documents(texts, **kwargs)

        def embed_query(self, text, **kwargs):
            kwargs.setdefault("task_type", "RETRIEVAL_QUERY")
            kwargs.setdefault("output_dimensionality", EMBEDDING_DIM)
            return super().embed_query(text, **kwargs)

    return Gemini768Embeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=_gemini_key())
