"""Shared Supabase pgvector store construction.

Reused by the RAG chain (Phase 2) and the FastAPI backend (Phase 3) so the
client + embeddings + store wiring lives in exactly one place.
"""

import os

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

# nomic-embed-text → 768-dim vectors; must match what Phase 1 stored and the
# vector(768) column / match_documents signature in schema.sql.
EMBEDDING_MODEL = "nomic-embed-text"


def get_vector_store() -> SupabaseVectorStore:
    # Optional: load a local .env if python-dotenv is installed.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ModuleNotFoundError:
        pass

    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )
    return SupabaseVectorStore(
        client=supabase,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        table_name="documents",
        query_name="match_documents",
    )
