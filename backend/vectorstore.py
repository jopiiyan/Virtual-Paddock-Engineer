import os

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client


EMBEDDING_MODEL = "nomic-embed-text"


def get_vector_store() -> SupabaseVectorStore:
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
