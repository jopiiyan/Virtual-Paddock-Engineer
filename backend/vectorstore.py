import os

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import Client, create_client


EMBEDDING_MODEL = "nomic-embed-text"

# Built once and reused — reconnecting / reloading the embedder per query is wasteful.
_CLIENT: Client | None = None
_EMBEDDER: OllamaEmbeddings | None = None


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ModuleNotFoundError:
        pass


def get_client() -> Client:
    global _CLIENT
    if _CLIENT is None:
        _load_env()
        _CLIENT = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    return _CLIENT


def get_embedder() -> OllamaEmbeddings:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _EMBEDDER


def dense_search(query: str, k: int, filter: dict | None = None) -> list[dict]:
    """Cosine kNN over pgvector via the match_documents RPC, called directly.

    We embed with the same model used at ingestion, then call the SQL function
    ourselves rather than going through LangChain's SupabaseVectorStore — that
    wrapper breaks across supabase-py versions (it reaches for a postgrest `.params`
    attribute removed in newer clients), and calling the RPC directly gives explicit,
    stable control over k, the JSONB containment filter, and the cosine similarity.

    Returns the raw RPC rows: dicts with id, content, metadata, similarity.
    """
    embedding = get_embedder().embed_query(query)
    resp = get_client().rpc(
        "match_documents",
        {"query_embedding": embedding, "match_count": k, "filter": filter or {}},
    ).execute()
    return resp.data or []


def fetch_documents(filter: dict | None = None) -> list[dict]:
    """Return every document matching a JSONB-containment filter (id, content,
    metadata). Used to build the in-memory BM25 index over the same corpus subset
    the dense leg searches, so the two legs never see different documents.

    `.contains("metadata", filter)` maps to the same `@>` operator match_documents
    uses, keeping lexical and vector corpora identical.
    """
    q = get_client().table("documents").select("id, content, metadata")
    if filter:
        q = q.contains("metadata", filter)
    rows, start, page = [], 0, 1000
    while True:
        batch = q.range(start, start + page - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < page:
            return rows
        start += page


def get_vector_store() -> SupabaseVectorStore:
    """Kept for backwards compatibility (ingestion inserts through this)."""
    _load_env()
    return SupabaseVectorStore(
        client=get_client(),
        embedding=get_embedder(),
        table_name="documents",
        query_name="match_documents",
    )
