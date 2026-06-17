"""Phase 3 — FastAPI backend over the Phase 2 RAG chain.

Endpoints
  GET  /api/health   — liveness.
  GET  /api/filters  — distinct drivers / grands prix / sessions for UI dropdowns.
  POST /api/chat      — non-streaming: {answer, driver, sources}.
  POST /api/chat/stream — SSE: a `sources` event, then `token` events, then `done`.

Driver is auto-detected from the question text (Hamilton → HAM); grand_prix and
session_type come from explicit dropdowns. We retrieve docs separately (not via
the StrOutputParser chain) so the retrieved stints can be returned as "receipts".

Run:  uvicorn backend.api:app --reload --port 8000   (from the project root)
"""

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from backend.chain import PROMPT, RETRIEVAL_K, format_docs
from backend.drivers import detect_driver, load_alias_map
from backend.telemetry import compare_telemetry, get_schedule
from backend.vectorstore import get_vector_store

# Build once at startup — these are reused across every request.
VECTOR_STORE = get_vector_store()
LLM = ChatOllama(model="llama3.2", temperature=0)
ALIAS_MAP = load_alias_map()
ANSWER_CHAIN = PROMPT | LLM | StrOutputParser()

# Allow any localhost/127.0.0.1 port in dev — Vite hops to 5174/5175 etc. when a
# port is taken, so pinning exact origins is brittle. Tighten this for production.
app = FastAPI(title="Virtual Paddock Engineer")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    driver: str | None = None        # explicit override; else auto-detected from text
    grand_prix: str | None = None    # dropdown filter
    session_type: str | None = None  # dropdown filter (needs Part 2 re-ingest to take effect)


def resolve_driver(req: ChatRequest) -> str | None:
    """Explicit driver wins; otherwise detect one from the question text."""
    if req.driver:
        return req.driver.upper()
    return detect_driver(req.message, ALIAS_MAP)


def build_filter(driver: str | None, req: ChatRequest) -> dict:
    """JSONB-containment filter for match_documents — only set keys are included."""
    flt: dict = {}
    if driver:
        flt["driver"] = driver
    if req.grand_prix:
        flt["grand_prix"] = req.grand_prix
    if req.session_type:
        flt["session_type"] = req.session_type
    return flt


def retrieve(message: str, flt: dict):
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": RETRIEVAL_K, "filter": flt})
    return retriever.invoke(message)


def docs_to_sources(docs) -> list[dict]:
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/filters")
def filters() -> dict:
    """Distinct filter values for the UI dropdowns, read from stored metadata."""
    rows = VECTOR_STORE._client.table("documents").select("metadata").execute().data
    grands_prix = sorted({r["metadata"].get("grand_prix") for r in rows if r.get("metadata")} - {None})
    sessions = sorted({r["metadata"].get("session_type") for r in rows if r.get("metadata")} - {None})
    # Driver dropdown comes from the alias map (code → a display name).
    drivers = sorted({code for code in ALIAS_MAP.values()})
    return {"drivers": drivers, "grands_prix": grands_prix, "session_types": sessions}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    driver = resolve_driver(req)
    docs = retrieve(req.message, build_filter(driver, req))
    answer = ANSWER_CHAIN.invoke({"context": format_docs(docs), "question": req.message})
    return {"answer": answer, "driver": driver, "sources": docs_to_sources(docs)}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    driver = resolve_driver(req)
    docs = retrieve(req.message, build_filter(driver, req))
    context = format_docs(docs)

    async def event_stream():
        # Receipts first, so the UI can render sources before tokens arrive.
        yield _sse({"type": "sources", "driver": driver, "sources": docs_to_sources(docs)})
        async for tok in ANSWER_CHAIN.astream({"context": context, "question": req.message}):
            yield _sse({"type": "token", "text": tok})
        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/schedule")
def schedule(year: int = 2025) -> dict:
    """Full season calendar for the race dropdowns (all rounds, loaded live)."""
    return {"year": year, "races": get_schedule(year)}


@app.get("/api/telemetry")
def telemetry(drivers: str, grand_prix: str, session_type: str = "R", year: int = 2025) -> dict:
    """Fastest-lap telemetry for overlay charts (speed / throttle+brake / track map).

    `drivers` is a comma-separated list of codes, e.g. ?drivers=HAM,NOR.
    First call for a session is slow (FastF1 downloads telemetry); then cached.
    """
    codes = [c.strip().upper() for c in drivers.split(",") if c.strip()]
    return compare_telemetry(year, grand_prix, session_type, codes)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
