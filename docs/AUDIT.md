# Phase 0 — Current-State Audit

> Read the repo, report the facts. **No features written in this phase.** This is
> the ground truth every later phase is measured against. Findings below are from
> the committed code; the one thing that lives only in the remote database (the
> actual indexed document count) is recovered by `scripts/inspect_corpus.py`.

## 1. The current retrieval call

- **top-k = 4.** Defined once as `RETRIEVAL_K = 4` (`backend/chain.py:13`) and reused
  in both the standalone chain (`chain.py:37`) and the API
  (`backend/api.py:112`, `search_kwargs={"k": RETRIEVAL_K, "filter": flt}`).
- **Similarity = cosine.** Computed server-side in the `match_documents` RPC as
  `1 - (documents.embedding <=> query_embedding)` and ordered by the pgvector
  cosine-distance operator `<=>` (`backend/schema.sql:27,30`).
- **Metadata filter = JSONB containment `@>`.** `match_documents` filters with
  `where metadata @> filter` (`schema.sql:29`). The filter dict is assembled in
  `build_filter()` (`api.py:99-108`) from only the keys that are set — an empty
  `{}` matches everything. Driver is auto-resolved (explicit → detected → last
  mentioned; `resolve_driver`, `api.py:73-87`); `grand_prix`/`session_type` come
  from UI dropdowns.
- **Caveat:** `session_type` filtering is documented as needing a re-ingest to take
  effect (`api.py:69`) — the corpus is Race-only, so the key is present but
  single-valued.

## 2. Schema and vector index

`backend/schema.sql` is the only SQL file (no migrations dir).

```sql
create table documents (
    id bigserial primary key,
    content text,
    metadata jsonb,
    embedding vector(768)
);
```

- `match_documents(query_embedding vector(768), match_count int default null, filter jsonb default '{}')`
  returns `(id, content, metadata, similarity)`.
- **HNSW index exists on `vector_cosine_ops`** (`schema.sql:36`) — matches the `<=>`
  operator, so the `ORDER BY` is index-eligible.
- **No GIN index on `metadata`** despite the `@>` containment filter in the hot
  path — filtering is an unindexed scan / HNSW post-filter. (Acceptable at this
  corpus size; noted for honesty.)
- **`match_count` SQL default is `NULL`** (i.e. `LIMIT NULL` = unbounded) if ever
  called without a count. LangChain always passes `k`, so in practice it is 4.

## 3. Embeddings

- `nomic-embed-text` (768-dim) via Ollama, the **same model at ingest and query**
  (`backend/vectorstore.py:8,24`; `backend/ingestion.py:207`). The 768 dimension is
  enforced by the column type and the RPC signature. No `base_url` set → Ollama
  defaults to `http://localhost:11434`.

## 4. Chunking strategy

Two builders in `backend/ingestion.py`, both **one-document-per-entity** (never
length-based splitting):

- **Stint summaries** (`build_stint_documents`, `ingestion.py:46-106`): per-driver,
  per-stint. `pick_quicklaps()` drops in/out/safety-car laps; stints with < 3 laps
  are skipped. Each doc is an NL summary (avg/best lap, sector times, trap speeds,
  degradation slope) ≈ **75–95 tokens**.
- **Race results** (`build_result_documents`, `ingestion.py:121-185`): one doc per
  driver (finish position / time / gap / grid / points), tagged
  `doc_type="result"` ≈ **30–45 tokens**.

Metadata per stint doc: `driver, year, grand_prix, session_type, stint, compound`.
Per result doc: `driver, year, grand_prix, session_type, doc_type="result",
position`.

## 5. Corpus scope (defines what the golden set may contain)

The ingestion code path writes **exactly one race per run** from env with hardcoded
defaults (`ingestion.py:198-199`): `INGEST_YEAR=2025`, `INGEST_GP=Silverstone`,
session `R` only. There is **no multi-race loop** in code.

**But the live database is not the code default.** Running
`scripts/inspect_corpus.py` against Supabase (2026-07-13) shows the corpus was
built by running ingestion **four times** with `INGEST_GP` overrides:

| grand_prix (2025, session R) | documents |
|---|---|
| Monaco | 68 |
| Monza | 54 |
| Spa-Francorchamps | 46 |
| Silverstone | 35 |
| **Total** | **203** |

- **203 documents** = **123 stint** docs + **80 result** docs (4 races × 20 drivers).
- **All 2025, all Race session, all 20 drivers**
  (`ALB, ALO, ANT, BEA, BOR, COL, GAS, HAD, HAM, HUL, LAW, LEC, NOR, OCO, PIA, RUS,
  SAI, STR, TSU, VER`).
- Compounds present: `HARD=44, MEDIUM=65, SOFT=10`, plus 4 stint docs with a null
  compound (a FastF1 data quirk to be aware of when writing exact-term questions).
- No document carries a deterministic `chunk_id` yet (Phase 1 adds it).

**Chosen scope (reconciled with corpus owner, 2026-07-13): Silverstone only.**
The golden set and **every eval run** are scoped to `grand_prix=Silverstone`
(35 docs ≈ 15 stint + 20 result). Monaco / Monza / Spa remain indexed but untested.
Consequence for the pipeline (Phase 2): a fixed corpus filter
`{"grand_prix": "Silverstone"}` must be applied to **both** the dense leg (via the
`match_documents` `filter` arg) and the BM25 leg (build the BM25 index over only
the Silverstone subset), so the other three races cannot leak into retrieval and
distort recall/precision. The 15-stint pool is thin for stint-vs-stint comparatives
— result docs (20) carry the finishing-order comparatives; keep this in mind when
authoring the comparative bucket in Phase 1.

## 6. Generation, chain, streaming

- Generator: `llama3.2` via `ChatOllama`, `temperature=0` (`chain.py:38`,
  `api.py:45`).
- LCEL: standalone chain wires retriever→prompt→llm→parser (`chain.py:36-44`); the
  API retrieves **separately** (so sources can be returned as "receipts") and runs
  `ANSWER_CHAIN = CHAT_PROMPT | LLM | StrOutputParser()` (`api.py:47`).
- **SSE streaming: yes.** `/api/chat/stream` (`api.py:148-165`) emits `sources`
  (receipts first) → `token`* → `done`, framed as `data: {json}\n\n`.

## 7. FastAPI backend structure

Single entry point `backend/api.py` (app at `api.py:51`). Endpoints:
`GET /api/health`, `GET /api/filters`, `POST /api/chat`, `POST /api/chat/stream`,
`GET /api/schedule`, `GET /api/telemetry`. Supporting modules: `drivers.py`
(alias/detection), `telemetry.py` (`compare_telemetry`, `get_schedule`),
`vectorstore.py` (store factory), `ingestion.py` (offline ETL).

## 8. React comparison dashboard (Resume bullet 1)

**Built and functional** (not a stub). `frontend/src/Compare.jsx` renders a
GP + Driver A/B picker and four Recharts views — Speed, Throttle, Speed-delta (with
a zero reference line), and a hand-rolled SVG track map — over the fastest-lap
telemetry, aligned on a shared 400-point distance grid (`telemetry.py`). React 18 +
Vite + Recharts `^3.8.1`. **No frontend work is in scope for this build.**

## 9. Existing tests / CI

**None.** No `tests/`, no `pytest` config, no `.github/workflows`, no Makefile.
`requirements.txt` is **unpinned** and lacks `ragas`, `rank_bm25`,
`sentence-transformers`, and any Gemini SDK. `.env` holds only `SUPABASE_URL` /
`SUPABASE_SERVICE_KEY` (no judge-model key). No `docs/`, `eval/`, or `configs/`
directories existed before this build.

## Gaps to close (tracked, not fixed here)

| Gap | Closed in |
|---|---|
| No eval harness / metrics | Phase 2 |
| No golden set | Phase 1 |
| No measured baseline | Phase 3 |
| Retrieval hard-wired (not config-driven) | Phase 2 (pipeline refactor) |
| Chunk ids not stable across re-ingest | Phase 1 (deterministic `chunk_id`) |
| No BM25 / hybrid / MQ / rerank | Phases 4–6 |
| No tests / CI | Phase 8 |
| Unpinned deps, README `main.py`→`api.py` mismatch | Phase 8 |

## Reproducing the live-corpus facts

```bash
python scripts/inspect_corpus.py
```

Requires `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` in `.env`. Prints the total doc
count and the distinct `driver` / `grand_prix` / `session_type` / `doc_type`
values actually indexed — the numbers that cannot be known from source alone.
