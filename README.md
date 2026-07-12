# Virtual Paddock Engineer

A full-stack Formula 1 analytics application that lets you **compare two drivers' race performance** through interactive telemetry charts and ask questions about it in **natural language**, answered by a Retrieval-Augmented Generation (RAG) pipeline grounded in real race data.

The entire LLM and embedding stack runs **locally via Ollama** — no external API keys, no per-token cost.

> **Advanced retrieval & evaluation.** On top of the app, this repo builds a measured
> retrieval-evaluation harness (classical IR metrics + RAGAS) over a 40-question
> hand-verified golden set, freezes a naive-RAG baseline, then ablates **hybrid search
> (BM25 + RRF)**, **multi-query expansion**, and **cross-encoder reranking** against it —
> with the latency cost of each. **Every number is reproducible** (`python -m eval.run_eval …`).
> The honest headline: on this small, clean corpus the naive dense baseline is hard to beat,
> and only hybrid clearly earns its place — see **[EVALUATION.md](EVALUATION.md)** for the
> ablation table, **[docs/DECISIONS.md](docs/DECISIONS.md)** for why each component was
> chosen, and **[docs/AUDIT.md](docs/AUDIT.md)** for the starting-state audit.

---

## Picture

<img width="1154" height="791" alt="image" src="https://github.com/user-attachments/assets/87e3e7c8-1bfe-43d8-8cec-f0b340cc3a90" />
<img width="1108" height="845" alt="image" src="https://github.com/user-attachments/assets/38980c5f-e350-41ba-bdac-c6337759a4de" />

---

## Features

- **Comparative driver dashboard** — pick two drivers and compare lap pace, tyre degradation, and top speed across a race through interactive charts.
- **RAG chat assistant** — ask questions like _"How did Hamilton's hard tyres hold up?"_ and get answers grounded only in retrieved race data.
- **Source "receipts"** — every answer shows the exact stint records it was based on, so nothing is taken on faith.
- **Streaming responses** — answers stream token-by-token over Server-Sent Events (SSE).
- **Fully local inference** — LLM (`llama3.2`) and embeddings (`nomic-embed-text`) run on Ollama; zero external API cost.

---

## Architecture

```
 ┌────────────────┐   HTTP / SSE   ┌──────────────────────────┐        ┌──────────────────────┐
 │ React Frontend │ <───────────>  │ FastAPI Backend          │ <────> │   Supabase Cloud     │
 │  (dashboard +  │                │ (LangChain LCEL pipeline)│        │ (PostgreSQL + vector)│
 │   RAG chat)    │                └────────────┬─────────────┘        └──────────────────────┘
 └────────────────┘                             │
                                        ┌────────▼────────┐
                                        │  Local Ollama   │
                                        │ llama3.2 +      │
                                        │ nomic-embed-text│
                                        └─────────────────┘
```

**Data flow at query time:** question → embed → vector search (`match_documents`) returns top-k stint summaries → injected into a context-grounded prompt → local LLM streams the answer back, alongside the source records.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Web Streams API (SSE consumption) |
| Backend | FastAPI, Server-Sent Events |
| Orchestration | LangChain (LCEL) |
| Vector store | Supabase (PostgreSQL + pgvector, HNSW index) |
| LLM & embeddings | Ollama — `llama3.2`, `nomic-embed-text` (768-dim) |
| Data source | FastF1 |

---

## How the RAG pipeline works

1. **Ingestion** — `FastF1` telemetry is parsed into one record *per driver, per stint*. Each record is written as a natural-language summary (e.g. _"In the 2025 Silverstone GP, HAM ran stint 2 on Hard tyres over 14 laps: average lap 1:31.420, top speed 312 km/h, tyre degradation +0.080 s/lap."_). These summaries — not raw numbers — are what gets embedded, so they retrieve well against natural-language questions.
2. **Storage** — each summary is embedded with `nomic-embed-text` (768-dim) and stored in a Supabase `documents` table, with structured tags (`driver`, `year`, `grand_prix`, `stint`, `compound`) in a `jsonb` metadata column for exact filtering.
3. **Retrieval** — at query time, the question is embedded and matched against stored vectors via a `match_documents` SQL function (cosine distance, accelerated by an HNSW index), with optional metadata filtering.
4. **Generation** — retrieved summaries are composed into a strict, context-grounded prompt and answered by a local LLM through a LangChain LCEL chain. The prompt forbids outside knowledge and instructs the model to say so when the answer isn't in the data.

---

## Getting Started

### Prerequisites

- [Ollama](https://ollama.com/) installed and running
- A [Supabase](https://supabase.com/) project
- Python 3.10+ and Node.js 18+

### 1. Pull the local models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Set up the Supabase schema

In the Supabase SQL editor, run:

```sql
-- Enable pgvector
create extension if not exists vector;

-- Documents table (column names are what LangChain's SupabaseVectorStore expects)
create table documents (
    id bigserial primary key,
    content text,
    metadata jsonb,
    embedding vector(768)        -- matches nomic-embed-text
);

-- Similarity-search function used by the retriever
create function match_documents (
    query_embedding vector(768),
    match_count int default null,
    filter jsonb default '{}'
) returns table (
    id bigint,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
    return query
    select id, content, metadata,
           1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    where metadata @> filter
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- ANN index for fast vector search
create index on documents using hnsw (embedding vector_cosine_ops);
```

### 3. Configure environment variables

Create a `.env` file in the backend directory:

```env
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_SERVICE_KEY=<your-service-role-key>
INGEST_YEAR=2025
INGEST_GP=Silverstone
```

> ⚠️ The service-role key is backend-only — never expose it to the frontend or commit it to git. Add `.env` to `.gitignore`.

### 4. Run the backend

```bash
cd backend
pip install -r requirements.txt

# Ingest a race into Supabase (one-time, per race)
python -m backend.ingestion

# Start the API
uvicorn backend.api:app --reload --port 8000
```

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open the dev server URL (e.g. `http://localhost:5173`), pick two drivers, and start asking questions.

---

## Reproduce the evaluation

Every number in [EVALUATION.md](EVALUATION.md) is produced by a script — nothing is estimated.

```bash
cd backend && pip install -r requirements.txt && cd ..

python eval/validate_golden.py                 # golden-set integrity check
python -m pytest backend/tests                 # metric / RRF / pipeline unit tests

# Freeze the baseline, then run each ablation config (RAGAS off; add generation for RAGAS):
python -m eval.run_eval --config configs/baseline.yaml     --out eval/results/baseline.json     --no-ragas
python -m eval.run_eval --config configs/bm25_only.yaml    --out eval/results/bm25_only.json    --no-generation
python -m eval.run_eval --config configs/hybrid.yaml       --out eval/results/hybrid.json       --no-generation
python -m eval.run_eval --config configs/hybrid_mq.yaml    --out eval/results/hybrid_mq.json    --no-generation
python -m eval.run_eval --config configs/hybrid_rerank.yaml --out eval/results/hybrid_rerank.json --no-generation
python -m eval.run_eval --config configs/full.yaml         --out eval/results/full.json         --no-ragas

python scripts/plot_tradeoff.py                # regenerate docs/tradeoff.png
```

Retrieval metrics (recall@k, MRR, nDCG) need only Ollama + Supabase and are deterministic.
The **RAGAS** answer-quality columns additionally need a `GEMINI_API_KEY` with judge quota
(see [EVALUATION.md](EVALUATION.md)); the harness records them as pending otherwise.

---

## Project Structure

```
.
├── backend/
│   ├── ingestion.py         # FastF1 → LangChain Documents → Supabase (stable chunk_ids)
│   ├── vectorstore.py       # Supabase client + dense_search (direct match_documents RPC)
│   ├── chain.py             # LCEL RAG chain (standalone-runnable)
│   ├── drivers.py           # "Hamilton" → HAM detection
│   ├── telemetry.py         # FastF1 telemetry for the Compare dashboard (no RAG)
│   ├── api.py               # FastAPI app (/api/chat, SSE streaming)
│   ├── retrieval/           # config-driven pipeline: dense · bm25 · fusion(RRF) · multi_query · rerank
│   ├── tests/               # pytest: metrics, RRF, tokenization, pipeline guardrails
│   ├── requirements.txt     # pinned runtime deps
│   └── requirements-test.txt# lean deps for CI
├── configs/                 # baseline.yaml, hybrid.yaml, … full.yaml (one per ablation row)
├── eval/                    # golden_set.jsonl, metrics.py, run_eval.py, ragas_eval.py, results/
├── scripts/                 # inspect_corpus, backfill_chunk_ids, generate_candidates, plot_tradeoff
├── docs/                    # AUDIT.md, DECISIONS.md, tradeoff.png
├── EVALUATION.md            # ← the ablation table + failure analysis + threats to validity
├── frontend/                # React dashboard + chat UI
└── .github/workflows/ci.yml # pytest on push
```

---

## What's been built and measured

Built, ablated against the frozen baseline, and reported in [EVALUATION.md](EVALUATION.md):

- [x] **Evaluation harness** — 40-question hand-verified golden set + classical IR metrics
  (recall@k, MRR, nDCG) and a RAGAS integration (Gemini judge, independent of the generator).
- [x] **Hybrid search (BM25 + RRF)** — closes the exact-term gap (compound/code queries) and
  lifts recall@10 for near-zero cost. **The one component that clearly earns its place.**
- [x] **Multi-query expansion** (paraphrase + decomposition) — helps paraphrase queries but
  regresses comparatives at ~15× latency; kept as a flag, **off by default**.
- [x] **Cross-encoder reranking** (`bge-reranker-base`) with a candidate-set sweep — a *negative*
  result on this corpus (out-of-domain reranker hurts exact-term); reported honestly.
- [x] **Conversational memory** — history-aware follow-ups (in the API).

The honest conclusion (with numbers): on this small, semantically-clean corpus the naive dense
baseline is hard to beat, and only hybrid is worth shipping. See EVALUATION.md for the full
table, per-type breakdown, latency/accuracy plot, and threats to validity.

## Roadmap — what I'd build next

The residual failures are *analytical/structured* queries that semantic retrieval is the wrong
tool for, so the next steps are:

- [ ] **Query router + text-to-SQL** — send analytical/aggregation questions ("who had the worst
  degradation?", "which compound?") to a SQL path over the structured stint data. No amount of
  reranking fixes these; this is the honest next build.
- [ ] **Self-query retrieval** — derive metadata filters from natural language automatically
  (e.g. "Hamilton's hard-tyre stints" → `{driver: HAM, compound: HARD}`).
- [ ] **Domain-fit reranker** — fine-tune the cross-encoder on F1 relevance, or a metadata-aware
  signal that knows a "which compound" question wants a stint doc, not a result doc.
- [ ] **Larger corpus** — extend the eval beyond one race; several ablation calls would likely flip.

---

## Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) for the Formula 1 telemetry data.
- [LangChain](https://github.com/langchain-ai/langchain), [Supabase](https://supabase.com/), and [Ollama](https://ollama.com/) for the RAG, storage, and local-inference stack.
