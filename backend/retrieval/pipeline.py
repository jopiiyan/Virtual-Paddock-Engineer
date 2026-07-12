"""The single retrieval entry point, driven entirely by PipelineConfig.

Stage order:  dense ─┐
              bm25  ─┴─► RRF fusion ─► multi-query ─► cross-encoder rerank ─► top_n

Only stages enabled in the config run. The baseline (dense-only) and the full
pipeline call THIS function — they differ only by config (R5). Stages that have
not been implemented yet raise NotImplementedError rather than silently no-op, so
the sequencing rule (R4: no feature before the baseline is frozen) is enforced by
the code itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from backend.retrieval.config import PipelineConfig
from backend.vectorstore import get_vector_store

# Cache the vector store across queries within a run (one Supabase connection).
_VECTOR_STORE = None


def _vs():
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = get_vector_store()
    return _VECTOR_STORE


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    metadata: dict
    score: float


@dataclass
class RetrieveResult:
    chunks: list[RetrievedChunk]                 # ranked, best first
    stage_ms: dict[str, float] = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


def _merge_filter(config: PipelineConfig, extra: dict | None) -> dict:
    """Corpus scope always applies; caller-supplied keys refine it further."""
    flt = dict(config.corpus_filter or {})
    if extra:
        flt.update(extra)
    return flt


def _dense(query: str, config: PipelineConfig, flt: dict) -> list[RetrievedChunk]:
    docs_scores = _vs().similarity_search_with_relevance_scores(
        query, k=config.dense.top_k, filter=flt
    )
    out = []
    for doc, score in docs_scores:
        md = doc.metadata or {}
        out.append(RetrievedChunk(
            chunk_id=md.get("chunk_id", str(md.get("id", ""))),
            content=doc.page_content,
            metadata=md,
            score=float(score),
        ))
    return out


def retrieve(query: str, config: PipelineConfig, extra_filter: dict | None = None) -> RetrieveResult:
    stage_ms: dict[str, float] = {}
    flt = _merge_filter(config, extra_filter)

    # --- Stage 1: candidate generation (dense and/or BM25) ---
    legs: list[list[RetrievedChunk]] = []

    if config.dense.enabled:
        t = time.perf_counter()
        dense = _dense(query, config, flt)
        stage_ms["dense_ms"] = (time.perf_counter() - t) * 1000
        legs.append(dense)

    if config.bm25.enabled:
        raise NotImplementedError("BM25 leg lands in Phase 4 (hybrid search).")

    if not legs:
        raise ValueError("No retrieval leg is enabled — at least `dense` must be on.")

    # --- Stage 2: fusion ---
    if config.fusion.method == "rrf":
        raise NotImplementedError("RRF fusion lands in Phase 4 (hybrid search).")
    elif config.fusion.method not in ("none", None):
        raise ValueError(f"Unknown fusion method: {config.fusion.method}")
    ranked = legs[0]   # single leg, no fusion needed

    # --- Stage 3: multi-query expansion ---
    if config.multi_query.enabled:
        raise NotImplementedError("Multi-query expansion lands in Phase 5.")

    # --- Stage 4: cross-encoder rerank ---
    if config.rerank.enabled:
        raise NotImplementedError("Cross-encoder rerank lands in Phase 6.")

    return RetrieveResult(chunks=ranked, stage_ms=stage_ms)
