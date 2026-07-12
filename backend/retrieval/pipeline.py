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

from backend.retrieval.bm25 import bm25_search
from backend.retrieval.config import PipelineConfig
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.retrieval.multi_query import expand_query
from backend.vectorstore import dense_search

# The query-expansion LLM, built once and reused across a run.
_LLM = None


def _get_llm(config: PipelineConfig):
    global _LLM
    if _LLM is None:
        from langchain_ollama import ChatOllama
        _LLM = ChatOllama(model=config.generation.model, temperature=0)
    return _LLM


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


def _to_chunks(rows: list[dict], score_key: str) -> list[RetrievedChunk]:
    out = []
    for row in rows:
        md = row.get("metadata") or {}
        out.append(RetrievedChunk(
            chunk_id=md.get("chunk_id", str(row.get("id", ""))),
            content=row.get("content", ""),
            metadata=md,
            score=float(row.get(score_key, 0.0)),
        ))
    return out


def _dense(query: str, config: PipelineConfig, flt: dict) -> list[RetrievedChunk]:
    return _to_chunks(dense_search(query, k=config.dense.top_k, filter=flt), "similarity")


def _bm25(query: str, config: PipelineConfig, flt: dict) -> list[RetrievedChunk]:
    return _to_chunks(bm25_search(query, k=config.bm25.top_k, filter=flt), "score")


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
        t = time.perf_counter()
        bm = _bm25(query, config, flt)
        stage_ms["bm25_ms"] = (time.perf_counter() - t) * 1000
        legs.append(bm)

    if not legs:
        raise ValueError("No retrieval leg is enabled — at least `dense` must be on.")

    # --- Stage 2: fusion ---
    if config.fusion.method == "rrf":
        t = time.perf_counter()
        # Fuse the legs' rank lists, then re-attach content/metadata for each id.
        lookup: dict[str, RetrievedChunk] = {}
        for leg in legs:
            for c in leg:
                lookup.setdefault(c.chunk_id, c)
        fused = reciprocal_rank_fusion([[c.chunk_id for c in leg] for leg in legs],
                                       k=config.fusion.rrf_k)
        ranked = [RetrievedChunk(cid, lookup[cid].content, lookup[cid].metadata, score)
                  for cid, score in fused]
        stage_ms["fusion_ms"] = (time.perf_counter() - t) * 1000
    elif config.fusion.method in ("none", None):
        if len(legs) > 1:
            raise ValueError("Multiple retrieval legs are enabled but fusion.method is 'none'.")
        ranked = legs[0]
    else:
        raise ValueError(f"Unknown fusion method: {config.fusion.method}")

    # --- Stage 3: multi-query expansion ---
    if config.multi_query.enabled:
        raise NotImplementedError("Multi-query expansion lands in Phase 5.")

    # --- Stage 4: cross-encoder rerank ---
    if config.rerank.enabled:
        raise NotImplementedError("Cross-encoder rerank lands in Phase 6.")

    return RetrieveResult(chunks=ranked, stage_ms=stage_ms)
