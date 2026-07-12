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
from backend.retrieval.rerank import rerank
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


def _rrf_merge(legs: list[list[RetrievedChunk]], k: int) -> list[RetrievedChunk]:
    """Fuse several ranked chunk-lists into one by RRF, re-attaching content/metadata."""
    lookup: dict[str, RetrievedChunk] = {}
    for leg in legs:
        for c in leg:
            lookup.setdefault(c.chunk_id, c)
    fused = reciprocal_rank_fusion([[c.chunk_id for c in leg] for leg in legs], k=k)
    return [RetrievedChunk(cid, lookup[cid].content, lookup[cid].metadata, score)
            for cid, score in fused]


def _candidate_stage(query: str, config: PipelineConfig, flt: dict,
                     stage_ms: dict[str, float]) -> list[RetrievedChunk]:
    """One query -> dense and/or BM25 legs -> (optional) RRF fusion. Timings accumulate
    into stage_ms so multi-query (which calls this per sub-query) reports summed cost."""
    legs: list[list[RetrievedChunk]] = []

    if config.dense.enabled:
        t = time.perf_counter()
        legs.append(_dense(query, config, flt))
        stage_ms["dense_ms"] = stage_ms.get("dense_ms", 0.0) + (time.perf_counter() - t) * 1000

    if config.bm25.enabled:
        t = time.perf_counter()
        legs.append(_bm25(query, config, flt))
        stage_ms["bm25_ms"] = stage_ms.get("bm25_ms", 0.0) + (time.perf_counter() - t) * 1000

    if not legs:
        raise ValueError("No retrieval leg is enabled — at least `dense` must be on.")

    if config.fusion.method == "rrf":
        t = time.perf_counter()
        ranked = _rrf_merge(legs, config.fusion.rrf_k)
        stage_ms["fusion_ms"] = stage_ms.get("fusion_ms", 0.0) + (time.perf_counter() - t) * 1000
        return ranked
    if config.fusion.method in ("none", None):
        if len(legs) > 1:
            raise ValueError("Multiple retrieval legs are enabled but fusion.method is 'none'.")
        return legs[0]
    raise ValueError(f"Unknown fusion method: {config.fusion.method}")


def retrieve(query: str, config: PipelineConfig, extra_filter: dict | None = None) -> RetrieveResult:
    stage_ms: dict[str, float] = {}
    flt = _merge_filter(config, extra_filter)

    # --- Stages 1-2: candidate generation + fusion, optionally over expanded queries ---
    if config.multi_query.enabled:
        t = time.perf_counter()
        queries = expand_query(query, config.multi_query.n_queries,
                               config.multi_query.mode, _get_llm(config))
        stage_ms["expand_ms"] = (time.perf_counter() - t) * 1000
        per_query = [_candidate_stage(q, config, flt, stage_ms) for q in queries]
        t = time.perf_counter()
        ranked = _rrf_merge(per_query, config.fusion.rrf_k)
        stage_ms["mq_fusion_ms"] = (time.perf_counter() - t) * 1000
    else:
        ranked = _candidate_stage(query, config, flt, stage_ms)

    # --- Stage 3: cross-encoder rerank (precision second stage) ---
    if config.rerank.enabled:
        t = time.perf_counter()
        shortlist = ranked[: config.rerank.candidates]
        ranked = rerank(query, shortlist, config.rerank.model)
        stage_ms["rerank_ms"] = (time.perf_counter() - t) * 1000

    return RetrieveResult(chunks=ranked, stage_ms=stage_ms)
