"""Typed pipeline configuration loaded from YAML.

The config is the single source of truth for which retrieval stages are active.
`config_hash()` gives a stable fingerprint of the resolved config so every logged
eval run records exactly what produced it (CLAUDE.md Phase 2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class DenseConfig:
    enabled: bool = True
    top_k: int = 5


@dataclass
class BM25Config:
    enabled: bool = False
    top_k: int = 20


@dataclass
class FusionConfig:
    method: str = "none"   # none | rrf
    rrf_k: int = 60


@dataclass
class MultiQueryConfig:
    enabled: bool = False
    n_queries: int = 3
    mode: str = "decompose"   # paraphrase | decompose


@dataclass
class RerankConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"
    candidates: int = 20
    top_n: int = 5


@dataclass
class GenerationConfig:
    enabled: bool = False   # off for the retrieval-only baseline gate
    model: str = "llama3.2"
    temperature: float = 0.0
    context_top_n: int = 5   # how many top-ranked chunks the LLM actually sees


@dataclass
class EvalConfig:
    golden_set: str = "eval/golden_set.jsonl"
    judge_model: str = "gemini"
    ragas: bool = False
    k_values: list[int] = field(default_factory=lambda: [3, 5, 10])


@dataclass
class PipelineConfig:
    name: str = "unnamed"
    # Fixed corpus scope applied to EVERY retrieval leg (dense + bm25), so the
    # untested races can never leak into results. See docs/AUDIT.md §5.
    corpus_filter: dict = field(default_factory=lambda: {"grand_prix": "Silverstone"})
    dense: DenseConfig = field(default_factory=DenseConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    multi_query: MultiQueryConfig = field(default_factory=MultiQueryConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def config_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def as_dict(self) -> dict:
        return asdict(self)


def _section(raw: dict, key: str, cls):
    return cls(**(raw.get(key) or {}))


def load_config(path: str | Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    retr = raw.get("retrieval") or {}
    return PipelineConfig(
        name=raw.get("name", Path(path).stem),
        corpus_filter=retr.get("corpus_filter", {"grand_prix": "Silverstone"}),
        dense=_section(retr, "dense", DenseConfig),
        bm25=_section(retr, "bm25", BM25Config),
        fusion=_section(retr, "fusion", FusionConfig),
        multi_query=_section(retr, "multi_query", MultiQueryConfig),
        rerank=_section(retr, "rerank", RerankConfig),
        generation=_section(raw, "generation", GenerationConfig),
        eval=_section(raw, "eval", EvalConfig),
    )
