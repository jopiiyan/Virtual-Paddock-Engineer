"""Tests for config loading and the pipeline's sequencing guardrails.

These need no network: they check that the config round-trips, that the hash is
stable and sensitive, and that not-yet-built stages refuse to run (enforcing R4 —
no feature before the baseline is frozen).
"""

import pytest

from backend.retrieval.config import PipelineConfig, load_config
from backend.retrieval.pipeline import retrieve


def test_baseline_config_loads():
    cfg = load_config("configs/baseline.yaml")
    assert cfg.name == "baseline"
    assert cfg.dense.enabled and cfg.dense.top_k == 20
    assert not cfg.bm25.enabled
    assert cfg.fusion.method == "none"
    assert cfg.corpus_filter == {"grand_prix": "Silverstone"}


def test_config_hash_is_stable_and_sensitive():
    a = load_config("configs/baseline.yaml")
    b = load_config("configs/baseline.yaml")
    assert a.config_hash() == b.config_hash()          # deterministic
    b.dense.top_k = 5
    assert a.config_hash() != b.config_hash()          # sensitive to any change


def _stub_leg(monkeypatch):
    """Replace both retrieval legs with a fixed in-memory result, so pipeline
    control-flow can be tested without Supabase / Ollama."""
    import backend.retrieval.pipeline as P
    chunks = [P.RetrievedChunk("x", "content", {"chunk_id": "x"}, 1.0)]
    monkeypatch.setattr(P, "_dense", lambda q, c, f: list(chunks))
    monkeypatch.setattr(P, "_bm25", lambda q, c, f: list(chunks))
    return P


def test_rerank_stage_runs(monkeypatch):
    # Rerank is implemented (Phase 6): with a fake cross-encoder it should be invoked
    # on the shortlist and record its stage timing — no model download needed.
    P = _stub_leg(monkeypatch)
    called = {}

    def fake_rerank(q, chunks, model):
        called["model"] = model
        return list(reversed(chunks))

    monkeypatch.setattr(P, "rerank", fake_rerank)
    cfg = PipelineConfig()
    cfg.rerank.enabled = True
    res = P.retrieve("q", cfg)
    assert called["model"] == cfg.rerank.model
    assert "rerank_ms" in res.stage_ms


def test_multi_query_runs_and_fuses(monkeypatch):
    # Multi-query is implemented (Phase 5): with a fake expander and stubbed legs it
    # should retrieve per sub-query and fuse, with no network and no error.
    P = _stub_leg(monkeypatch)
    monkeypatch.setattr(P, "_get_llm", lambda cfg: object())
    monkeypatch.setattr(P, "expand_query", lambda q, n, mode, llm: [q, q + " variant"])
    cfg = PipelineConfig()
    cfg.multi_query.enabled = True
    res = P.retrieve("compare HAM and VER", cfg)
    assert res.chunk_ids == ["x"]
    assert "expand_ms" in res.stage_ms


def test_two_legs_without_fusion_is_an_error(monkeypatch):
    P = _stub_leg(monkeypatch)
    cfg = PipelineConfig()
    cfg.bm25.enabled = True          # dense + bm25 both on, fusion still 'none'
    with pytest.raises(ValueError):
        P.retrieve("q", cfg)


def test_no_leg_enabled_is_an_error():
    cfg = PipelineConfig()
    cfg.dense.enabled = False
    with pytest.raises((ValueError, NotImplementedError)):
        retrieve("any question", cfg)
