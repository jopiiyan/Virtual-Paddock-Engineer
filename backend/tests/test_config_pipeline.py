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


def test_bm25_stage_refuses_until_implemented():
    # dense off + bm25 on reaches the bm25 gate without any network call.
    cfg = PipelineConfig()
    cfg.dense.enabled = False
    cfg.bm25.enabled = True
    with pytest.raises(NotImplementedError):
        retrieve("any question", cfg)


def test_no_leg_enabled_is_an_error():
    cfg = PipelineConfig()
    cfg.dense.enabled = False
    with pytest.raises((ValueError, NotImplementedError)):
        retrieve("any question", cfg)
