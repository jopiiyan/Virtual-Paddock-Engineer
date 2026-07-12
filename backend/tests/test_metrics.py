"""Hand-worked tests for the retrieval metrics.

A bug in this module would silently corrupt every number in the ablation table, so
each metric is checked against a manually computed expected value — not against
itself. Worked example used throughout:

    ranked   = ["a", "b", "c", "d"]      relevant = {"b", "d"}
    relevant appears at ranks 2 and 4.
"""

import math

import pytest

from eval import metrics as M

RANKED = ["a", "b", "c", "d"]
RELEVANT = {"b", "d"}


def test_recall_at_k():
    assert M.recall_at_k(RANKED, RELEVANT, 1) == 0.0        # {a} ∩ {b,d} = ∅
    assert M.recall_at_k(RANKED, RELEVANT, 2) == 0.5        # {a,b} → 1 of 2
    assert M.recall_at_k(RANKED, RELEVANT, 3) == 0.5        # {a,b,c} → 1 of 2
    assert M.recall_at_k(RANKED, RELEVANT, 4) == 1.0        # all → 2 of 2


def test_hit_rate_at_k():
    assert M.hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0      # no relevant in top-1
    assert M.hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0      # 'b' at rank 2


def test_reciprocal_rank():
    # first relevant ('b') is at rank 2 -> 1/2
    assert M.reciprocal_rank(RANKED, RELEVANT) == 0.5
    # none retrieved -> 0
    assert M.reciprocal_rank(["x", "y"], RELEVANT) == 0.0


def test_ndcg_at_k():
    # DCG@4 = 1/log2(3) + 1/log2(5); IDCG@4 (2 relevant) = 1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert M.ndcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(dcg / idcg)
    # @2 only sees [a, b]; 'b' at rank 2 contributes 1/log2(3)
    assert M.ndcg_at_k(RANKED, RELEVANT, 2) == pytest.approx((1 / math.log2(3)) / idcg)


def test_perfect_ranking_scores_one():
    ranked = ["b", "d", "a", "c"]
    assert M.recall_at_k(ranked, RELEVANT, 2) == 1.0
    assert M.reciprocal_rank(ranked, RELEVANT) == 1.0
    assert M.ndcg_at_k(ranked, RELEVANT, 4) == pytest.approx(1.0)


def test_undefined_without_relevant():
    for fn in (M.recall_at_k, M.hit_rate_at_k, M.ndcg_at_k):
        with pytest.raises(ValueError):
            fn(RANKED, set(), 5)
    with pytest.raises(ValueError):
        M.reciprocal_rank(RANKED, set())


def test_aggregate_skips_unanswerable():
    # Second item has an empty relevant set (unanswerable) and must be skipped.
    per_q = [(["b", "a"], {"b"}), (["x", "y"], set())]
    agg = M.aggregate(per_q, k_values=[1, 2])
    assert agg["n_answerable"] == 1
    assert agg["recall@1"] == 1.0     # only the first question counts
    assert agg["mrr"] == 1.0
