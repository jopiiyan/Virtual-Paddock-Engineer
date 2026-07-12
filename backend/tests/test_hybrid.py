"""Tests for the hybrid-search primitives: RRF fusion and BM25 tokenization.

RRF is checked against a hand-computed toy example (the interviewer may ask what
k=60 does; the arithmetic must be ownable). Tokenization is checked so the exact
tokens BM25 relies on (driver codes, compounds) survive.
"""

import pytest

from backend.retrieval.bm25 import tokenize
from backend.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_hand_computed():
    # list A: a,b,c   list B: b,c,d   (k=60, ranks are 1-based)
    #   a = 1/61
    #   b = 1/62 + 1/61   (rank 2 in A, rank 1 in B)  -> highest
    #   c = 1/63 + 1/62
    #   d = 1/63
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    ids = [cid for cid, _ in fused]
    assert ids == ["b", "c", "a", "d"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1][1] == pytest.approx(1 / 63 + 1 / 62)


def test_rrf_k_damps_top_rank_dominance():
    # P is rank 1 in list1 but rank 10 in list2 (a big spike in one place).
    # Q is a steady rank 3 in both lists.
    list1 = ["P", "x1", "Q"]                                    # P@1, Q@3
    list2 = ["y1", "y2", "Q", "z1", "z2", "z3", "z4", "z5", "z6", "P"]  # Q@3, P@10
    small = dict(reciprocal_rank_fusion([list1, list2], k=1))
    large = dict(reciprocal_rank_fusion([list1, list2], k=60))
    assert small["P"] > small["Q"]        # small k: P's rank-1 spike dominates
    assert large["Q"] > large["P"]        # large k: Q's consistency wins (the flip)


def test_rrf_single_list_preserves_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert [cid for cid, _ in fused] == ["a", "b", "c"]


def test_rrf_ties_broken_deterministically():
    # identical lists -> equal scores -> tie-break by id, stable across runs.
    a = reciprocal_rank_fusion([["c", "a", "b"], ["c", "a", "b"]], k=60)
    b = reciprocal_rank_fusion([["c", "a", "b"], ["c", "a", "b"]], k=60)
    assert a == b


def test_tokenize_keeps_codes_and_compounds():
    toks = tokenize("HAM ran SOFT tyres, finished P4! best 1:29.337")
    assert toks == ["ham", "ran", "soft", "tyres", "finished", "p4", "best", "1", "29", "337"]
