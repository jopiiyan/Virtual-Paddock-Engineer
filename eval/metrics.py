"""Classical IR retrieval metrics — pure functions, no LLM, deterministic.

These are the fast, cheap gate: they answer "did we fetch the right chunks?"
against the golden set's `relevant_chunk_ids`, with no judge model involved. A bug
here invalidates every number in the ablation table, so each function is unit-tested
against a hand-worked example in backend/tests/test_metrics.py.

Convention: `ranked` is the retrieved chunk_ids best-first; `relevant` is the set of
ground-truth chunk_ids. Questions with no relevant chunks (unanswerable) are NOT
scored here — abstention is a generation property, handled separately.
"""

from __future__ import annotations

import math
from statistics import mean


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant chunks that appear in the top-k."""
    if not relevant:
        raise ValueError("recall_at_k is undefined when there are no relevant chunks")
    topk = set(ranked[:k])
    return len(topk & relevant) / len(relevant)


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant chunk is in the top-k, else 0.0."""
    if not relevant:
        raise ValueError("hit_rate_at_k is undefined when there are no relevant chunks")
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / (rank of the first relevant chunk); 0 if none retrieved."""
    if not relevant:
        raise ValueError("reciprocal_rank is undefined when there are no relevant chunks")
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k. DCG uses gain 1 for a relevant hit, discounted by
    log2(rank+1); IDCG is the best achievable ordering (all relevant up front)."""
    if not relevant:
        raise ValueError("ndcg_at_k is undefined when there are no relevant chunks")
    dcg = 0.0
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def aggregate(
    per_question: list[tuple[list[str], set[str]]],
    k_values: list[int] = (3, 5, 10),
) -> dict[str, float]:
    """Mean each metric over the answerable questions.

    `per_question` is a list of (ranked_chunk_ids, relevant_chunk_id_set). Items
    with an empty relevant set are skipped (unanswerable questions aren't scored on
    retrieval). Returns a flat dict, e.g. {"recall@5": .., "mrr": .., "ndcg@10": ..}.
    """
    answerable = [(r, rel) for r, rel in per_question if rel]
    if not answerable:
        return {}
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = mean(recall_at_k(r, rel, k) for r, rel in answerable)
        out[f"hit_rate@{k}"] = mean(hit_rate_at_k(r, rel, k) for r, rel in answerable)
    out["mrr"] = mean(reciprocal_rank(r, rel) for r, rel in answerable)
    out["ndcg@10"] = mean(ndcg_at_k(r, rel, 10) for r, rel in answerable)
    out["n_answerable"] = len(answerable)
    return out
