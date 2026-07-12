"""Reciprocal Rank Fusion — implemented explicitly, no black-box helper.

RRF combines several ranked lists by summing 1/(k + rank) across the lists, where
rank is 1-based (best = 1). It fuses **ranks**, not scores, which is why it works for
a cosine-similarity retriever and a BM25 retriever whose raw scores are on
incomparable scales: the scales are discarded, only positions matter.

    score(d) = Σ_over_lists  1 / (k + rank_list(d))

`k` (default 60, from the original RRF paper) damps the influence of the very top
ranks so no single retriever's #1 dominates the fused order — a document has to do
well across lists to rank high. See docs/DECISIONS.md D7.
"""

from __future__ import annotations

from collections import defaultdict

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = DEFAULT_RRF_K
) -> list[tuple[str, float]]:
    """Fuse ranked id-lists into one, best first.

    Args:
        rankings: each inner list is one retriever's chunk_ids, best-ranked first.
        k: the RRF damping constant.
    Returns:
        (chunk_id, fused_score) pairs sorted by score desc, ties broken by chunk_id
        for determinism.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
