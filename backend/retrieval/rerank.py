"""Cross-encoder reranking — the precision second stage.

A cross-encoder scores each (query, chunk) pair jointly through one transformer, so it
judges relevance far more sharply than the bi-encoder first stage — at the cost of one
forward pass per candidate, which is why it only ever runs on a shortlist (docs/
DECISIONS.md D9). Model is loaded once and cached; CPU inference.
"""

from __future__ import annotations

# Cache loaded models by name — loading BAAI/bge-reranker-base is ~280 MB / a few seconds.
_MODELS: dict = {}


def _model(name: str):
    if name not in _MODELS:
        from sentence_transformers import CrossEncoder
        _MODELS[name] = CrossEncoder(name)
    return _MODELS[name]


def rerank(query: str, chunks: list, model_name: str) -> list:
    """Return the chunks re-sorted by cross-encoder relevance (best first).

    Reorders the whole shortlist rather than truncating — truncation to the generation
    context size happens downstream, so recall@k over the reranked order stays
    well-defined. Each chunk's score is replaced with its cross-encoder score.
    """
    if not chunks:
        return chunks
    pairs = [[query, c.content] for c in chunks]
    scores = _model(model_name).predict(pairs)
    order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))
    reranked = []
    for i in order:
        c = chunks[i]
        c.score = float(scores[i])
        reranked.append(c)
    return reranked
