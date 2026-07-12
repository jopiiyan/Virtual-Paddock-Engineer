"""Lexical BM25 retrieval leg (rank_bm25), for the hybrid search path.

BM25 scores exact term overlap, which is exactly where dense embeddings are weak: an
exact token like a tyre compound (`SOFT`) or a driver code (`VER`) in the query gets a
strong score against the same token in a chunk, instead of being smeared across all
semantically-similar chunks. See docs/DECISIONS.md D7 for why rank_bm25 over Postgres
FTS (tiny corpus; honesty about the in-memory / non-scaling tradeoff).

The index is built over the SAME corpus subset the dense leg searches (the config's
corpus_filter), so the lexical and vector legs never see different documents.
"""

from __future__ import annotations

import json
import re

from rank_bm25 import BM25Okapi

from backend.vectorstore import fetch_documents

# Split on runs of alphanumerics, lowercased. Keeps driver codes ("VER" -> "ver"),
# compounds ("SOFT" -> "soft") and position tokens ("P4" -> "p4") as whole tokens.
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _BM25Index:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.bm25 = BM25Okapi([tokenize(r.get("content", "")) for r in rows])

    def search(self, query: str, k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out = []
        for i in order:
            md = self.rows[i].get("metadata") or {}
            out.append({
                "chunk_id": md.get("chunk_id", str(self.rows[i].get("id", ""))),
                "content": self.rows[i].get("content", ""),
                "metadata": md,
                "score": float(scores[i]),
            })
        return out


# Build the index once per (corpus subset) per run — the corpus is tiny and static.
_INDEX_CACHE: dict[str, _BM25Index] = {}


def _index_for(filter: dict | None) -> _BM25Index:
    key = json.dumps(filter or {}, sort_keys=True)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = _BM25Index(fetch_documents(filter))
    return _INDEX_CACHE[key]


def bm25_search(query: str, k: int, filter: dict | None = None) -> list[dict]:
    """Top-k BM25 rows (chunk_id, content, metadata, score) over the filtered corpus."""
    return _index_for(filter).search(query, k)
