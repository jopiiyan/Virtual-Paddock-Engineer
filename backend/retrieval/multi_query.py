"""Multi-query expansion: rewrite one query into several, retrieve each, fuse.

Two modes (docs/DECISIONS.md D8):
  - paraphrase: N rewordings, to give the dense retriever more surface forms.
  - decompose:  split a multi-entity question into per-entity sub-queries, to stop
                one entity starving the other in a comparative query.

The generator is the same local `llama3.2` at temperature 0, so expansion is
deterministic. The original query is always kept in the set, so expansion can only add
recall, never lose the verbatim query.
"""

from __future__ import annotations

import re

_PARAPHRASE_PROMPT = (
    "Rewrite the following Formula 1 question in {n} different ways that mean the same "
    "thing, to help a search engine. Vary the wording and synonyms; keep the meaning "
    "identical. Return ONLY the {n} rewrites, one per line, no numbering.\n\nQuestion: {q}"
)

_DECOMPOSE_PROMPT = (
    "Break the following Formula 1 question into the minimal set of simpler standalone "
    "search queries needed to answer it. If it compares two drivers or stints, produce "
    "one query per driver/stint. If it is already simple, just restate it once. Return "
    "ONLY the queries, one per line, no numbering.\n\nQuestion: {q}"
)


def _parse_lines(text: str, limit: int) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)   # strip stray bullets/numbers
        if line:
            out.append(line)
    return out[:limit]


def expand_query(query: str, n: int, mode: str, llm) -> list[str]:
    """Return the original query plus up to N generated sub-queries (deduped).

    `llm` is any LangChain chat model exposing `.invoke(str) -> message`.
    On any generation failure, degrade gracefully to just the original query.
    """
    template = _DECOMPOSE_PROMPT if mode == "decompose" else _PARAPHRASE_PROMPT
    try:
        resp = llm.invoke(template.format(n=n, q=query))
        text = resp.content if hasattr(resp, "content") else str(resp)
        generated = _parse_lines(text, n)
    except Exception:
        generated = []

    seen, queries = set(), []
    for q in [query, *generated]:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            queries.append(q)
    return queries
