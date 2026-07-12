"""Inspect the live Supabase `documents` table and report what is actually indexed.

The one fact that cannot be known from committed source is the real corpus: how
many chunks exist and which drivers / grands prix / sessions they cover. This
script recovers it directly from the database (no Ollama / embeddings needed —
we only read the `metadata` column).

Usage:
    python scripts/inspect_corpus.py

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY (loaded from .env if python-dotenv
is installed). Read-only: it issues SELECTs only, never writes.
"""

import os
from collections import Counter

from supabase import create_client


def _client():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ModuleNotFoundError:
        pass
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


def fetch_all_metadata(client) -> list[dict]:
    """Page through the whole table pulling id + metadata (Supabase caps rows/req)."""
    rows: list[dict] = []
    page_size = 1000
    start = 0
    while True:
        resp = (
            client.table("documents")
            .select("id, metadata")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


def _distinct(rows: list[dict], key: str) -> Counter:
    c: Counter = Counter()
    for r in rows:
        md = r.get("metadata") or {}
        val = md.get(key)
        if val is not None:
            c[val] += 1
    return c


def main() -> None:
    client = _client()
    rows = fetch_all_metadata(client)

    print(f"Total indexed documents: {len(rows)}\n")

    for key in ("year", "grand_prix", "session_type", "doc_type", "compound"):
        counts = _distinct(rows, key)
        if not counts:
            print(f"{key}: (none present)")
            continue
        rendered = ", ".join(f"{v}={n}" for v, n in sorted(counts.items(), key=lambda kv: str(kv[0])))
        print(f"{key} ({len(counts)} distinct): {rendered}")

    drivers = _distinct(rows, "driver")
    print(f"\ndrivers ({len(drivers)} distinct): {', '.join(sorted(drivers))}")

    # A stint doc has a `stint` key; a result doc has doc_type == 'result'.
    stint_docs = sum(1 for r in rows if (r.get("metadata") or {}).get("stint") is not None)
    result_docs = sum(1 for r in rows if (r.get("metadata") or {}).get("doc_type") == "result")
    print(f"\nstint documents: {stint_docs}")
    print(f"result documents: {result_docs}")

    # Whether the deterministic chunk_id (Phase 1) has been applied yet.
    with_chunk_id = sum(1 for r in rows if (r.get("metadata") or {}).get("chunk_id"))
    print(f"documents with a deterministic chunk_id: {with_chunk_id}/{len(rows)}")


if __name__ == "__main__":
    main()
