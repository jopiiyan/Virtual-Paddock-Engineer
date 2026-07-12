"""Backfill a deterministic `chunk_id` onto every existing `documents` row.

Why not re-ingest? Re-ingesting would re-embed via Ollama and risk duplicating the
races already indexed. The chunk_id is derived purely from existing metadata
(driver / year / grand_prix / session / stint | result), so we can compute it and
UPDATE the jsonb metadata in place — no embeddings touched, vectors unchanged.

Idempotent: safe to run repeatedly. Uses the SAME id helpers as ingestion so the
format can never drift.

Usage:
    python scripts/backfill_chunk_ids.py            # apply to all rows
    python scripts/backfill_chunk_ids.py --dry-run  # show what would change
"""

import argparse
import os

from supabase import create_client

from backend.ingestion import result_chunk_id, stint_chunk_id


def _client():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ModuleNotFoundError:
        pass
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


def compute_chunk_id(md: dict) -> str | None:
    """Deterministic id for a row's metadata, or None if it can't be classified."""
    driver = md.get("driver")
    year = md.get("year")
    gp = md.get("grand_prix")
    session = md.get("session_type")
    if not all([driver, year, gp, session]):
        return None
    if md.get("doc_type") == "result":
        return result_chunk_id(year, gp, session, driver)
    if md.get("stint") is not None:
        return stint_chunk_id(year, gp, session, driver, int(md["stint"]))
    return None


def fetch_all(client) -> list[dict]:
    rows, start, page = [], 0, 1000
    while True:
        resp = client.table("documents").select("id, metadata").range(start, start + page - 1).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page:
            return rows
        start += page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = _client()
    rows = fetch_all(client)

    updated, skipped, unclassified = 0, 0, 0
    for r in rows:
        md = r.get("metadata") or {}
        cid = compute_chunk_id(md)
        if cid is None:
            unclassified += 1
            print(f"  ! id={r['id']} could not be classified: {md}")
            continue
        if md.get("chunk_id") == cid:
            skipped += 1
            continue
        md["chunk_id"] = cid
        if not args.dry_run:
            client.table("documents").update({"metadata": md}).eq("id", r["id"]).execute()
        updated += 1

    verb = "would update" if args.dry_run else "updated"
    print(f"\n{len(rows)} rows | {verb} {updated} | already-current {skipped} | unclassified {unclassified}")


if __name__ == "__main__":
    main()
