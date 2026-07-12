"""Validate eval/golden_set.jsonl against the live corpus.

Guards the project's most valuable asset. Checks:
  1. Every line is valid JSON with the required fields.
  2. ids are unique; query_type is one of the five buckets.
  3. Every `relevant_chunk_id` actually exists in the Silverstone corpus
     (so labels can never dangle after a re-ingest).
  4. Unanswerable questions have an empty `relevant_chunk_ids`; answerable ones
     have at least one.
  5. Reports the per-bucket distribution.

Usage:
    python eval/validate_golden.py

Requires SUPABASE_URL / SUPABASE_SERVICE_KEY (reads the Silverstone chunk_ids).
Exit code is non-zero if any check fails, so CI can gate on it.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "golden_set.jsonl"
CORPUS_GP = "Silverstone"
REQUIRED = {"id", "question", "ground_truth", "relevant_chunk_ids", "query_type", "difficulty", "notes"}
BUCKETS = {"factual", "comparative", "exact_term", "paraphrase", "unanswerable"}


def load_corpus_chunk_ids() -> set[str]:
    try:
        from dotenv import load_dotenv
        load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))
    except ModuleNotFoundError:
        pass
    from supabase import create_client
    c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    ids, start = set(), 0
    while True:
        batch = (
            c.table("documents").select("metadata")
            .eq("metadata->>grand_prix", CORPUS_GP)
            .range(start, start + 999).execute().data
        )
        for r in batch:
            cid = (r.get("metadata") or {}).get("chunk_id")
            if cid:
                ids.add(cid)
        if len(batch) < 1000:
            return ids
        start += 1000


def main() -> int:
    errors: list[str] = []
    rows = []
    for i, line in enumerate(GOLDEN.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {i}: invalid JSON ({e})")
            continue
        rows.append(obj)
        missing = REQUIRED - obj.keys()
        if missing:
            errors.append(f"{obj.get('id', f'line {i}')}: missing fields {missing}")
        if obj.get("query_type") not in BUCKETS:
            errors.append(f"{obj.get('id')}: bad query_type {obj.get('query_type')!r}")

    ids = [r["id"] for r in rows]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    if dupes:
        errors.append(f"duplicate ids: {dupes}")

    corpus_ids = load_corpus_chunk_ids()
    for r in rows:
        rels = r.get("relevant_chunk_ids", [])
        qt = r.get("query_type")
        if qt == "unanswerable":
            if rels:
                errors.append(f"{r['id']}: unanswerable but has relevant_chunk_ids {rels}")
        else:
            if not rels:
                errors.append(f"{r['id']}: answerable but no relevant_chunk_ids")
            for cid in rels:
                if cid not in corpus_ids:
                    errors.append(f"{r['id']}: relevant_chunk_id not in corpus: {cid}")

    dist = Counter(r["query_type"] for r in rows)
    print(f"Golden set: {len(rows)} questions | corpus chunks: {len(corpus_ids)}")
    for b in sorted(BUCKETS):
        print(f"  {b:12s}: {dist.get(b, 0)}")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
