"""Draft candidate golden-set questions from the indexed chunks with an LLM.

This is a DRAFTING aid only. It emits `eval/candidates.jsonl` for a human to
review — it does NOT write the golden set. Per CLAUDE.md Phase 1, every candidate
question and answer must be verified by a human against the FastF1 source before it
is promoted into eval/golden_set.jsonl. An LLM writing both the questions and the
answers with no ground truth in the loop is exactly the closed loop the eval is
meant to avoid.

The committed eval/golden_set.jsonl was authored and verified by hand against the
Silverstone corpus (itself derived deterministically from FastF1 lap/result data);
this script exists so the drafting step is reproducible and honestly documented.

Usage:
    python scripts/generate_candidates.py --per-bucket 4
Requires Ollama running (llama3.2) and Supabase creds.
"""

import argparse
import json
import os
import random
from pathlib import Path

CORPUS_GP = "Silverstone"
OUT = Path(__file__).resolve().parent.parent / "eval" / "candidates.jsonl"

BUCKET_GUIDANCE = {
    "factual": "a simple single-fact lookup answerable from ONE chunk",
    "comparative": "a comparison between TWO different drivers (needs two chunks)",
    "exact_term": "a query hinging on an exact token: a driver code (e.g. VER), a "
                  "tyre compound (SOFT/MEDIUM/HARD), or a trap code (ST/FL/I1/I2)",
    "paraphrase": "a vague, natural-language question with little lexical overlap "
                  "with the chunk text (no driver codes, no exact numbers)",
    "unanswerable": "a plausible question the corpus CANNOT answer (weather, "
                    "qualifying, another year, pit-stop laps, car setup)",
}


def _corpus(gp: str) -> list[dict]:
    try:
        from dotenv import load_dotenv
        load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))
    except ModuleNotFoundError:
        pass
    from supabase import create_client
    c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    rows, start = [], 0
    while True:
        b = c.table("documents").select("content, metadata").eq("metadata->>grand_prix", gp).range(start, start + 999).execute().data
        rows += b
        if len(b) < 1000:
            return rows
        start += 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.2", temperature=0)

    corpus = _corpus(CORPUS_GP)
    catalogue = "\n".join(f"- {r['metadata']['chunk_id']}: {r['content']}" for r in corpus)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for bucket, guidance in BUCKET_GUIDANCE.items():
            prompt = (
                f"You are drafting evaluation questions for an F1 retrieval system.\n"
                f"Here is the ENTIRE corpus (chunk_id: text):\n{catalogue}\n\n"
                f"Draft {args.per_bucket} '{bucket}' questions: each is {guidance}.\n"
                f"Return ONLY a JSON array; each item: "
                f'{{"question": str, "draft_answer": str, "relevant_chunk_ids": [str]}}. '
                f"For unanswerable questions use an empty relevant_chunk_ids list."
            )
            raw = llm.invoke(prompt).content
            for item in _safe_parse(raw):
                item["query_type"] = bucket
                item["_status"] = "UNVERIFIED — verify against FastF1 before promoting"
                f.write(json.dumps(item) + "\n")
            print(f"drafted {bucket}")

    print(f"\nWrote candidates to {OUT}. THESE ARE UNVERIFIED. Human-verify each "
          f"against FastF1 source, then hand-copy survivors into eval/golden_set.jsonl.")


def _safe_parse(raw: str) -> list[dict]:
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []


if __name__ == "__main__":
    main()
