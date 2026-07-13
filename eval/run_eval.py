"""Run the evaluation harness for one pipeline config and emit a results row.

Run as a module from the project root so `backend` and `eval` both import:

    python -m eval.run_eval --config configs/baseline.yaml
    python -m eval.run_eval --config configs/baseline.yaml --out eval/results/baseline.json
    python -m eval.run_eval --config configs/baseline.yaml --no-ragas   # judge quota unavailable

Takes a pipeline config, runs every golden question through backend.retrieval.retrieve,
and computes:
  (a) retrieval IR metrics (recall@k, hit_rate@k, MRR, nDCG@10) — always;
  (b) per-question-type breakdown of recall@5 — always;
  (c) latency p50/p95 (end-to-end and per stage) — always;
  (d) RAGAS answer-quality metrics via Gemini + an abstention rate on the
      unanswerable questions — only when generation + ragas are enabled and a
      Gemini key is present.

Every run is appended to eval/results/runs.jsonl with the config hash + timestamp
(never overwritten). `--out` additionally writes a full standalone summary (used to
freeze the baseline in Phase 3).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from backend.retrieval.config import PipelineConfig, load_config
from backend.retrieval.pipeline import retrieve
from eval import metrics as M

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval" / "results" / "runs.jsonl"

REFUSAL_CUES = (
    "isn't in the data", "not in the data", "isn't in the", "don't have",
    "do not have", "cannot answer", "can't answer", "out of scope",
    "no weather", "no drs", "no setup", "not available", "not stored",
    "only the race", "only the 2025",
)


def load_golden(path: str) -> list[dict]:
    rows = []
    for line in (ROOT / path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((pct / 100) * (len(s) - 1))))
    return s[idx]


def build_generator(config: PipelineConfig):
    """Reuse the app's grounding prompt + Ollama generator for answer quality."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    from backend.chain import GROUNDING_INSTRUCTION

    prompt = ChatPromptTemplate.from_template(
        GROUNDING_INSTRUCTION + "\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    llm = ChatOllama(model=config.generation.model, temperature=config.generation.temperature)
    return prompt | llm | StrOutputParser()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="also write a full standalone summary here")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N questions")
    ap.add_argument("--no-ragas", action="store_true",
                    help="skip RAGAS even if the config enables it (e.g. judge quota unavailable)")
    ap.add_argument("--no-generation", action="store_true",
                    help="retrieval metrics only; skip LLM generation, abstention and RAGAS")
    ap.add_argument("--ragas-workers", type=int, default=1,
                    help="parallel RAGAS judge calls; keep 1 on the free tier, raise (e.g. 12) on a paid key")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.no_generation:
        config.generation.enabled = False
    if args.no_ragas:
        config.eval.ragas = False
    golden = load_golden(config.eval.golden_set)
    if args.limit:
        golden = golden[: args.limit]

    gen_chain = build_generator(config) if config.generation.enabled else None

    per_question_metric_inputs: list[tuple[list[str], set[str]]] = []
    by_type: dict[str, list[tuple[list[str], set[str]]]] = defaultdict(list)
    total_latencies: list[float] = []
    stage_latencies: dict[str, list[float]] = defaultdict(list)
    details: list[dict] = []
    ragas_samples: list[dict] = []
    abstain_hits, abstain_total = 0, 0

    for q in golden:
        relevant = set(q["relevant_chunk_ids"])
        t0 = time.perf_counter()
        result = retrieve(q["question"], config)
        wall_ms = (time.perf_counter() - t0) * 1000
        ranked = result.chunk_ids

        total_latencies.append(wall_ms)
        for stage, ms in result.stage_ms.items():
            stage_latencies[stage].append(ms)

        per_question_metric_inputs.append((ranked, relevant))
        by_type[q["query_type"]].append((ranked, relevant))

        answer = None
        if gen_chain is not None:
            ctx_chunks = result.chunks[: config.generation.context_top_n]
            context = "\n\n".join(c.content for c in ctx_chunks)
            answer = gen_chain.invoke({"context": context, "question": q["question"]})
            if q["query_type"] == "unanswerable":
                abstain_total += 1
                if any(cue in answer.lower() for cue in REFUSAL_CUES):
                    abstain_hits += 1
            elif relevant:
                ragas_samples.append({
                    "question": q["question"],
                    "answer": answer,
                    "contexts": [c.content for c in ctx_chunks],
                    "ground_truth": q["ground_truth"],
                })

        details.append({
            "id": q["id"], "query_type": q["query_type"],
            "retrieved": ranked, "relevant": sorted(relevant),
            "latency_ms": round(wall_ms, 1),
            **({"answer": answer} if answer is not None else {}),
        })

    retrieval = M.aggregate(per_question_metric_inputs, config.eval.k_values)
    per_type_recall5 = {
        t: round(mean(M.recall_at_k(r, rel, 5) for r, rel in items if rel), 4)
        for t, items in by_type.items() if any(rel for _, rel in items)
    }

    latency = {
        "p50_ms": round(percentile(total_latencies, 50), 1),
        "p95_ms": round(percentile(total_latencies, 95), 1),
        "mean_ms": round(mean(total_latencies), 1) if total_latencies else 0.0,
        "per_stage_p50_ms": {s: round(percentile(v, 50), 1) for s, v in stage_latencies.items()},
    }

    row = {
        "name": config.name,
        "config_hash": config.config_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(golden),
        "retrieval_metrics": {k: round(v, 4) for k, v in retrieval.items()},
        "per_type_recall@5": per_type_recall5,
        "latency": latency,
        "config": config.as_dict(),
    }

    if gen_chain is not None and abstain_total:
        row["abstention_rate"] = round(abstain_hits / abstain_total, 4)

    if config.eval.ragas and gen_chain is not None and ragas_samples:
        from eval.ragas_eval import evaluate_samples, gemini_available
        if gemini_available():
            row["ragas_metrics"] = {k: round(v, 4) for k, v in evaluate_samples(ragas_samples, max_workers=args.ragas_workers).items()}
        else:
            row["ragas_metrics"] = {"skipped": "no GEMINI_API_KEY"}

    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with RUNS.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"\n=== {config.name}  (hash {config.config_hash()}) ===")
    print(json.dumps(row["retrieval_metrics"], indent=2))
    print("per-type recall@5:", json.dumps(per_type_recall5))
    print("latency:", json.dumps(latency))
    if "abstention_rate" in row:
        print("abstention_rate (unanswerable):", row["abstention_rate"])
    if "ragas_metrics" in row:
        print("ragas:", json.dumps(row["ragas_metrics"]))
    print(f"\nappended -> {RUNS.relative_to(ROOT)}")

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**row, "details": details}, indent=2))
        print(f"summary -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
