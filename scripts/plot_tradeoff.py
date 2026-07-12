"""Plot the accuracy/latency tradeoff across all evaluated configs.

Reads every eval/results/<name>.json and scatters recall@5 (quality) against p50
retrieval latency (cost), so the "every quality gain has a price" story is visual.
Saves docs/tradeoff.png. Reproducible: regenerate after any new run.

    python scripts/plot_tradeoff.py
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["baseline", "bm25_only", "hybrid", "hybrid_rerank_c10", "hybrid_rerank",
         "hybrid_rerank_c50", "hybrid_mq", "hybrid_mq_paraphrase", "full"]


def main() -> None:
    rows = {}
    for f in glob.glob(str(ROOT / "eval/results/*.json")):
        d = json.load(open(f))
        rows[d["name"]] = d

    names = [n for n in ORDER if n in rows]
    xs = [max(rows[n]["latency"]["p50_ms"], 0.5) for n in names]   # clamp for log axis
    ys = [rows[n]["retrieval_metrics"]["recall@5"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(xs, ys, s=90, zorder=3, color="#2563eb", edgecolor="white", linewidth=1)
    for n, x, y in zip(names, xs, ys):
        ax.annotate(n, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8)

    # Baseline reference line — everything to the right that sits below it is a bad trade.
    base = rows["baseline"]["retrieval_metrics"]["recall@5"]
    ax.axhline(base, ls="--", lw=1, color="#9ca3af", zorder=1)
    ax.text(ax.get_xlim()[1] if False else max(xs), base, "  baseline recall@5",
            va="bottom", ha="right", fontsize=8, color="#6b7280")

    ax.set_xscale("log")
    ax.set_xlabel("p50 retrieval latency (ms, log scale) — cost")
    ax.set_ylabel("recall@5 — quality")
    ax.set_title("Accuracy vs latency across retrieval configs (Silverstone golden set)")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    out = ROOT / "docs" / "tradeoff.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
