"""RAGAS answer-quality metrics, judged by Google Gemini.

The slow gate: "is the generated answer good?" — faithfulness (hallucination proxy),
answer_relevancy, context_precision, context_recall. Judged by Gemini, which is a
DIFFERENT and much stronger model than the generator (llama3.2). Self-grading with
the system-under-test model is explicitly disallowed (see docs/DECISIONS.md,
"who judged the judge?").

All heavy imports are lazy so the retrieval-only harness never needs ragas / Gemini.
Targets ragas 0.1.x (columns: question, answer, contexts, ground_truth). Requires
GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
"""

from __future__ import annotations

import os

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def gemini_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "RAGAS needs a Gemini key: set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env."
        )
    return key


def evaluate_samples(samples: list[dict], judge_model: str = "gemini-1.5-flash") -> dict[str, float]:
    """Run RAGAS over samples and return mean scores per metric.

    Each sample: {"question", "answer", "contexts": [str, ...], "ground_truth"}.
    Unanswerable questions (empty ground_truth) should be excluded by the caller —
    context_recall/precision are undefined without a reference answer.
    """
    key = _gemini_key()

    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    judge = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=judge_model, google_api_key=key, temperature=0)
    )
    judge_emb = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=key)
    )

    ds = Dataset.from_list([
        {
            "question": s["question"],
            "answer": s["answer"],
            "contexts": s["contexts"],
            "ground_truth": s["ground_truth"],
        }
        for s in samples
    ])

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(ds, metrics=metrics, llm=judge, embeddings=judge_emb)

    # ragas returns a result mapping metric name -> score (mean over rows).
    scores = {}
    for m in RAGAS_METRICS:
        val = result.get(m) if hasattr(result, "get") else None
        if val is not None:
            scores[m] = float(val)
    return scores
