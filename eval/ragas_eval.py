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


def _gemini_is_finished(response) -> bool:
    """RAGAS 0.2.x hard-codes `finish_reason == "stop"` (lowercase), but Gemini
    returns "STOP" (uppercase), so its default parser flags every completed call as
    unfinished and raises LLMDidNotFinishException. Treat the Gemini finish reasons
    as terminal (case-insensitively); missing reason == finished.
    """
    ok = {"STOP", "MAX_TOKENS", "FINISH_REASON_STOP"}
    for gen_list in response.generations:
        for gen in gen_list:
            reason = (getattr(gen, "generation_info", None) or {}).get("finish_reason")
            if reason is None:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    reason = (getattr(msg, "response_metadata", None) or {}).get("finish_reason")
            if reason is not None and str(reason).upper() not in ok:
                return False
    return True


def gemini_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "RAGAS needs a Gemini key: set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env."
        )
    return key


def evaluate_samples(
    samples: list[dict],
    judge_model: str = "gemini-2.5-flash",
    max_workers: int = 1,
) -> dict[str, float]:
    """Run RAGAS over samples and return mean scores per metric.

    Each sample: {"question", "answer", "contexts": [str, ...], "ground_truth"}.
    Unanswerable questions (empty ground_truth) should be excluded by the caller —
    context_recall/precision are undefined without a reference answer.

    The judge runs on Gemini's free tier (single-digit requests/minute), so we throttle
    hard: one worker, a long per-call timeout, and many retries with backoff. This
    trades wall-clock time for completion rather than dropping rows to TimeoutError.
    """
    key = _gemini_key()

    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    judge = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=judge_model,
            google_api_key=key,
            temperature=0,
            # gemini-2.5-flash is a thinking model: without these it spends the whole
            # output budget on internal reasoning and the answer never finishes, so
            # ragas raises LLMDidNotFinishException on every row. Disable thinking and
            # give the (short, structured) judge verdict room to complete.
            thinking_budget=0,
            max_output_tokens=2048,
        ),
        # Override ragas' finish-reason check, which only accepts lowercase "stop"
        # and so rejects Gemini's "STOP" on every call.
        is_finished_parser=_gemini_is_finished,
    )
    judge_emb = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=key)
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
    # Respect the free-tier rate limit: serialize calls, tolerate long backoffs.
    run_config = RunConfig(timeout=600, max_workers=max_workers, max_retries=15, max_wait=90)
    result = evaluate(ds, metrics=metrics, llm=judge, embeddings=judge_emb, run_config=run_config)

    # Most reliable extraction across ragas 0.2.x: mean each metric column from the
    # per-row dataframe (NaNs, e.g. a judge parse failure on one row, are ignored).
    df = result.to_pandas()
    scores: dict[str, float] = {}
    for m in RAGAS_METRICS:
        if m in df.columns:
            val = df[m].mean(skipna=True)
            if val == val:   # not NaN
                scores[m] = float(val)
    return scores
