# Design Decisions

Every decision here is recorded **with the alternative it beat and the reason**, so
the engineering logic is owned and defensible in interview (CLAUDE.md R3). New
entries are added *before* the corresponding code.

---

## D1 — Two metric families, not one

**Decision:** evaluate retrieval with classical IR metrics *and* answers with RAGAS.

- **Retrieval IR metrics** (`eval/metrics.py`) — `recall@k`, `hit_rate@k`, `MRR`,
  `nDCG@10`, computed against the golden set's `relevant_chunk_ids`. No LLM, fully
  deterministic, milliseconds to compute. This is the **fast gate**: it answers
  "did we fetch the right chunks?" independently of the generator's quality.
- **RAGAS metrics** — LLM-judged answer quality. The **slow gate**: "given the
  chunks, is the answer good?"

**Why both:** they fail independently. A retrieval change can lift recall while the
weak generator still writes a poor answer, or the generator can paper over a
mediocre retrieval. Reporting only one hides half the story. The retrieval metrics
also serve as the project's real regression test suite (see D6).

**Metric definitions (so the numbers are ownable):**
- `recall@k` = fraction of the relevant chunks that appear in the top *k*. "Of what
  I *should* have found, how much did I?"
- `hit_rate@k` = 1 if *any* relevant chunk is in the top *k*. "Did I find *anything*
  useful up top?"
- `MRR` = mean of `1/rank` of the *first* relevant chunk. Rewards putting a relevant
  chunk high, not just somewhere.
- `nDCG@10` = discounted cumulative gain (binary relevance) normalised by the ideal
  ordering. The only listed metric that rewards *ranking* the relevant chunks in the
  right order, not just their presence — which is exactly what reranking should move.

Unanswerable questions carry no relevant chunks, so they are **excluded** from
retrieval metrics (recall is undefined) and scored instead by an **abstention rate**
on generation (did the system refuse rather than hallucinate?).

---

## D2 — Judge model: Google Gemini (not llama3.2, not self-grading)

**Decision:** RAGAS is judged by **Gemini** (`gemini-1.5-flash`), configured in
`configs/*.yaml` and wired in `eval/ragas_eval.py`.

**Alternatives rejected:**
- **`llama3.2` (the generator) as judge** — self-grading bias, and a 3B model is too
  weak/unstable to produce reproducible judgements. This is a closed loop with no
  ground truth in it. Explicitly disallowed.
- **A larger local Ollama model (14B+)** — viable and fully local, but slower and
  hardware-bound; Gemini's free tier is a stronger judge at no local cost and is
  already on the project's skills list.

**Why it must differ from the generator:** if the same model writes and grades the
answer, high faithfulness can mean "the judge agrees with itself," not "the answer is
grounded." Using a stronger, independent judge makes the score an external check.

**"Who judged the judge?" (the bias we accept and disclose):** LLM-as-judge is not
ground truth. Gemini can be miscalibrated, position-biased, or lenient on fluent
wrong answers. We mitigate by (a) using a model from a different family than the
generator, (b) `temperature=0` for reproducibility, and (c) anchoring the *retrieval*
half of the evaluation on human-labeled `relevant_chunk_ids`, which need **no** judge
at all. The retrieval metrics are the trustworthy backbone; RAGAS is corroborating
evidence, not the sole basis for any claim. This is listed as a threat to validity in
`EVALUATION.md`.

**The four RAGAS metrics (what each catches):**
- `faithfulness` — is every claim in the answer supported by the retrieved context?
  Low ⇒ hallucination. This is the headline groundedness number.
- `answer_relevancy` — does the answer actually address the question (vs. padding /
  drifting)?
- `context_precision` — are the retrieved chunks relevant, and are the relevant ones
  ranked *first*? A retrieval-ranking signal, judged rather than label-based.
- `context_recall` — could the ground-truth answer be produced from the retrieved
  context? Did we fetch *everything* needed?

---

## D3 — Config-driven pipeline, single code path

**Decision:** one `retrieve()` (`backend/retrieval/pipeline.py`) driven entirely by
YAML; the baseline and the full pipeline differ only by config (R5).

**Rejected:** keeping the current naive retrieval as-is and bolting features on as
separate scripts. That would make the baseline and the advanced pipeline
incomparable — the classic "the baseline was a different codebase" invalidation.

**Guardrail:** stages not yet built (BM25, RRF, multi-query, rerank) raise
`NotImplementedError` rather than silently no-op, so the sequencing rule (R4: no
feature before the baseline is frozen) is enforced by the code, not just discipline.

---

## D4 — Frozen-baseline retrieval depth = 20, context = 5

**Decision:** the frozen baseline retrieves **top-20** dense candidates and feeds the
**top-5** to the generator, even though the live app uses `k=4` end-to-end.

**Why depart from the app's k=4:**
1. `recall@10` and `nDCG@10` are undefined if only 4 candidates are ever returned —
   the metric columns in the ablation table require a depth of at least 10.
2. Holding the dense candidate depth (20) and the generation context size (5)
   **constant across every config** means the only thing that changes down the
   ablation is the retrieval *technique* — not how deep we look or how much the LLM
   reads. That isolates each component's contribution, which is the whole point.

The LLM still sees only 5 chunks (close to the app's 4), so answer-quality numbers
remain representative of the shipped system. This standardisation is disclosed in
`EVALUATION.md`.

---

## D5 — Deterministic where it counts

Generation runs at `temperature=0`; the retrieval metrics are pure functions;
`nomic-embed-text` embeddings are deterministic for a given input. So a rerun
reproduces the retrieval metrics exactly. The remaining nondeterminism is the Gemini
judge (network model, may drift version-to-version) — hence RAGAS is treated as
corroborating, and every run is logged with its config hash + timestamp to
`eval/results/runs.jsonl` (never overwritten) so results are traceable.

---

## D6 — The eval harness *is* the test suite

Non-deterministic LLM pipelines can't be tested with `assertEqual(answer, expected)`.
Instead: deterministic retrieval metrics are the fast regression gate (unit-tested
against hand-worked examples in `backend/tests/test_metrics.py`), and RAGAS is the
slow gate. A bug in the metric code would invalidate the whole table, so the metric
code itself is the most heavily tested part of the repo.

---

<!-- D7 (RRF / hybrid) added in Phase 4, D8 (cross-encoder) in Phase 6, before their code. -->
<!-- Baseline failure analysis (Phase 3) is appended below once the baseline is frozen. -->
