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

## Phase 3 — Frozen baseline & failure analysis

**Frozen baseline** (`eval/results/baseline.json`, config hash `acd8c44d5b65`): naive
dense-only retrieval, depth 20, context 5, over the 40-question Silverstone golden
set (34 answerable). Deterministic retrieval metrics:

| recall@3 | recall@5 | recall@10 | MRR | nDCG@10 | hit_rate@10 | abstention (unanswerable) |
|---|---|---|---|---|---|---|
| 0.774 | 0.848 | 0.931 | 0.806 | 0.811 | 1.000 | 1.000 |

Per-type recall@5: **factual 1.00**, exact_term 0.917, comparative 0.80,
**paraphrase 0.688**. (RAGAS columns pending Gemini judge quota — see D2; the
harness fills them on a rerun without `--no-ragas`.)

**This is the hypothesis that motivates every feature that follows.** Two honest
headline findings:

1. **The naive baseline is already strong** (recall@5 = 0.85, hit_rate@10 = 1.0).
   That is expected on a **35-document corpus** — there is simply not much to confuse
   a dense retriever at this scale. So the interesting question is not "how do we lift
   the average" but "**which specific query shapes still fail, and does each proposed
   component fix its predicted shape?**" A component that doesn't move its target
   bucket on this corpus is a component we should *not* claim.

2. **The residual failures are concentrated and map cleanly onto the planned
   components** (every failing question below is from `baseline.json` `details`):

   - **Comparative — one entity crowds out the other** (the textbook dense failure):
     - `c_03` (Norris vs Piastri best lap): Piastri's stint ranks 2, **Norris's ranks 16**.
     - `c_08` (Hamilton vs Leclerc compound): Leclerc ranks 1, **Hamilton's ranks 15**.
     - `c_10` (Norris vs Verstappen avg lap): **both** relevant stints miss the top 5 (ranks 7, 13).
     → Top-k fills with chunks about *one* driver (or neither). This is precisely what
     **multi-query decomposition** (per-entity sub-queries, Phase 5) is meant to fix,
     and the reason decomposition — not paraphrase — is the interesting MQ variant here.

   - **Exact-term — dense can't pin an exact token**:
     - `e_05` ("which drivers ran **SOFT**"): needs HAM/LEC/STR stints; dense returns
       HARD and MEDIUM stints first (STR at 1, but **LEC at 7, HAM at 16**). Only 1 of 3
       relevant chunks in the top 5.
     → The lexical token `SOFT` carries the answer and the embedding smears across all
     "final stint" chunks regardless of compound. This is the one clear case for
     **BM25 + hybrid fusion** (Phase 4). Note it is *one* question — on this corpus the
     exact-term bucket is otherwise already 0.92 because driver codes (`VER`, `HAM`)
     appear verbatim in the chunk text and embed fine. **We should therefore expect
     BM25's gain to be real but narrow, concentrated almost entirely on compound-token
     queries — and we will report it honestly even if the average barely moves.**

   - **Paraphrase — relevant chunk retrieved but ranked just out of reach**:
     - `p_01` ("came out on top at the **British** GP" → Norris win): correct chunk at **rank 7**.
     - `p_02` ("**seven-time champion** at his home race" → Hamilton P4): correct chunk at **rank 6**.
     - `p_05` ("good day for **Ferrari**?" → Hamilton P4 + Leclerc P14): Hamilton at 1,
       **Leclerc's result not retrieved at all** (compound paraphrase + two-entity).
     → For `p_01`/`p_02` the right chunk is *present but under-ranked* — a **precision /
     ranking** problem, which is exactly what a **cross-encoder reranker** (Phase 6)
     should fix by pulling the rank-6/7 chunk into the top-5. `p_05` needs *both*
     decomposition (split Ferrari → Hamilton + Leclerc) and reranking.

**Predicted component → bucket mapping (to be confirmed or falsified by measurement):**

| Component | Predicted to fix | Failing questions it targets |
|---|---|---|
| BM25 + RRF (Phase 4) | exact compound-token queries | `e_05` |
| Multi-query decomposition (Phase 5) | two-entity comparatives | `c_03`, `c_08`, `c_10`, `p_05` |
| Cross-encoder rerank (Phase 6) | under-ranked-but-present chunks | `p_01`, `p_02`, and MRR/precision broadly |

Groundedness is already solid: **abstention_rate = 1.0** — the naive pipeline refused
all 6 unanswerable questions rather than hallucinating, so the later components must
be checked for *not regressing* refusal, not for establishing it.

---

## D7 — Hybrid search: BM25 + dense, fused with Reciprocal Rank Fusion

*Written before the Phase 4 implementation (R3).*

**Decision:** add a lexical BM25 leg alongside the dense leg and combine them with
**Reciprocal Rank Fusion (RRF)**.

**Why a lexical leg at all — grounded in the measured baseline (D-Phase-3):** dense
retrieval smears across semantically-similar chunks and cannot pin an exact token.
The frozen baseline shows this on `e_05` ("which drivers ran **SOFT**"): it returns
HARD/MEDIUM stint chunks ahead of the SOFT ones (LEC at rank 7, HAM at 16). BM25
scores exact term overlap, so `SOFT` in the query strongly matches `SOFT` in the
chunk. This is the specific, measured gap BM25 exists to close — not a fashion.

**BM25 implementation: `rank_bm25` (Python), not Postgres FTS.**
- *Chosen:* `rank_bm25` — true BM25, trivial to implement and explain, and the corpus
  is tiny (35 Silverstone chunks), so an in-memory index rebuilt per run is free and
  has no sync problem in practice.
- *Rejected:* Postgres `tsvector` + `ts_rank_cd`. It is the stronger answer *at scale*
  (one datastore, lexical and vector indices can't drift, metadata filters compose in
  SQL), and it is named as the better production choice — but it is not true BM25 (a
  ranking variant) and adds SQL surface for no measurable benefit at 35 documents. The
  honest tradeoff: `rank_bm25` does not scale (in-memory, duplicates the corpus); if
  this corpus grew to thousands of chunks the decision flips to Postgres FTS.
- The BM25 index is built over **only the Silverstone subset** (the same corpus filter
  applied to the dense leg), so the untested races cannot leak in via the lexical path.

**Fusion: RRF, not weighted score normalisation.**
`score(d) = Σ_retrievers 1 / (k + rank_r(d))`, with **k = 60**, over each retriever's
rank list (best rank = 1).
- *Why rank-based, not score-based:* cosine similarity (~0.6–0.7 here) and BM25 scores
  (unbounded, corpus-dependent) live on **incomparable scales**. Normalising and
  weighting them (min-max, z-score) is brittle — it depends on score distributions
  that shift per query. RRF throws the raw scores away and fuses **ranks**, which are
  directly comparable across any two retrievers. That scale-invariance is the whole
  reason RRF is the standard hybrid-fusion primitive.
- *What `k = 60` does:* it damps the influence of the very top ranks. With
  `1/(k+rank)`, the gap between rank 1 and rank 2 is `1/61 − 1/62 ≈ 0.00026` — small —
  so no single retriever's #1 can dominate the fused order; a document needs support
  from *both* legs (or a strong showing in one plus presence in the other) to rank
  high. Small k (e.g. 1) makes fusion behave almost like "trust whoever put it first";
  k = 60 (the value from the original RRF paper) is the robust default. It is a config
  knob (`fusion.rrf_k`), not a magic constant.
- *Implemented explicitly* in `backend/retrieval/fusion.py` (no black-box helper), so
  the arithmetic is ownable and unit-tested against a hand-computed toy example.

**Ablation plan:** evaluate **dense-only (= baseline), BM25-only, and hybrid** as
separate rows. BM25-only is not a wasted run — it is the evidence that fusion beats
*either* leg alone, and it turns "I added hybrid" into an experiment rather than a
fashion choice. Expectation from the baseline analysis: the gain is **narrow and
concentrated on `e_05`-type compound-token queries**; the corpus-wide average may
barely move, and that will be reported honestly.

### D7 result (measured — retrieval metrics, 34 answerable)

| Config | recall@5 | recall@10 | MRR | nDCG@10 | exact_term@5 | paraphrase@5 |
|---|---|---|---|---|---|---|
| dense (baseline) | 0.848 | 0.931 | **0.806** | **0.811** | 0.917 | 0.688 |
| bm25-only | 0.662 | 0.706 | 0.594 | 0.614 | **1.000** | 0.375 |
| hybrid (RRF) | 0.824 | **0.971** | 0.725 | 0.777 | **1.000** | 0.688 |

**The hypothesis held, with an honest twist.** BM25-only confirms the mechanism: it is
worse everywhere *except* exact-term, where it is perfect (1.00) — and it is the leg
that finally retrieves all three SOFT stints in `e_05`. It is also the worst on
paraphrase (0.375), which is exactly why we **fuse** rather than replace.

Hybrid then does two things and costs a third:
- **Closes the exact-term gap** (0.917 → 1.00) — BM25's contribution survives fusion.
- **Lifts deep recall** (recall@10 0.931 → 0.971) — the union of legs finds more.
- **But slightly lowers top-rank precision** (MRR 0.806 → 0.725, recall@5 0.848 →
  0.824). Equal-weight RRF lets BM25's noisy paraphrase/comparative rankings push some
  strong dense hits down out of the top slots.

**This is not a clean win, and that is the point.** Hybrid is a better *candidate
generator* (higher recall@10) but a slightly worse *final ranker* (lower MRR) on this
corpus. That is the precise motivation for the two-stage architecture: keep hybrid for
recall, then add a **cross-encoder reranker** (Phase 6) to restore precision at the
top. A weighted RRF (favouring the dense leg) would likely recover some MRR too and is
noted as future work — but reranking is the more general fix and the one measured next.
Latency cost of the BM25 leg is negligible (~0.3 ms in-memory vs ~160 ms for the dense
embed+query), so hybrid adds recall essentially for free on wall-clock.

---

## D8 — Multi-query expansion: decomposition, not just paraphrase

*Written before the Phase 5 implementation (R3).*

**Decision:** optionally expand the user query into N sub-queries with `llama3.2`,
retrieve (dense+BM25+RRF) for each, and fuse all the results with RRF. Two modes:
- **paraphrase** — N rewordings of the same question. Helps vague/paraphrase queries
  by giving the dense retriever several surface forms to match.
- **decompose** — split a multi-entity question into **per-entity sub-queries**
  ("compare HAM and VER" → "HAM final stint pace", "VER final stint pace").

**Why decomposition is the interesting mode here — grounded in the baseline (D-Phase-3
and D7):** the residual comparative failures (`c_03`, `c_08`, `c_10`, `p_05`) all have
the *same* shape — the query retrieves lots of chunks about **one** entity and starves
the other (Norris's stint at rank 16 while Piastri's is at rank 2). A single query
embedding is a blend of both entities and tends to collapse onto whichever dominates.
Decomposing into one sub-query per entity guarantees each entity gets its own retrieval
budget, then RRF merges them. This attacks the comparative failure mode **directly**,
where paraphrase expansion would not. Evaluated as a distinct variant from paraphrase so
the difference is measured, not assumed.

**Cost, stated up front:** multi-query is the most expensive component — N× the
retrieval calls **plus one extra LLM round-trip** to generate the sub-queries. If the
recall gain is small and p95 latency jumps, **that is a legitimate finding**: the honest
outcome is "measured it, the gain didn't justify the latency, made it configurable and
off by default," which is a stronger signal than uncritical feature-stacking. The
latency cost is reported in `EVALUATION.md`, not hidden.

### D8 result (measured — decompose mode; paraphrase variant reported in EVALUATION.md)

| Config | recall@5 | recall@10 | MRR | nDCG@10 | comparative@5 | paraphrase@5 | p50 ms |
|---|---|---|---|---|---|---|---|
| hybrid | 0.824 | **0.971** | 0.725 | 0.777 | 0.75 | 0.688 | ~160 |
| hybrid + MQ (decompose) | 0.804 | 0.878 | **0.754** | 0.739 | **0.70** | **0.875** | **2445** |

**This is the "measured it, didn't justify it" outcome — and the most useful result in
the build.** Three honest findings:

1. **Latency explodes ~15×** (p50 160 ms → 2445 ms). The single LLM expansion call is
   ~1.5 s on its own; the rest is N× the retrieval. For an interactive assistant this is
   a large regression.
2. **Paraphrase recall jumps** (0.688 → 0.875) — rewording the query gives the dense
   retriever several surface forms of a vague question, which genuinely helps.
3. **But decomposition *hurt* the bucket it was designed for** — comparative fell
   (0.75 → 0.70) and deep recall dropped (recall@10 0.971 → 0.878). Two reasons, both
   worth stating: (a) RRF across many sub-query lists rewards chunks that appear in
   *several* sub-queries and can drop a uniquely-relevant chunk found by only one; and
   (b) `llama3.2` cannot decompose entities it doesn't know — e.g. "was it a good day for
   **Ferrari**?" did not split into Hamilton + Leclerc, because the 3B model doesn't know
   the lineup. A stronger expansion model (or a driver-roster tool) would likely fix (b).

**Decision: keep multi-query implemented but OFF by default.** The only clear win
(paraphrase) does not justify a 15× latency hit plus a regression on comparatives. This
is exactly the anti-feature-stacking result an R&D reviewer is screening for: it stays a
config flag, and the honest recommendation is "not worth shipping as-is on this corpus."

---

## D9 — Cross-encoder reranking: the precision second stage

*Written before the Phase 6 implementation (R3).*

**Decision:** retrieve a wide candidate set (hybrid, k≈20), then rerank it with a
**cross-encoder** (`BAAI/bge-reranker-base`, CPU) and keep the top-n (5) for the
generation context.

**Bi-encoder vs cross-encoder — the distinction the interviewer will probe:**
- A **bi-encoder** (our `nomic-embed-text` dense retriever) embeds the query and each
  document *independently* into vectors, then compares with cosine. Because the document
  vectors don't depend on the query, they can be **precomputed and ANN-indexed** (the
  HNSW index) → fast, scales to the whole corpus → but there is **no token-level
  interaction** between query and document; relevance is judged only by distance between
  two separately-formed summaries.
- A **cross-encoder** takes `(query, document)` **jointly** through one transformer, so
  every query token can attend to every document token → a far sharper relevance
  judgement → but nothing can be precomputed and it costs **one forward pass per
  candidate** at query time → it can only be run on a **shortlist**, never the corpus.

**Hence the two-stage architecture, which is the whole point:** a cheap, recall-oriented
first stage (hybrid: bi-encoder + BM25) casts a wide net; an expensive, precision-oriented
second stage (cross-encoder) re-sorts the shortlist. This is exactly the fix the D7 result
called for: hybrid raised recall@10 but *lowered* MRR/precision by letting BM25 noise into
the top ranks — the reranker's job is to restore top-rank precision **without** giving up
the recall the wide first stage bought.

**Model choice:** `BAAI/bge-reranker-base` — strong accuracy, ~280 MB, runs on CPU in a
few hundred ms for ~20 pairs (acceptable here; the corpus is tiny and queries are
interactive-but-not-realtime). Rejected `ms-marco-MiniLM-L-6-v2` (smaller/faster but
weaker) as the default; it stays available via config if CPU latency proves too high.

**Candidate-set sweep:** evaluate k ∈ {10, 20, 50} candidates into the reranker and plot
recall/precision against latency — a genuine tradeoff curve (more candidates = more recall
headroom for the reranker to exploit, but linearly more CPU forward passes). Reported in
`EVALUATION.md`.

### D9 result (measured — a negative result, honestly reported)

| Config | recall@5 | recall@10 | MRR | nDCG@10 | comparative@5 | exact_term@5 | p50 ms |
|---|---|---|---|---|---|---|---|
| hybrid (no rerank) | **0.824** | **0.971** | **0.725** | **0.777** | 0.75 | **1.00** | ~160 |
| hybrid + rerank, k=10 | 0.789 | **0.971** | 0.694 | 0.749 | **0.85** | 0.79 | ~331 |
| hybrid + rerank, k=20 | 0.760 | 0.828 | 0.655 | 0.677 | 0.75 | 0.79 | ~497 |
| hybrid + rerank, k=50 | 0.774 | 0.814 | 0.636 | 0.662 | 0.80 | 0.79 | ~598 |

**The cross-encoder did not help on this corpus — and the honest explanation matters more
than a clean win.** Two findings:

1. **Reranking lowered overall recall@5, MRR and nDCG, and broke exact-term (1.00 →
   0.79).** `bge-reranker-base` is trained on general web/MS-MARCO relevance; our chunks
   are terse F1 telemetry summaries. For `e_05` ("which drivers ran SOFT") the reranker
   promotes *result* docs that name the driver over the *stint* docs that actually record
   the compound — its notion of relevance doesn't align with our stint/result distinction.
   It *did* help comparatives (`c_03`'s Norris stint jumped from rank 16 into the top-5),
   which is real, but not enough to offset the exact-term regression.
2. **The candidate sweep is inverted: k=10 beats k=20 and k=50.** More candidates give an
   out-of-domain reranker more distractors to be fooled by, so precision *falls* as the
   shortlist grows. That is the opposite of the "more candidates = more headroom" intuition,
   and it is only visible because we swept it.

**Decision: do not enable reranking by default on this corpus.** The right fix is not a
bigger shortlist but a *domain-appropriate* reranker — either fine-tuning the cross-encoder
on F1 relevance labels, or (more honest for the data we actually have) a **metadata-aware
signal** that knows a "which compound" question wants a stint doc, not a result doc. That
belongs with the text-to-SQL router in "what I'd do next" (`EVALUATION.md`), because these
are *structured* queries semantic reranking is the wrong tool for.

### Overall verdict (the reason the ablation exists)

Across all nine configs, **the naive dense baseline has the best recall@5 (0.848), MRR
(0.806) and nDCG@10 (0.811).** On a 35-document, semantically-separable corpus, dense
retrieval is already near its ceiling, and the advanced stack mostly moves quality *around*
between buckets rather than up:
- **hybrid** is the one keeper — it closes the exact-term gap (`e_05`) and lifts recall@10,
  for the price of a slight MRR dip and a near-zero-cost BM25 leg. Ship it.
- **rerank** and **multi-query** each rescue one bucket (comparative, paraphrase respectively)
  while regressing others, at real latency cost. Keep as flags, off by default.
- the **full** stack is the worst of all worlds (worst-case latency, precision below
  baseline) — the concrete cautionary tale against feature-stacking.

This is the honest R&D conclusion the whole build was designed to be able to state *with
numbers*: I would ship dense+hybrid, and I can prove the rest doesn't earn its place **on
this corpus** — while being explicit that a larger, noisier corpus would likely flip several
of these calls.

---

