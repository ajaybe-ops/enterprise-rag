# Enterprise RAG

A retrieval system built around one idea: **you cannot improve a metric you
cannot explain.**

Most RAG evaluation reports a number — recall@k, nDCG, an LLM-as-judge score —
and when the number moves, nothing tells you whether the corpus changed, the
chunker changed, the ranking changed, or the run was just noisy. So the number
gets tuned, and the tuning is invisible.

This project treats the corpus as a versioned artifact and every evaluation run
as attributable to a specific version of it. When a metric moves, the tool
tells you which of those four things moved with it.

```
Verdict: corpus_edit - corpus changed (~1 modified) while configuration held

| metric          | baseline | candidate | delta   | regressed |
| --------------- | -------- | --------- | ------- | --------- |
| pass_rate       | 0.8667   | 0.8444    | -0.0223 | YES       |
| recall_at_5     | 0.9333   | 0.9333    | +0.0000 | no        |
| ndcg_at_5       | 0.9074   | 0.9074    | +0.0000 | no        |
| fact_coverage   | 1.0000   | 0.9778    | -0.0222 | YES       |
| p95_latency_ms  | 75.3000  | 96.3800   | +21.08  | YES       |

### [high] version_conflict.atlas-ingest
- obligations lost: facts_hit
- cause: corpus_edit - required document atlas-ingest.slo.md was edited
  - content 0f4e175633127e5c -> fc24ceefee365808
```

Editing one number in one document moved fact coverage and left every ranking
metric untouched. No reranker change would have produced that signature, and no
prompt change would have produced it either. It is a content regression, and the
report says so.

---

## Quick start

```bash
pip install -r requirements.txt -r requirements-dev.txt

# 1. Generate a seeded corpus and its probe suite (reproducible byte-for-byte)
python scripts/generate_corpus.py

# 2. Score the retrieval pipeline against evidence contracts
python scripts/evaluate.py \
    --corpus evaluation/generated/docs \
    --probes evaluation/generated/probes.json \
    --out runs/baseline.json \
    --save-manifest runs/manifest.json

# 3. Serve it
uvicorn app.main:app --reload
pytest          # 68 tests
```

---

## What it measures

Not similarity to a reference answer. Whether the retrieved evidence actually
supports what the system claims.

| Contract | Meaning |
| --- | --- |
| `must_retrieve` | a required source appears in the top *k* |
| `must_contain` | a required fact string appears in retrieved text |
| `must_outrank` | a source must outrank every source in `must_not` |
| `must_not` | forbidden sources — authorization, or superseded versions |
| `should_abstain` | the system must decline to answer |

The distinction that matters is between the two kinds of `must_not`:

- **Authorization** (`must_not_kind="authorization"`) is absolute. A forbidden
  document at rank 7 is still a leak.
- **Version preference** (`must_not_kind="preference"`) is about *ordering*.
  Retrieving a superseded document is fine; ranking it above the current one is
  the bug.

Conflating them produces a metric that punishes correct behaviour. An earlier
version of this tool reported `leaks 18` — and exited non-zero — for a run in
which all 18 were correctly outranked and there were zero violations. The exit
code is now gated on `authz_violations + preference_violations`, with the
retrieval-but-outranked count reported separately as a note.

Metrics are computed from source-level and fact-level structure only, so they
are deterministic, need no LLM judge, and cost nothing to re-run. A judge-based
score is not reproducible enough to be a regression gate.

---

## Current results

87 documents, 90 probes, all reproducible offline. These are **relative** numbers
on a synthetic corpus — they are not a claim about production quality.

```
pass 0.867   recall@5 0.933   nDCG@5 0.907   MRR 0.811
facts 1.000   violations 0authz/0pref   p95 ~75ms
```

Weakest kind: `multi_hop` at 0.33. That is a genuine limitation, left visible
rather than tuned away.

### Ablations

Every variant differs from the baseline by exactly one parameter, scored against
the same corpus and probes.

| Variant | Pass | Δ | p95 |
| --- | --- | --- | --- |
| baseline (hybrid + rerank + escalation) | 0.867 | — | 100ms |
| **bm25-only** | **1.000** | **+0.133** | 16ms |
| dense-only | 1.000 | +0.133 | 319ms |
| no-escalation | 1.000 | +0.133 | 53ms |
| no-rerank | 0.800 | −0.067 | 88ms |
| candidate_k=10 | 0.800 | −0.067 | 85ms |
| no-status-demotion / final_k=3 / threshold=0.75 | 0.867 | 0.000 | — |

**The hybrid baseline loses to its own single legs, and confidence-driven
escalation is actively harmful** — `no-escalation` gains 0.133 pass rate. Dense
retrieval costs 20× the latency of BM25 for the same score.

This is the kind of finding the tool exists to surface. A conventional
demonstration would have shipped the hybrid stack, presented the ensemble as the
result, and never noticed it was losing to `bm25-only` on the same probes. On
this corpus the honest recommendation is to drop escalation and either commit to
BM25 or measure a real embedding model before paying for the dense leg.

`context_precision` sits at 0.242, which is a real weakness — the retrieved sets
are larger and noisier than the probes require.

---

## Design decisions

**A manifest is content-addressed.** Documents are hashed, chunks are
content-derived, and the manifest id folds in chunk identity — not chunk *count*.
Hashing only the count would make a rechunk invisible: same number of chunks,
different boundaries, same id, and a retrieval regression caused entirely by
chunking would be blamed on the code. `tests/test_manifest.py` pins this.

**Metric direction is explicit.** `MetricDelta` carries `higher_is_better`.
Recall falling is a regression; latency rising is a regression; a leak count
rising is a regression. The previous uniform comparison flagged only decreases,
which meant a pipeline that became 10× slower or started leaking was reported
healthy — the gate was blind to the two failures it existed to catch.

**Authorization fails closed.** `Principal.may_read` denies on missing tenant,
missing credentials, and an unauthenticated principal. Filtering preserves
retrieval order, so a security check can never change ranking as a side effect
and be reviewed in isolation.

**Injection redaction is span-scoped, not document-scoped.** Dropping a whole
runbook because one sentence looks adversarial discards the legitimate
operational content beside it. The instruction is removed; the document stays.

**Injected text is never trusted.** Retrieved documents are redacted and
delimited before they reach a prompt. The delimited context and the reference
build's answer are kept as separate values — feeding the delimiter block through
citation verification makes every sentence "untraceable" and the system
correctly withdraws a good answer. That bug shipped once; the docstring now
explains why the two must not be conflated.

**No vendor dependencies.** Tracing is a ~150-line in-process span recorder with
no network, no account, and no configuration. The previous implementation
required an OpenTelemetry SDK and a hosted exporter account before the test
suite could run, and emitted a warning on every import of an unrelated module —
a warning that trains people to ignore warnings.

`requirements.txt` lists 11 packages, derived from the actual import surface.
It previously listed 25, of which 14 were unreachable from any import.

---

## Layout

```
app/
  main.py            FastAPI: /health, /ask, /corpus
  answering.py       citation verification and abstention
  observability.py   vendor-free structured tracing
  corpus/            content-addressed manifests and diffs
  evaluation/        probe contracts, metrics, runner, regression attribution
  retrieval/         corpus loading, hybrid retrieval, status policy
  security/          authorization, injection scanning
  ingestion/         parsing and the ingest path
scripts/
  generate_corpus.py seeded corpus + probe generator
  evaluate.py        scoring, regression comparison, CI exit codes
  ablate.py          paired one-parameter ablations
tests/               68 tests
```

## Honest limitations

- `tenant` defaults to `public` when absent, and `status` defaults to `current`.
  Both are fail-open in the direction of serving. The trade-off is documented at
  the definition: the stricter alternative makes a default corpus silently empty,
  and an empty result is harder to notice than a wrong one. Ingest writes both
  fields explicitly; a production corpus must carry them.
- `POST /ask` takes `tenant` from the request body. That is a **reference
  harness**, not an authentication mechanism — it is trivially spoofable by
  design, so the tenant resolution must move behind a real identity provider
  before deployment. `Principal.authenticated=False` denies everything, which is
  the shape a real auth layer plugs into.
- All numbers are relative, on a synthetic corpus, with no external services. No
  claims about hosted LLM quality, production latency, or cost.
- Multi-hop retrieval is weak (0.33) and the hybrid stack is currently
  out-performed by BM25 alone on these probes.
- Latency figures are single-run on one machine and include no warm-up control.

## License

MIT.
