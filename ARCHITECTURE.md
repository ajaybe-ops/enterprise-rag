# Architecture

A single-process retrieval service. Every stage can degrade independently, and
every degradation is reported rather than hidden.

```
                 ┌──────────────────────────────────────────┐
  documents ────▶│ ingestion     parse → normalise → write   │
                 └───────────────┬──────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────┐
                 │ corpus        front matter → chunks      │
                 │               content-addressed manifest │
                 └───────────────┬──────────────────────────┘
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │  authorization   Principal → allow / withhold      │◀── fails closed
        └───────────────┬────────────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────────┐
        │  retrieval      BM25  ∥  dense(LSA)  → RRF fusion   │
        │                 → status filter/demotion → rerank   │
        └───────────────┬────────────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────────┐
        │  answering      injection scan → redact → delimit  │
        │                 → citation verify → abstain/answer  │
        └───────────────┬────────────────────────────────────┘
                        ▼
                 response + citations + manifest_id
```

## Stages

| Stage | Module | Failure behaviour |
| --- | --- | --- |
| Ingestion | `app/ingestion/ingest.py` | unparseable file skipped, reported; run continues |
| Corpus | `app/corpus/manifest.py` | no fallback — the manifest is the identity of the corpus |
| Retrieval | `app/retrieval/pipeline.py` | dense leg degrades to LSA; reported in `/health` |
| Status | `app/retrieval/status.py` | `superseded` excluded; `draft` demoted on preliminary queries |
| Authorization | `app/security/access.py` | **never degrades** — denies on any ambiguity |
| Injection | `app/security/injection.py` | `block` spans redacted; `flag` reported only |
| Answering | `app/answering.py` | unverifiable sentences dropped; all dropped ⇒ abstain |

## Retrieval

Two legs run independently and fuse with reciprocal rank fusion.

- **BM25** — sparse, lexical. Fast, and on this corpus it is the strongest leg.
- **Dense** — truncated SVD over a hashed bag-of-words matrix by default. No
  network, no model download, deterministic. See ADR 0004.

Fusion output goes through status policy, then reranking. Reranking is applied
*after* fusion, so it sees the merged set rather than one leg's opinion.

**Confidence escalation is currently disabled by default.** The ablation shows it
costs 0.133 pass rate on this corpus: escalation fires on mid-confidence queries
where a second pass adds noise rather than recall. See the table in `README.md`.

## Answers

The reference build is extractive. Retrieved text is the answer; there is no
generation step in the default path, and the provider hook is a single seam.

Verification is per sentence. A sentence survives if enough of its content words
appear in the retrieved evidence. Surviving sentences carry the chunk that
actually supports them, not the top-ranked chunk. If no sentence survives, the
system abstains rather than returning unverified text.

Two properties are deliberate and easy to break:

- **Authorization is checked after retrieval, not enforced by the retriever.**
  A filter that only trims the final list can still let a restricted document
  influence ranking. Filtering preserves order and reports what it withheld.
- **The delimited context and the reference answer are separate values.** The
  delimiter block is a prompt-transport concern. Feeding it through citation
  verification makes every sentence untraceable and the system correctly
  withdraws a good answer. This shipped once.

## Observability

`app/observability.py` is an in-process span recorder: no network, no account, no
configuration. `drain()` is the seam for a real collector.

The previous implementation required an OpenTelemetry SDK and a hosted exporter
account before the test suite could run, and warned on every import of an
unrelated module. A tracing system that must be provisioned before the code can
be read is not a tracing system, it is a signup flow.

## Evaluation

`app/evaluation/` is the reason the rest of this exists.

- `probe.py` — evidence contracts (`must_retrieve`, `must_contain`, `must_outrank`,
  `must_not`, `should_abstain`)
- `metrics.py` — deterministic structural metrics
- `runner.py` — offline execution, run reports, config fingerprints
- `regression.py` — comparison and causal attribution

Attribution distinguishes `corpus_edit` from a configuration change, and
`MetricDelta` carries an explicit `higher_is_better` per metric. Latency and leak
count rise-is-worse; recall and pass rate fall-is-worse. A uniform comparison
would have been blind to a pipeline getting slower or leakier.

## Module map

| Path | Role |
| --- | --- |
| `app/main.py` | FastAPI surface: `/health`, `/ask`, `/corpus` |
| `app/answering.py` | citation verification, abstention |
| `app/observability.py` | vendor-free structured tracing |
| `app/types.py` | `Chunk`, content-derived ids |
| `app/config.py` | immutable settings, env parsing |
| `app/corpus/` | manifests, front matter, diffs |
| `app/evaluation/` | probes, metrics, runner, regression |
| `app/retrieval/` | corpus loading, BM25, dense, fusion, status policy |
| `app/security/` | authorization, injection scanning |
| `app/ingestion/` | loaders and the ingest path |

## Security posture

- Authorization fails closed. `Principal(authenticated=False)` reads nothing.
- `POST /ask` takes `tenant` from the request body — a **reference harness**, not
  authentication. It is trivially spoofable and must move behind a real identity
  provider before deployment.
- Retrieved text is untrusted. It is scanned, redacted, and delimited before it
  could reach a prompt.
- Error responses return a type and message, never a stack trace.
