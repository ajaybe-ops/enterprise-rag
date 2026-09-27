"""Run the probe suite against the retrieval stack, and diff it against a baseline.

This is the command the whole evaluation system exists to serve:

    # establish a baseline
    python scripts/evaluate.py --corpus corpus/knowledge_base --probes probes/enterprise.json

    # after a corpus edit, or a code change, or a config change
    python scripts/evaluate.py --corpus corpus/knowledge_base --probes probes/enterprise.json \\
        --baseline runs/baseline.json --out runs/candidate.json

With `--baseline` it prints a regression report: the suite verdict, the metric
deltas, the corpus diff, and -- the point of the exercise -- a per-probe cause
for every regression, with the evidence behind it.

It also runs fully offline by default. BM25 is lexical and the dense leg falls
back to a local latent-semantic index, so a baseline can be established on a
laptop with no API key, no vector database and no network. That is a requirement,
not a convenience: a quality gate that needs infrastructure cannot be run on
every commit, and a gate that is not run on every commit does not exist.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.corpus.manifest import attach_chunks, build_manifest  # noqa: E402
from app.evaluation.probe import ProbeSet  # noqa: E402
from app.evaluation.regression import compare  # noqa: E402
from app.evaluation.runner import RetrievalExecutor, RunReport, config_fingerprint, run_probe_set  # noqa: E402
from app.retrieval.corpus import load_corpus  # noqa: E402
from app.retrieval.pipeline import HybridRetriever, build_retrievers  # noqa: E402


def assemble_context(chunks) -> str:
    """Concatenate retrieved spans the way the serving path does.

    Shared with the API so `fact_coverage` measures the real assembly step. A
    gate that measured its own private assembly would happily pass a system that
    ships a different one.
    """
    return "\n\n".join(
        f"[{getattr(chunk, 'source', '?')}#{getattr(chunk, 'ordinal', '?')}]\n{chunk.text}"
        for chunk in chunks
    )


def tenant_scope(probe):
    """Documents a probe's subject is allowed to see.

    `public` is treated as "every tenant's public material plus the shared
    corpus", which for the generated suite means the restricted documents are
    reachable only by their own tenant. Encoded as a filter the executor applies
    to results, so a probe can observe a leak rather than merely assert one.
    """
    tenant = probe.tenant or "public"
    if tenant == "public":
        return None
    return [tenant, "public"]


def build_settings(args) -> Settings:
    settings = Settings.from_env()
    overrides = {}
    if args.no_dense:
        overrides["use_dense"] = False
    if args.no_bm25:
        overrides["use_bm25"] = False
    if args.final_k is not None:
        overrides["final_k"] = args.final_k
    if args.candidate_k is not None:
        overrides["candidate_k"] = args.candidate_k
    if args.rerank is not None:
        overrides["rerank"] = args.rerank
    if args.no_adaptive:
        overrides["adaptive"] = False
    if args.threshold is not None:
        overrides["confidence_threshold"] = args.threshold

    if not overrides:
        return settings

    retrieval = dataclasses.replace(settings.retrieval, **overrides)
    if retrieval.candidate_k < retrieval.final_k:
        raise SystemExit(
            f"candidate_k ({retrieval.candidate_k}) must be >= final_k ({retrieval.final_k})"
        )
    return dataclasses.replace(settings, retrieval=retrieval)


def build_report(args) -> RunReport:
    corpus_root = pathlib.Path(args.corpus)
    settings = build_settings(args)

    corpus = load_corpus(corpus_root, source_type="true", max_chars=args.chunk_size)
    if not corpus.chunks:
        raise SystemExit(f"no chunks loaded from {corpus_root}")

    probes = ProbeSet.load(args.probes, validate_against=corpus.sources)

    # The manifest is built from documents alone and then has chunk ids attached.
    # It has to be *this* manifest, not a fresh document-only one, when checking
    # the probe set's recorded corpus id: a document-only manifest necessarily
    # hashes differently, so comparing against it warns on every single run and
    # trains people to ignore the warning that matters.
    manifest = attach_chunks(build_manifest(corpus_root), corpus.chunks)
    if probes.corpus_manifest_id and probes.corpus_manifest_id != manifest.manifest_id:
        print(
            f"warning: probe set was generated against manifest "
            f"{probes.corpus_manifest_id} but the corpus is {manifest.manifest_id}",
            file=sys.stderr,
        )

    retrievers = build_retrievers(settings, corpus)
    if not retrievers:
        raise SystemExit("no retrievers enabled; check use_bm25/use_dense")
    hybrid = HybridRetriever(settings=settings, retrievers=retrievers)

    executor = RetrievalExecutor(
        retriever=hybrid,
        settings=settings,
        top_k=args.k,
        assemble_context=assemble_context,
        tenant_filter=tenant_scope,
    )

    report = run_probe_set(
        probes,
        executor,
        label=args.label or f"{hybrid.strategy} k={args.k}",
        manifest_id=manifest.manifest_id,
        config_fingerprint_value=config_fingerprint(settings),
        k=args.k,
        notes=args.notes or "",
    )
    if args.save_manifest:
        manifest.save(args.save_manifest)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, help="corpus root directory")
    parser.add_argument("--probes", required=True, help="probe set JSON")
    parser.add_argument("--out", default="runs/candidate.json", help="where to write the run report")
    parser.add_argument("--baseline", help="baseline run report to compare against")
    parser.add_argument("--label", default="", help="human label for this run")
    parser.add_argument("--notes", default="", help="free-text note recorded in the report")
    parser.add_argument("--k", type=int, default=5, help="final result depth to score")
    parser.add_argument("--chunk-size", type=int, default=1200, help="chunk size; changing it is an ablation")
    parser.add_argument("--save-manifest", help="also write the corpus manifest here")

    # Ablation knobs. Each changes exactly one thing, which is what makes a
    # paired comparison informative.
    parser.add_argument("--no-dense", action="store_true", help="disable the dense leg (BM25 only)")
    parser.add_argument("--no-bm25", action="store_true", help="disable the lexical leg (dense only)")
    parser.add_argument("--no-adaptive", action="store_true", help="disable confidence escalation")
    parser.add_argument("--final-k", type=int, help="override final_k")
    parser.add_argument("--candidate-k", type=int, help="override candidate_k")
    parser.add_argument("--rerank", choices=["rrf", "jina", "none"], help="override rerank strategy")
    parser.add_argument("--threshold", type=float, help="override confidence_threshold")

    args = parser.parse_args()
    report = build_report(args)
    report.save(args.out)

    aggregate = report.aggregate
    # `violations`, not `leaks`. The two are not the same number and conflating
    # them is how a security metric becomes decorative:
    #
    #   leakage_count        every chunk from a forbidden source that appeared
    #                        anywhere in the retrieved set
    #   authz_violations     forbidden chunks that the caller was not entitled
    #                        to see, at any rank
    #   preference_violations forbidden chunks that outranked a required source
    #
    # A system that retrieves a superseded document and correctly ranks the
    # current one first has a non-zero leakage_count and zero violations. The
    # previous headline printed that non-zero number as "leaks N", and the exit
    # code returned 2 on it -- so a fully correct run failed CI, which is how a
    # gate gets disabled.
    print(
        f"{report.label}: pass {aggregate.pass_rate:.3f}  "
        f"recall@{report.k} {aggregate.recall_at_k:.3f}  "
        f"nDCG@{report.k} {aggregate.ndcg_at_k:.3f}  "
        f"MRR {aggregate.mrr:.3f}  "
        f"facts {aggregate.fact_coverage:.3f}  "
        f"violations {aggregate.authz_violations:.0f}authz/"
        f"{aggregate.preference_violations:.0f}pref  "
        f"p95 {aggregate.p95_latency_ms:.0f}ms"
    )
    if aggregate.leakage_count:
        print(
            f"  note: {aggregate.leakage_count} forbidden-source chunk(s) retrieved but "
            f"correctly outranked -- not a violation"
        )
    weakest = aggregate.weakest_kinds()
    if weakest:
        print("  weakest: " + ", ".join(f"{kind} {rate:.2f}" for kind, rate in weakest))
    print(f"  -> {args.out}")

    if not args.baseline:
        # Gate on violations, not on leakage. Gating on leakage_count fails a run
        # in which the pipeline behaved correctly.
        return 0 if (aggregate.authz_violations or aggregate.preference_violations) == 0 else 2

    baseline = RunReport.load(args.baseline)
    manifest_path_before = pathlib.Path(args.baseline).with_name("manifest.json")
    manifest_path_after = pathlib.Path(args.save_manifest) if args.save_manifest else None

    probes = ProbeSet.load(args.probes)
    report_out = compare(
        baseline,
        args.out,
        probes=probes,
        manifest_before=manifest_path_before if manifest_path_before.exists() else None,
        manifest_after=manifest_path_after,
    )
    if manifest_path_after:
        text = report_out.to_markdown()
        target = pathlib.Path(args.out).with_name("regression.md")
        target.write_text(text, encoding="utf-8")
        print()
        print(text)
        print(f"  -> {target}")

    regressed = [d for d in report_out.metric_deltas if d.regressed]
    return 1 if (regressed or report_out.blocking) else 0


if __name__ == "__main__":
    raise SystemExit(main())
