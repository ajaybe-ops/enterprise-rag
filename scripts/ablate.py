"""Paired ablations over the probe suite.

Why ablations need to be paired
-------------------------------
Comparing two runs of a retrieval system tells you almost nothing on its own,
because every run is a different set of documents, a different configuration and
a different machine. To say "the dense leg helps", the dense leg has to be the
*only* thing that changed between the two runs.

So every variant here is derived from one baseline configuration by changing
exactly one parameter, and every run is scored against the same probe set and
the same corpus manifest. The suite is the control.

What a variant is allowed to conclude
-------------------------------------
Each row is reported with a verdict about whether the change is *worth it*, not
just whether the number went up. A variant that raises recall by 0.004 and
doubles p95 latency is not an improvement, and a table of raw numbers invites
exactly that conclusion. The comparison is therefore stated in the units that
matter: quality against cost.

Determinism note
----------------
Variants run in a fixed order against a fixed corpus, and the suite is
deterministic, so the same command produces the same table. Latency is the one
column that is not reproducible on shared hardware, which is why it is reported
but never used to decide a verdict.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from enterprise_rag.config.settings import Settings  # noqa: E402
from enterprise_rag.indexing.manifest import attach_chunks, build_manifest  # noqa: E402
from enterprise_rag.evaluation.probe import ProbeSet  # noqa: E402
from enterprise_rag.evaluation.runner import RetrievalExecutor, config_fingerprint, run_probe_set  # noqa: E402
from enterprise_rag.indexing.corpus import load_corpus  # noqa: E402
from enterprise_rag.pipelines.retrieval import HybridRetriever, build_retrievers  # noqa: E402
from enterprise_rag.retrieval.status import StatusPolicy  # noqa: E402
from scripts.evaluate import assemble_context  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Variant:
    """One configuration, described by the single change that defines it."""

    name: str
    overrides: dict
    description: str
    #: Retrieval is rebuilt per variant; a dense index fit per variant would be
    #: the same work every time and would dominate the runtime.
    rebuild: bool = True


VARIANTS: tuple[Variant, ...] = (
    Variant("baseline", {}, "BM25 + dense, reranked, status-aware, final_k=5"),
    Variant("bm25-only", {"use_dense": False}, "lexical leg alone"),
    Variant("dense-only", {"use_bm25": False}, "dense leg alone"),
    Variant("no-rerank", {"rerank": "none"}, "fusion output without reranking"),
    Variant("no-escalation", {"adaptive": False}, "single retrieval pass"),
    Variant("final_k=3", {"final_k": 3}, "shallower answers"),
    Variant("final_k=10", {"final_k": 10}, "deeper answers"),
    Variant("candidate_k=10", {"candidate_k": 10, "final_k": 5}, "narrower candidate pool"),
    Variant("no-status-demotion", {"status_demotion": 1.0}, "drafts neither promoted nor demoted"),
    Variant("threshold=0.75", {"confidence_threshold": 0.75}, "much more willing to abstain"),
)


def build(settings: Settings, corpus, variant: Variant) -> HybridRetriever:
    retrievers = build_retrievers(settings, corpus)
    if not retrievers:
        raise SystemExit(f"{variant.name}: no retrievers enabled")
    return HybridRetriever(settings=settings, retrievers=retrievers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--out", default="runs/ablations.json")
    parser.add_argument("--markdown", default="runs/ablations.md")
    parser.add_argument("--only", nargs="*", help="restrict to named variants")
    args = parser.parse_args()

    root = pathlib.Path(args.corpus)
    corpus = load_corpus(root, source_type="true")
    probes = ProbeSet.load(args.probes, validate_against=corpus.sources)
    manifest = attach_chunks(build_manifest(root), corpus.chunks)
    base_settings = Settings.from_env()

    variants = [v for v in VARIANTS if not args.only or v.name in args.only]
    if args.only and len(variants) != len(set(args.only)):
        missing = set(args.only) - {v.name for v in variants}
        raise SystemExit(f"unknown variant(s): {sorted(missing)}")

    rows = []
    baseline_aggregate = None
    for variant in variants:
        settings = (
            dataclasses.replace(base_settings, retrieval=dataclasses.replace(base_settings.retrieval, **variant.overrides))
            if variant.overrides
            else base_settings
        )
        retriever = build(settings, corpus, variant)
        executor = RetrievalExecutor(
            retriever=retriever,
            settings=settings,
            top_k=settings.retrieval.final_k,
            assemble_context=assemble_context,
        )
        report = run_probe_set(
            probes,
            executor,
            label=variant.name,
            manifest_id=manifest.manifest_id,
            config_fingerprint_value=config_fingerprint(settings),
            k=settings.retrieval.final_k,
        )
        aggregate = report.aggregate
        if baseline_aggregate is None:
            baseline_aggregate = aggregate

        rows.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "overrides": variant.overrides,
                "config_fingerprint": report.config_fingerprint,
                "pass_rate": aggregate.pass_rate,
                "recall": aggregate.recall_at_k,
                "ndcg": aggregate.ndcg_at_k,
                "mrr": aggregate.mrr,
                "fact_coverage": aggregate.fact_coverage,
                "abstention": aggregate.abstention_accuracy,
                "authz_violations": aggregate.authz_violations,
                "p95_latency_ms": aggregate.p95_latency_ms,
                "mean_latency_ms": aggregate.mean_latency_ms,
                "by_kind": aggregate.by_kind,
                "delta_pass_rate": aggregate.pass_rate - baseline_aggregate.pass_rate,
            }
        )
        print(
            f"{variant.name:<22} pass {aggregate.pass_rate:.3f} "
            f"({aggregate.pass_rate - baseline_aggregate.pass_rate:+.3f})  "
            f"recall {aggregate.recall_at_k:.3f}  nDCG {aggregate.ndcg_at_k:.3f}  "
            f"p95 {aggregate.p95_latency_ms:.0f}ms"
        )

    payload = {
        "corpus_manifest_id": manifest.manifest_id,
        "dataset_id": probes.dataset_id,
        "k": baseline_aggregate.k,
        "rows": rows,
        "note": (
            "Latency is machine-dependent and is reported for context only; no verdict "
            "in this table is derived from it."
        ),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Ablations",
        "",
        f"Corpus `{manifest.manifest_id}`  Dataset `{probes.dataset_id}`  "
        f"{probes_dataset_count(probes)} probes",
        "",
        "Each row changes exactly one parameter relative to `baseline`; all other",
        "settings, the corpus and the probe set are held fixed.",
        "",
        "| variant | pass | delta | recall | nDCG | MRR | facts | abstain | authz | p95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | {row['pass_rate']:.3f} | {row['delta_pass_rate']:+.3f} | "
            f"{row['recall']:.3f} | {row['ndcg']:.3f} | {row['mrr']:.3f} | "
            f"{row['fact_coverage']:.3f} | {row['abstention']:.3f} | "
            f"{row['authz_violations']} | {row['p95_latency_ms']:.0f} |"
        )
    lines += ["", "## Per failure mode (pass rate)", "", "| variant | " + " | ".join(
        sorted({k for row in rows for k in row["by_kind"]})
    ) + " |", "| --- | " + " | ".join(["---"] * len({k for row in rows for k in row["by_kind"]})) + " |"]
    kinds = sorted({k for row in rows for k in row["by_kind"]})
    for row in rows:
        cells = [f"{row['by_kind'].get(kind, {}).get('pass_rate', 0.0):.2f}" for kind in kinds]
        lines.append(f"| `{row['variant']}` | " + " | ".join(cells) + " |")
    lines += ["", f"_{payload['note']}_", ""]

    md = pathlib.Path(args.markdown)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  -> {out}\n  -> {md}")

    # A variant that introduces an authorisation violation is a failed ablation
    # regardless of its quality numbers, so the process exit code reflects the
    # worst authz result across all variants.
    worst = max((row["authz_violations"] for row in rows), default=0)
    return 2 if worst else 0


def probes_dataset_count(probes: ProbeSet) -> int:
    return len(probes)


if __name__ == "__main__":
    raise SystemExit(main())
