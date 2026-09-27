"""Generate a reproducible evaluation corpus and its ground-truth probe set.

Why generate instead of hand-writing
------------------------------------
A hand-written corpus has three problems that make it useless as a regression
gate:

1. It saturates. The bundled six-document smoke corpus returns MRR 1.000 -- the
   system cannot be distinguished from a broken one that always returns the right
   answer first, because it always does.
2. Nobody can write interesting failure cases by accident. A supersession
   conflict, a lexical trap, a cross-tenant leak and an unanswerable question all
   have to be *deliberately constructed*, and hand-written corpora reliably end
   up as a wall of easy questions.
3. Ground truth drifts from the corpus. Fix a typo in a document and the probe
   that asserted on the old wording fails for a reason that has nothing to do
   with retrieval.

Generating the corpus and its probes from one seed in one pass fixes all three:
ground truth is derived from the same values used to write the text, so it cannot
be wrong; the failure modes are constructed on purpose; and the corpus scales to
a few hundred documents, which is what gives a metric enough samples to move
when something actually changes.

Seeded, not random
------------------
`--seed` fixes the output byte for byte. A corpus that differs between two
machines cannot be used to compare two runs, and the whole attribution mechanism
depends on being able to reconstruct it exactly. Every run therefore also writes
a manifest, so the corpus identity is checked rather than assumed.

What is honest about the result
-------------------------------
Synthetic text is not real retrieval difficulty. It is a *controlled* environment
for detecting change, and the numbers it produces support relative claims --
"this edit cost 4 points of recall" -- and not absolute ones. The README states
this, and the real Kubernetes corpus is kept alongside for smoke coverage. See
docs/adr/0004.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.corpus.manifest import attach_chunks, build_manifest  # noqa: E402
from app.evaluation.probe import Probe, ProbeSet, dataset_id_for  # noqa: E402
from app.retrieval.corpus import load_corpus  # noqa: E402

# Vocabulary chosen so generated documents read like runbooks rather than like
# word salad: each domain term is a real operational concept, and each service
# name is a unique token that no other service can match. That uniqueness is
# load-bearing -- it guarantees every probe is winnable, so a failure means the
# pipeline is at fault rather than the corpus being unfair.
SERVICES = (
    "atlas-ingest", "atlas-router", "beacon-scheduler", "cinder-replicator",
    "delta-compactor", "ember-gateway", "flint-indexer", "gantry-queue",
    "harbor-cache", "ionis-store", "juniper-auth", "kestrel-metrics",
    "lumen-billing", "meridian-notify", "nimbus-registry", "orchid-search",
    "pyrite-archive", "quartz-scheduler",
)

TENANTS = ("acme", "globex")

# Distractor topics: high lexical overlap with a service, no authoritative
# answer. These are what make a "lexical_trap" probe possible.
TRAP_SUFFIXES = ("runbook-legacy", "postmortem-draft", "design-rfc", "faq", "glossary")

COMPONENTS = ("ingest", "router", "scheduler", "replicator", "compactor", "gateway")


def _slug(service: str, kind: str) -> str:
    return f"{service}.{kind}.md"


def _front_matter(fields: dict[str, str]) -> str:
    lines = ["---"]
    lines += [f"{key}: {value}" for key, value in fields.items()]
    lines += ["---", ""]
    return "\n".join(lines)


def build_corpus(seed: int, count: int) -> tuple[dict[str, str], dict[str, dict]]:
    """Return (documents, ground_truth) for `count` services.

    The two are produced together on purpose: every fact a probe will assert on
    is the same value that was written into the document text.
    """
    rng = random.Random(seed)
    documents: dict[str, str] = {}
    truth: dict[str, dict] = {}

    for index, service in enumerate(SERVICES[:count]):
        component = COMPONENTS[index % len(COMPONENTS)]
        tenant = TENANTS[index % len(TENANTS)]
        major, minor = 1 + (index % 3), index % 7

        # Only genuinely tenant-scoped material carries a non-public tenant.
        # Marking every document as tenant-owned is both unrealistic -- an SLO
        # and a runbook describe shared infrastructure and are normally readable
        # by the whole organisation -- and it makes the access-control test
        # meaningless, because a caller restricted to `public` would then see an
        # empty corpus and every probe would fail on authorisation rather than on
        # retrieval. The restriction belongs on the commercially sensitive
        # capacity document, and only there.
        public = {"tenant": "public"}

        # Numbers chosen per service and reused verbatim everywhere, so a probe
        # asserting "250" cannot pass against a different service's document.
        p99 = rng.choice((120, 180, 250, 320, 450, 600, 850))
        error_budget = rng.choice((0.1, 0.25, 0.5, 1.0, 2.5))
        queue_depth = rng.choice((500, 1000, 2500, 5000, 10000))
        replica_lag = rng.choice((5, 15, 30, 60, 120))
        page_threshold = rng.choice((70, 80, 85, 90, 95))

        # -- current specification ---------------------------------------- #
        spec = _front_matter(
            {
                "version": f"{major}.{minor}",
                "updated": f"2026-0{1 + index % 9}-1{index % 9}",
                "owner": f"team-{component}",
                "status": "current",
                **public,
            }
        ) + f"""# {service} specification

The {service} service owns the {component} path for the platform. It is
deployed as a stateless deployment with horizontal autoscaling driven by queue
depth. Traffic enters through the internal mesh and is authenticated with a
service identity token.

## Capacity

The service is sized for a sustained queue depth of {queue_depth} messages per
replica. Autoscaling begins above that figure and the target maximum is four
replicas. Sustained depth above the target triggers a capacity review rather than
automatic scale-out, because the downstream {component} tier is the binding
constraint.

## Replication

Replica lag is monitored as a first-class signal. The alerting threshold is
{replica_lag} seconds, and the service is considered degraded once lag exceeds
twice that value. Automatic failover is not enabled; recovery is operator-initiated
so that a bad configuration change is not replicated across the fleet.

## Configuration

All tunables live in the service config map. Restarting the deployment applies
changes. There is no hot-reload path for the {component} subsystem, and adding
one is tracked but not scheduled.
"""
        documents[_slug(service, "spec")] = spec

        # -- SLO document: the canonical home of the latency number -------- #
        slo = _front_matter(
            {
                "version": f"{major}.{minor}",
                "updated": f"2026-0{1 + index % 9}-1{index % 9}",
                "owner": f"team-{component}",
                "status": "current",
                **public,
            }
        ) + f"""# {service} service level objectives

These are the contractual objectives for {service}. They are reviewed quarterly
and deviations are reported to the owning team.

## Latency

The p99 request latency objective for {service} is {p99} milliseconds, measured
at the edge over a rolling 28-day window. A breach is defined as p99 exceeding
the objective for more than one hour.

## Availability

The availability objective is 99.9 percent measured monthly. Planned maintenance
is excluded from the calculation only when announced at least 72 hours ahead.

## Error budget

The error budget for {service} is {error_budget} percent of requests. When the
budget is exhausted, feature work is halted and reliability work is prioritised.
"""
        documents[_slug(service, "slo")] = slo

        # -- runbook: the page threshold ---------------------------------- #
        runbook = _front_matter(
            {
                "version": f"{major}.{minor}",
                "updated": f"2026-0{1 + index % 9}-1{index % 9}",
                "owner": f"team-{component}",
                "status": "current",
                **public,
            }
        ) + f"""# {service} on-call runbook

Page the on-call engineer for {service} when the burn rate exceeds the threshold
or when the service has been in a degraded state for fifteen minutes.

## Alerting

The primary alert fires when error rate stays above the objective for ten
minutes. A secondary alert fires when replica lag exceeds {replica_lag} seconds.

## First response

Confirm the alert against the dashboard before taking action. If queue depth is
above {queue_depth}, scale the {component} tier first. Do not restart the
deployment to clear a latency alert; the restart is more disruptive than the
symptom.
"""
        documents[_slug(service, "runbook")] = runbook

        # -- superseded revision: the version_conflict material ----------- #
        # Only some services get one. A corpus where every document has a
        # conflict teaches nothing about how often conflicts matter.
        has_conflict = index % 3 == 0
        if has_conflict:
            stale_p99 = p99 + rng.choice((150, 300, 600))
            stale = _front_matter(
                {
                    "version": f"{major}.0",
                    "updated": "2025-03-11",
                    "owner": f"team-{component}",
                    "status": "superseded",
                    "supersedes": "none",
                    **public,
                }
            ) + f"""# {service} service level objectives (2025 revision)

> Superseded. Retained for audit history only; the current objectives live in
> the current SLO document.

These were the objectives agreed for {service} in the 2025 planning cycle.

## Latency

The p99 request latency objective for {service} was {stale_p99} milliseconds.
This figure was relaxed in the following planning cycle after the {component}
rewrite reduced tail latency.

## Availability

The availability objective was 99.5 percent.
"""
            documents[f"{service}.slo-2025.md"] = stale

        # -- lexical trap: shares vocabulary, not the answer --------------- #
        trap_kind = TRAP_SUFFIXES[index % len(TRAP_SUFFIXES)]
        trap_p99 = p99 + rng.choice((200, 400))
        trap = _front_matter(
            {
                "version": "0.1",
                "updated": "2025-01-05",
                "owner": f"team-{component}",
                "status": "draft",
                **public,
            }
        ) + f"""# {service} {trap_kind} (draft, not approved)

This draft discusses p99 request latency for {service}, the error budget, and the
queue depth threshold. It is an unapproved working document and the numbers in
it are proposals, not commitments.

## Latency proposal

A p99 request latency of {trap_p99} milliseconds is proposed for {service}. This
proposal predates the {component} rewrite and was never adopted.

## Error budget proposal

An error budget of {error_budget} percent is discussed but not ratified.
"""
        documents[_slug(service, trap_kind)] = trap

        # -- cross-tenant document: the access-control material ---------- #
        # Only for the first tenant, so the leakage probe is not symmetric
        # across the whole corpus.
        if tenant == TENANTS[0]:
            restricted = _front_matter(
                {
                    "version": f"{major}.{minor}",
                    "updated": f"2026-0{1 + index % 9}-1{index % 9}",
                    "owner": f"team-{component}",
                    "status": "current",
                    "tenant": tenant,
                }
            ) + f"""# {service} capacity allocation (restricted)

Restricted to the {tenant} tenant. The figures in this document are commercially
sensitive and are not shared outside the tenant boundary.

## Allocation

The contracted capacity for {service} is {queue_depth} messages per second, with
a committed p99 latency of {p99} milliseconds. Overage is billed at a rate that
depends on the tenant agreement.

## Expansion

Capacity expansions require {tenant} sign-off and are provisioned within one
business day of approval.
"""
            documents[f"{service}.capacity-restricted.md"] = restricted

        truth[service] = {
            "tenant": tenant,
            "component": component,
            "p99": p99,
            "error_budget": error_budget,
            "queue_depth": queue_depth,
            "replica_lag": replica_lag,
            "page_threshold": page_threshold,
            "has_conflict": has_conflict,
            "trap_kind": trap_kind,
            "trap_p99": trap_p99,
            "restricted": tenant == TENANTS[0],
        }

    return documents, truth


def build_probes(truth: dict[str, dict], count: int) -> list[Probe]:
    """Construct one probe of each kind per service.

    `must_contain` always asserts a literal value that appears in the target
    document, and `must_retrieve` always names the document that holds it. So a
    probe can fail for exactly the reasons it exists -- wrong document, right
    document wrong span, stale document winning, forbidden document leaking --
    and never because the expected string is misspelled.
    """
    probes: list[Probe] = []
    services = list(truth)[:count]

    for service in services:
        facts = truth[service]
        slo = _slug(service, "slo")
        spec = _slug(service, "spec")
        runbook = _slug(service, "runbook")
        trap = _slug(service, facts["trap_kind"])

        # direct: plainly worded, one document, one number.
        probes.append(
            Probe(
                probe_id=f"direct.{service}.p99",
                query=f"What is the p99 request latency objective for {service}?",
                kind="direct",
                must_retrieve=(slo,),
                must_contain=(f"{facts['p99']} milliseconds",),
                tenant="public",
                rationale="The obvious case. A suite where this fails is broken, not difficult.",
                tags=("latency",),
            )
        )

        # paraphrase: same fact, no shared vocabulary with the source heading.
        probes.append(
            Probe(
                probe_id=f"paraphrase.{service}.budget",
                query=(
                    f"How many faulty responses can {service} return before reliability work "
                    f"takes priority over building new things?"
                ),
                kind="paraphrase",
                must_retrieve=(slo,),
                must_contain=(f"error budget for {service} is {facts['error_budget']} percent",),
                tenant="public",
                rationale=(
                    "A lexical retriever matching on 'error budget' should pass; one that "
                    "only matches the SLO heading should not. Isolates vocabulary dependence."
                ),
                tags=("error-budget",),
            )
        )

        # multi_hop: needs the spec for the number and the runbook to act on it.
        probes.append(
            Probe(
                probe_id=f"multi_hop.{service}.lag",
                query=(
                    f"At what replica lag does {service} become degraded, and what should the "
                    f"on-call do first?"
                ),
                kind="multi_hop",
                must_retrieve=(spec, runbook),
                must_contain=(f"{facts['replica_lag']} seconds",),
                tenant="public",
                rationale=(
                    "The lag threshold is stated in both documents; the action is only in the "
                    "runbook. A retriever that returns one good document should still fail."
                ),
                tags=("oncall", "multi-hop"),
            )
        )

        # lexical_trap: the draft shares vocabulary and must lose.
        probes.append(
            Probe(
                probe_id=f"lexical_trap.{service}",
                query=f"What p99 latency has been proposed for {service}?",
                kind="lexical_trap",
                must_retrieve=(trap,),
                must_contain=(f"p99 request latency of {facts['trap_p99']} milliseconds",),
                must_not=(slo,),
                must_outrank=(trap,),
                tenant="public",
                rationale=(
                    "The query is about the *proposal*, not the objective. Expressed as "
                    "precedence, not absence: returning the SLO at rank 2 is harmless, "
                    "returning it at rank 1 is the failure."
                ),
                tags=("status-awareness",),
            )
        )

        if facts["has_conflict"]:
            probes.append(
                Probe(
                    probe_id=f"version_conflict.{service}",
                    query=f"What is the agreed p99 latency objective for {service} right now?",
                    kind="version_conflict",
                    must_retrieve=(slo,),
                    must_contain=(f"is {facts['p99']} milliseconds",),
                    must_not=(f"{service}.slo-2025.md",),
                    must_outrank=(slo,),
                    tenant="public",
                    rationale=(
                        "A 2025 revision with a different, larger figure is in the corpus. Both "
                        "are topically perfect, so only status handling separates them."
                    ),
                    tags=("staleness",),
                )
            )

        # forbidden: exists, answerable, but the caller may not see it.
        if facts["restricted"]:
            probes.append(
                Probe(
                    probe_id=f"forbidden.{service}",
                    query=f"What is the contracted capacity for {service}?",
                    kind="forbidden",
                    must_retrieve=(),
                    must_contain=(),
                    must_not=(f"{service}.capacity-restricted.md",),
                    must_not_kind="authorization",
                    should_abstain=True,
                    tenant="globex",
                    subject="globex-analyst",
                    rationale=(
                        "The answer exists in a restricted document. The caller is a different "
                        "tenant, so surfacing it at all is the failure, regardless of ranking."
                    ),
                    tags=("acl", "tenancy"),
                )
            )

    # unanswerable: about services that are deliberately absent from the corpus.
    for absent in ("zephyr-dispatcher", "obsidian-reconciler", "tungsten-scheduler"):
        probes.append(
            Probe(
                probe_id=f"unanswerable.{absent}",
                query=f"What is the p99 latency objective for {absent}?",
                kind="unanswerable",
                should_abstain=True,
                rationale=(
                    f"{absent} is not in the corpus. Confidently answering from a "
                    "similarly-named service is the classic fabrication failure."
                ),
                tags=("abstention",),
            )
        )

    return probes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="evaluation/generated", help="output directory")
    parser.add_argument("--seed", type=int, default=20260101, help="RNG seed; fixes output byte for byte")
    parser.add_argument("--services", type=int, default=len(SERVICES), help="number of services to generate")
    parser.add_argument("--max-chars", type=int, default=1200, help="chunk size for the manifest")
    parser.add_argument("--clean", action="store_true", help="remove the output directory first")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    docs_dir = out / "docs"
    if args.clean and out.exists():
        shutil.rmtree(out)
    docs_dir.mkdir(parents=True, exist_ok=True)

    documents, truth = build_corpus(args.seed, args.services)
    for name, text in sorted(documents.items()):
        (docs_dir / name).write_text(text, encoding="utf-8")

    probes = build_probes(truth, args.services)
    # Compute the dataset id up front: `ProbeSet` is frozen by design, so a
    # content hash that can be derived from the payload must not be assigned
    # after construction.
    probe_set = ProbeSet(
        probes=tuple(probes),
        name="generated-enterprise",
        dataset_id=dataset_id_for([p.to_dict() for p in probes]),
        notes=(
            f"Generated by scripts/generate_corpus.py with seed={args.seed}, "
            f"services={args.services}. Ground truth is derived from the same values "
            "used to write the documents, so it cannot drift from the corpus."
        ),
    )

    problems = probe_set.validate(documents.keys())
    if problems:
        print(f"internal error: generated an incoherent probe set:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    probes_path = out / "probes.json"
    probe_set.save(probes_path)

    # Build the manifest through the *same* loader the evaluation path uses.
    # Chunking here independently of `load_corpus` was subtly wrong: the loader
    # strips front matter before splitting and the generator did not, so the two
    # produced different chunk boundaries and the manifest id recorded in the
    # probe set never matched the corpus it was scored against -- a permanent
    # warning that would have hidden a real one.
    corpus = load_corpus(docs_dir, source_type="true", max_chars=args.max_chars)
    manifest = attach_chunks(build_manifest(docs_dir), corpus.chunks)
    manifest_id = manifest.manifest_id
    probe_set = ProbeSet.load(probes_path)
    manifest_path = out / "manifest.json"
    manifest.save(manifest_path)

    # Re-save the probe set now that the corpus manifest id is known, so the two
    # files cross-reference each other and a reader can tell which corpus a set
    # of ground truth was written against.
    probe_set = ProbeSet(
        probes=probe_set.probes,
        name=probe_set.name,
        dataset_id=probe_set.dataset_id,
        corpus_manifest_id=manifest_id,
        notes=probe_set.notes,
    )
    probe_set.save(probes_path)

    print(f"corpus    {len(documents)} documents -> {docs_dir}")
    print(f"          manifest_id {manifest_id}  ({manifest.total_chunks} chunks)")
    print(f"probes    {len(probes)} probes -> {probes_path}")
    print(f"          dataset_id  {probe_set.dataset_id}")
    print(f"          kinds       {probe_set.kind_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
