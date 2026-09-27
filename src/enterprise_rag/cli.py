"""Command-line entry point.

The web API is the product; this is the operator's view of the same pipeline.
Everything here runs the real code path -- nothing is reimplemented for the
terminal, so a number printed by `enterprise-rag` is the number the server and
the evaluation harness produce.

    enterprise-rag capabilities      # what is actually enabled here
    enterprise-rag index  --path D   # build an index, print the manifest id
    enterprise-rag ask "..." --tenant acme
    enterprise-rag inspect CACHE_KEY

`capabilities` exists because the most common support question about a
retrieval system is "is it actually using the thing you think it is?". It
answers from the resolved settings rather than from the environment, so
placeholder keys are reported as absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from enterprise_rag.config.settings import ConfigError, Settings
from enterprise_rag.security.access import Principal

__all__ = ["build_parser", "main"]


def _resolve_settings(args: argparse.Namespace) -> Settings:
    """Settings for the CLI, honouring `--corpus` without mutating the process."""
    settings = Settings.from_env()
    if getattr(args, "path", None):
        settings = settings.with_overrides(corpus_dir=str(args.path))
    return settings


def _cmd_capabilities(settings: Settings) -> int:
    payload = {
        "manifest_target": settings.corpus_dir,
        "capabilities": settings.capabilities(),
        "retrieval": {
            "bm25": settings.retrieval.use_bm25,
            "dense": settings.retrieval.use_dense,
            "fusion": settings.retrieval.fusion,
            "candidate_k": settings.retrieval.candidate_k,
            "final_k": settings.retrieval.final_k,
            "rerank": settings.retrieval.rerank,
            "adaptive": settings.retrieval.adaptive,
            "status_exclude": list(settings.retrieval.status_exclude),
            "status_demote": list(settings.retrieval.status_demote),
        },
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    for warning in settings.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _cmd_index(settings: Settings, args: argparse.Namespace) -> int:
    # Imported here so `--help` stays fast and does not require the numeric
    # stack: an operator checking their version should not pay for numpy.
    from enterprise_rag.api.deps import build_pipeline

    path = Path(settings.corpus_dir)
    if not path.is_dir():
        print(f"error: corpus directory not found: {path}", file=sys.stderr)
        return 2

    pipeline, degradations = build_pipeline(settings, path)
    if not args.json:
        print(f"corpus      {path}")
        print(f"manifest    {pipeline.manifest_id}")
        print(f"chunks      {pipeline.corpus_chunks}")
    else:
        json.dump(
            {
                "corpus": str(path),
                "manifest_id": pipeline.manifest_id,
                "chunks": pipeline.corpus_chunks,
                "degradations": list(degradations),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")

    for degradation in degradations:
        print(f"degraded: {degradation}", file=sys.stderr)
    return 0


def _cmd_ask(settings: Settings, args: argparse.Namespace) -> int:
    from enterprise_rag.api.deps import build_pipeline

    path = Path(settings.corpus_dir)
    if not path.is_dir():
        print(f"error: corpus directory not found: {path}", file=sys.stderr)
        return 2

    principal = Principal(subject=args.subject, tenants=frozenset(args.tenant))
    pipeline, _ = build_pipeline(settings, path)
    result = pipeline.query(args.question, principal)

    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0 if result.answer.status == "answered" else 1

    print(result.answer.text)
    if result.citations and result.citations.citations:
        print("\nevidence:", file=sys.stderr)
        for citation in result.citations.citations:
            print(f"  [{citation.ordinal}] {citation.source}", file=sys.stderr)
    print(
        f"\n({result.answer.status}, confidence {result.answer.confidence:.2f}, "
        f"{result.answer.status_detail})",
        file=sys.stderr,
    )
    return 0 if result.answer.status == "answered" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enterprise-rag",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        dest="path",
        metavar="DIR",
        help="corpus directory; overrides RAG_CORPUS_DIR for this invocation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("capabilities", help="print the resolved configuration and capabilities")

    index = sub.add_parser("index", help="build the index and print the manifest id")
    index.add_argument("--json", action="store_true", help="machine-readable output")

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question", help="the question to answer")
    ask.add_argument(
        "--tenant",
        action="append",
        default=["public"],
        help="tenant the caller may read; repeatable (default: public)",
    )
    ask.add_argument(
        "--subject",
        default="cli",
        help="caller identity, recorded in the trace (default: cli)",
    )
    ask.add_argument("--json", action="store_true", help="machine-readable output")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = _resolve_settings(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "capabilities":
        return _cmd_capabilities(settings)
    if args.command == "index":
        return _cmd_index(settings, args)
    if args.command == "ask":
        return _cmd_ask(settings, args)
    return 2  # unreachable: argparse enforces the subcommand


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
