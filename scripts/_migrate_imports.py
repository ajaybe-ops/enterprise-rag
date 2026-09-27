"""One-off: rewrite intra-package imports after the app/ -> src/enterprise_rag/ move.

Kept in the repository because the mapping it encodes is the authoritative record
of the module layout. If a module is moved again, this table is what has to be
updated, and having it in code means the move is reviewable rather than a
sequence of ad-hoc search-and-replace operations.
"""

from __future__ import annotations

import pathlib
import re

# Applied in order. The catch-all ``app.`` -> ``enterprise_rag.`` rule must come
# LAST: when it ran first it consumed every more specific pattern, which is how a
# dozen files ended up importing a package that does not export what they wanted.
SPECIFIC: tuple[tuple[str, str], ...] = (
    (r"from app\.config import", "from enterprise_rag.config.settings import"),
    (r"from app\.corpus\.manifest import", "from enterprise_rag.indexing.manifest import"),
    (r"from app\.corpus import", "from enterprise_rag.indexing.manifest import"),
    (r"from app\.retrieval\.corpus import", "from enterprise_rag.indexing.corpus import"),
    (r"from app\.answering import", "from enterprise_rag.generation.extractive import"),
    (
        r"from enterprise_rag.observability import tracing as logfire",
        "from enterprise_rag.observability import tracing as logfire",
    ),
    (
        r"from app\.ingestion\.chunking\.splitter import",
        "from enterprise_rag.chunking.splitter import",
    ),
    (r"from app\.ingestion\.loaders\.", "from enterprise_rag.parsing."),
    (r"from app\.ingestion\.ingest import", "from enterprise_rag.ingestion.pipeline import"),
    (r"from app\.retrieval\.bm25 import", "from enterprise_rag.retrieval.sparse import"),
    (r"from app\.retrieval\.query import", "from enterprise_rag.retrieval.analysis import"),
    (r"from app\.retrieval\.rerank import", "from enterprise_rag.reranking.rerank import"),
    (r"from app\.retrieval\.pipeline import", "from enterprise_rag.pipelines.retrieval import"),
    (r"from app\.llm import", "from enterprise_rag.generation.provider import"),
    (r"from app\.services\.security import", "from enterprise_rag.api.security import"),
    (r"from app\.main import", "from enterprise_rag.api.app import"),
    (r"from app\.main\b", "from enterprise_rag.api.app"),
    (r"from enterprise_rag import", "from enterprise_rag import"),
    (r"\bapp\.(config|corpus|retrieval|ingestion|evaluation|security|services|types|text|llm|answering|main|observability|generation|injection|metrics)\b",
     r"enterprise_rag.\1"),
    # Second pass. The over-eager catch-all above rewrote these files into
    # half-correct paths before the specific rules got a chance to run: the
    # top-level package was renamed but the subpackage path was not. These
    # rules repair the half-correct forms.
    (r"from enterprise_rag\.config import", "from enterprise_rag.config.settings import"),
    (
        r"from enterprise_rag\.corpus\.manifest import",
        "from enterprise_rag.indexing.manifest import",
    ),
    (
        r"from enterprise_rag\.retrieval\.corpus import",
        "from enterprise_rag.indexing.corpus import",
    ),
    (
        r"from enterprise_rag\.answering import",
        "from enterprise_rag.generation.extractive import",
    ),
    (
        r"from enterprise_rag.observability import tracing as logfire",
        "from enterprise_rag.observability import tracing as logfire",
    ),
    (r"\benterprise_rag\.corpus\.manifest\b", "enterprise_rag.indexing.manifest"),
    (r"\benterprise_rag\.retrieval\.corpus\b", "enterprise_rag.indexing.corpus"),
    (r"\benterprise_rag\.ingestion\.chunking\.splitter\b", "enterprise_rag.chunking.splitter"),
    (r"\benterprise_rag\.ingestion\.loaders\.", "enterprise_rag.parsing."),
    (r"\benterprise_rag\.ingestion\.ingest\b", "enterprise_rag.ingestion.pipeline"),
    (r"\benterprise_rag\.retrieval\.bm25\b", "enterprise_rag.retrieval.sparse"),
    (r"\benterprise_rag\.retrieval\.query\b", "enterprise_rag.retrieval.analysis"),
    (r"\benterprise_rag\.retrieval\.rerank\b", "enterprise_rag.reranking.rerank"),
    (r"\benterprise_rag\.retrieval\.pipeline\b", "enterprise_rag.pipelines.retrieval"),
    (r"\benterprise_rag\.llm\b", "enterprise_rag.generation.provider"),
    (r"\benterprise_rag\.answering\b", "enterprise_rag.generation.extractive"),
    (r"\benterprise_rag\.services\.security\b", "enterprise_rag.api.security"),
    (r"\benterprise_rag\.main\b", "enterprise_rag.api.app"),
    # Third pass: the catch-all cannot tell a package reference from a
    # FastAPI application instance. ``app.get``/``app.post``/``app.state`` are
    # attribute lookups on the ASGI app, not imports, so they must be restored.
    # Listed explicitly rather than with a general rule: a general rule here
    # would be indistinguishable from the import rewrite it has to undo.
    (
        r"\benterprise_rag\.(get|post|put|delete|patch|head|options|websocket"
        r"|state|exception_handler|dependency_overrides|on_event|middleware"
        r"|include_router|router)\b",
        r"app.\1",
    ),
)

ROOTS = ("src", "scripts", "tests", "examples")


def rewrite(path: pathlib.Path) -> bool:
    original = path.read_text(encoding="utf-8")
    text = original
    for pattern, replacement in SPECIFIC:
        text = re.sub(pattern, replacement, text)
    if text != original:
        path.write_text(text, encoding="utf-8", newline="\n")
        return True
    return False


def main() -> int:
    changed = 0
    for root in ROOTS:
        base = pathlib.Path(root)
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if rewrite(path):
                changed += 1
                print(f"rewrote {path}")
    print(f"{changed} file(s) rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
