"""Content-addressed corpus manifests.

See `manifest.py` for why a corpus has to be a versioned artifact before
regression detection can distinguish a pipeline change from a content change.
"""

from app.corpus.manifest import (
    MANIFEST_VERSION,
    ChangedDocument,
    CorpusManifest,
    DocumentRecord,
    ManifestDiff,
    attach_chunks,
    build_manifest,
    content_hash,
    normalise_text,
)

__all__ = [
    "MANIFEST_VERSION",
    "ChangedDocument",
    "CorpusManifest",
    "DocumentRecord",
    "ManifestDiff",
    "attach_chunks",
    "build_manifest",
    "content_hash",
    "normalise_text",
]
