"""Retrieval result caching, keyed by corpus identity.

The failure this is designed to prevent
---------------------------------------
A RAG cache is only safe if a cache hit cannot outlive the evidence it was built
from. Re-ingest a document, or roll out a chunker change, and every previously
cached result now describes a corpus that no longer exists -- while looking
completely normal. The system answers from stale evidence, confidently, with
citations to chunks that were deleted. No metric moves, because retrieval still
"works"; the answers are simply wrong.

So the key is not the query. The key is the query *and* the corpus manifest id,
and the manifest id is a content hash over the chunk identities of the index
actually serving traffic. Any change to any chunk, any document, or the chunking
parameters that determine chunk identity changes the manifest id, which
invalidates every entry. Staleness is structurally impossible rather than
handled by a TTL.

What is and is not cached
-------------------------
Only retrieval, not answers. An answer cache would also have to invalidate on
model, prompt and context-builder changes, and a wrong answer served from cache
is far more damaging than a slow one. Retrieval is deterministic given the
corpus, so caching it loses nothing but time.

Bounded by entry count rather than time: a burst of one-off queries is the
common case, and a TTL-bounded cache sized for steady traffic grows without
limit under a scan. Oldest entries are evicted first (FIFO), which is the
right policy here because every entry is equally likely to be re-requested
after a re-ingest -- and a re-ingest clears the whole cache anyway.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

from enterprise_rag.types import Chunk

__all__ = ["CacheStats", "ResultCache", "cache_key"]


def cache_key(query: str, scope: frozenset[str], manifest_id: str, final_k: int) -> str:
    """Stable key for one retrieval under one authorization scope.

    `scope` is part of the key, not a filter applied after lookup. Caching a
    cross-tenant result and filtering on the way out is the classic way a
    shared cache becomes a data-leak primitive; making the scope part of the
    identity means a hit for tenant A can never be returned to tenant B, even if
    the filtering code is removed later.
    """
    material = "\x1f".join(
        (manifest_id, str(final_k), query.strip().casefold(), *sorted(scope))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "entries": self.entries,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class ResultCache:
    """Bounded LRU over retrieval results, partition-aware.

    `manifest_id` is stored alongside the entries rather than merely mixed into
    the key, so that a corpus change can clear everything in one operation
    instead of waiting for entries to age out. The mix-in is what makes a stale
    hit impossible; the partition is what makes the invalidation instant.
    """

    max_entries: int = 256
    enabled: bool = True
    _manifest_id: str = ""
    _entries: OrderedDict[str, tuple[Chunk, ...]] = field(default_factory=OrderedDict)
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        if self.max_entries < 0:
            raise ValueError("max_entries must be >= 0")

    @property
    def manifest_id(self) -> str:
        return self._manifest_id

    def bind(self, manifest_id: str) -> None:
        """Point the cache at a corpus version, discarding anything older.

        Called once per index build. When the manifest changes, every cached
        result describes a corpus that no longer exists, so all of them go.
        """
        if manifest_id == self._manifest_id:
            return
        if self._entries:
            self.stats.invalidations += len(self._entries)
        self._entries.clear()
        self._manifest_id = manifest_id

    def get(self, key: str) -> tuple[Chunk, ...] | None:
        if not self.enabled or not self._entries:
            return None
        found = self._entries.get(key)
        if found is None:
            self.stats.misses += 1
            return None
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return found

    def put(self, key: str, chunks: Sequence[Chunk]) -> None:
        if not self.enabled or self.max_entries == 0:
            return
        self._entries[key] = tuple(chunks)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.stats.evictions += 1
        self.stats.entries = len(self._entries)

    def clear(self) -> None:
        self.stats.invalidations += len(self._entries)
        self._entries.clear()
        self._manifest_id = ""
        self.stats.entries = 0

    def to_dict(self) -> dict:
        return {"manifest_id": self._manifest_id, "max_entries": self.max_entries, **self.stats.to_dict()}
