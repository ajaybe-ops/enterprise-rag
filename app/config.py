"""Immutable application configuration.

Design constraints, in priority order:

1. **No import-time environment reads.** The previous version read every
   variable as a class attribute at import, which made config frozen at first
   import, impossible to override in a test, and turned a malformed integer
   into a raw `ValueError` raised during import of an unrelated module.
   `Settings.from_env()` is now explicit and `Settings` is a frozen dataclass.

2. **Invalid config is a startup error, not a request error.** Bad integers and
   contradictory settings raise `ConfigError` with the offending key, at
   construction, before anything serves traffic.

3. **Placeholder detection is one function.** A `.env` copied from the template
   still contains `your-...` strings. Treating those as configured produces a
   system that claims capabilities it cannot deliver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Mapping

__all__ = [
    "ConfigError",
    "RetrievalConfig",
    "Settings",
    "get_settings",
    "reset_settings",
    "set_settings",
]

# Substrings that mean "the operator copied the template and never filled it in".
_PLACEHOLDER_MARKERS = ("<", "your-", "your_", "changeme", "placeholder", "xxx", "example.com")

# Field names whose *values* must never be logged. Matched case-insensitively
# against the whole name, because the previous version compared lowercase field
# names against uppercase suffixes and therefore never matched anything — a
# redaction helper that silently redacts nothing is worse than none, because
# callers trust it.
_SECRET_NAME_PARTS = (
    "_api_key",
    "_key",
    "_token",
    "_secret",
    "_password",
    "credential",
)

# Suffixes that mark a value as a connection string, which may embed a password
# (`postgresql://user:pw@host/db`). Matching on the suffix rather than on a
# substring keeps a generic service URL out of this set: an API base URL is
# routinely safe to log and excluding it keeps a config dump useful.
_SECRET_NAME_SUFFIXES = ("_dsn", "_db_url", "_database_url")

# Endpoints that are not credentials, so logging them keeps a config dump
# useful. These are checked before the suffix rules above: `jina_base_url` ends
# in neither, but a future endpoint name must not be able to opt out of
# redaction by accident.
_SECRET_NAME_EXCEPTIONS = frozenset(
    {
        "upstash_url",
        "qdrant_url",
        "jina_base_url",
        "logfire_base_url",
    }
)


def _is_secret_name(name: str) -> bool:
    if name.lower() in _SECRET_NAME_EXCEPTIONS:
        return False
    lowered = name.lower()
    if any(lowered.endswith(suffix) for suffix in _SECRET_NAME_SUFFIXES):
        return True
    return any(part in lowered for part in _SECRET_NAME_PARTS)


class ConfigError(ValueError):
    """Raised for malformed or contradictory configuration."""


def _placeholder(value: str | None) -> bool:
    if not value:
        return True
    lowered = value.strip().lower()
    return not lowered or any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def _get_str(env: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not an integer") from exc


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a number") from exc


def _get_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_list(env: Mapping[str, str], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Comma-separated env value as a tuple, falling back to `default`.

    An explicitly empty value yields an empty tuple rather than the default.
    `RAG_STATUS_EXCLUDE=` is a meaningful instruction -- "exclude nothing" -- and
    silently reverting it to the four retired statuses would make it impossible
    to disable status filtering at all.

    The values are lowercased because front matter is not normalised and
    `status: Superseded` would otherwise slip past `status: superseded`.
    """
    raw = env.get(key)
    if raw is None:
        return default
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Everything that controls *how* candidates are found and ordered.

    Kept separate from `Settings` because this is the surface the benchmark
    sweeps: candidate-pool size, fusion strategy and rerank depth are the three
    knobs worth measuring, and grouping them keeps the sweep honest.
    """

    # Which retrievers participate. Order is irrelevant; fusion is by strategy.
    use_bm25: bool = True
    use_dense: bool = True

    # BM25 (Robertson). k1 controls term-frequency saturation, b controls
    # length normalisation. Defaults are the values from the original Okapi BM25
    # work and are hard to beat without labelled data.
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Fusion. "rrf" is Reciprocal Rank Fusion (score-free, scale-free);
    # "weighted" is min-max normalised score fusion.
    fusion: str = "rrf"
    fusion_rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    # Candidate pool. `candidate_k` feeds each retriever; `final_k` is what the
    # reranker returns. candidate_k must exceed final_k or reranking cannot
    # reorder anything, only truncate.
    candidate_k: int = 30
    final_k: int = 5

    # Reranking. "none" | "jina" | "lexical_overlap" (offline, order-only).
    rerank: str = "jina"

    # Adaptive retrieval. Below the confidence threshold the pipeline escalates;
    # at or below `abstain_below` it refuses to answer.
    #
    # PROVISIONAL: 0.25 sits in the gap measured by
    # scripts/probe_confidence_separation.py (in-domain min 0.26, off-domain
    # max 0.25 on eight hand-picked questions). That gap is ~0.01, far too thin
    # to call validated. This value is the single source of truth for both the
    # pipeline and the retriever node; refit it against a labelled query set
    # before trusting it.
    adaptive: bool = True
    confidence_threshold: float = 0.45
    abstain_below: float = 0.25
    max_escalations: int = 1

    # Document-status awareness. `status_exclude` lists front-matter `status`
    # values whose content is known to be wrong for present-tense questions and
    # which are therefore dropped from the candidate pool; `status_demote` lists
    # values that are merely less authoritative and are scored down but kept
    # reachable. Both are configurable because which vocabularies a corpus uses
    # is a property of that corpus, and a hard-coded list is guaranteed to be
    # wrong at someone else's site. Comma-separated in the environment.
    status_exclude: tuple[str, ...] = ("superseded", "retired", "archived", "deprecated")
    status_demote: tuple[str, ...] = ("draft", "proposed", "experimental", "unapproved")
    status_demotion: float = 0.55

    # Non-fatal configuration problems found during construction. Declared as
    # fields because the dataclass uses slots=True: `object.__setattr__` in
    # `_collect` can only assign attributes that have a slot.
    warnings: tuple[str, ...] = ()
    _pending_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.fusion not in {"rrf", "weighted"}:
            raise ConfigError(f"fusion must be 'rrf' or 'weighted', got {self.fusion!r}")
        if self.rerank not in {"none", "jina", "lexical_overlap"}:
            raise ConfigError(f"rerank must be none/jina/lexical_overlap, got {self.rerank!r}")
        if self.candidate_k < self.final_k:
            raise ConfigError(
                f"candidate_k ({self.candidate_k}) must be >= final_k ({self.final_k}); "
                "otherwise reranking can only truncate, never reorder"
            )
        if not 0.0 <= self.abstain_below <= self.confidence_threshold <= 1.0:
            raise ConfigError(
                "expected 0.0 <= abstain_below <= confidence_threshold <= 1.0, got "
                f"abstain_below={self.abstain_below}, confidence_threshold={self.confidence_threshold}"
            )
        if self.bm25_k1 < 0:
            raise ConfigError("bm25_k1 must be >= 0")
        if not 0.0 <= self.bm25_b <= 1.0:
            raise ConfigError(f"bm25_b must be in [0, 1], got {self.bm25_b}")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ConfigError("fusion weights must be >= 0")
        if not self.use_bm25 and not self.use_dense:
            raise ConfigError("at least one of use_bm25/use_dense must be enabled")
        if self.fusion == "weighted" and self.dense_weight == self.sparse_weight == 0:
            raise ConfigError("weighted fusion with both weights 0 discards all evidence")
        if self.candidate_k == self.final_k:
            self._collect(
                "candidate_k == final_k: reranking can truncate but cannot reorder"
            )
        if not self.use_dense and self.rerank == "jina":
            # Jina rerank scores query/document pairs independently of the dense
            # retriever, so this is coherent rather than broken. But it usually
            # means the operator disabled dense retrieval and forgot to change
            # the reranker, so it is surfaced rather than swallowed.
            self._collect(
                "rerank='jina' with use_dense=False: the cross-encoder will rerank "
                "sparse-only candidates, which usually indicates a misconfiguration"
            )

    def _collect(self, message: str) -> None:
        """Record a non-fatal configuration problem.

        `__post_init__` runs on a frozen dataclass, so the list is stashed on a
        private attribute and frozen into the public `warnings` tuple. Warnings
        are not raised because a suboptimal-but-running configuration is more
        common than a broken one, and refusing to start turns a warning into an
        outage.
        """
        # Bypass the frozen guard deliberately: this happens exactly once,
        # during construction, before the object is shared.
        existing = list(getattr(self, "_pending_warnings", ()))
        existing.append(message)
        object.__setattr__(self, "_pending_warnings", existing)
        object.__setattr__(self, "warnings", tuple(existing))


@dataclass(frozen=True, slots=True)
class Settings:
    """Fully-resolved application configuration."""

    # --- Corpus ------------------------------------------------------------
    # Relative to the repository root. Only the in-scope directory is loaded:
    # the off-domain set is a distractor for evaluation, not a serving source.
    corpus_dir: str = "DATA/true_data"
    max_chunk_chars: int = 1200
    chunk_overlap: int = 120

    # --- LLM ---------------------------------------------------------------
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_fallback_model: str = "gpt-4o"
    llm_max_retries: int = 2
    llm_timeout_s: float = 60.0

    # --- Gateway -----------------------------------------------------------
    portkey_api_key: str | None = None
    portkey_primary_slug: str = "marathon-api"
    portkey_fallback_slug: str = "anthropic-fallback"
    portkey_config_id: str | None = None

    # --- Embeddings / rerank ----------------------------------------------
    jina_api_key: str | None = None
    jina_base_url: str = "https://api.jina.ai/v1"
    jina_embed_model: str = "jina-embeddings-v3"
    jina_rerank_model: str = "jina-reranker-v2-base-multilingual"
    jina_embed_dim: int = 1024
    jina_embed_batch: int = 64
    jina_timeout_s: float = 30.0
    lsa_dim: int = 160
    # Dimension of the deterministic hash embedding used when no embedding API is
    # configured. It is a *fallback for continuity*, not semantic retrieval:
    # hashing words into a fixed vector preserves no relationships beyond
    # co-occurrence in the hashed stream, so the health check and every
    # benchmark label it as "hash" rather than "dense".
    hash_embed_dim: int = 256
    # Local sentence-transformers backend, used when Jina is unavailable. The
    # dimension is declared here rather than read from the model because it is
    # needed *before* the model loads: it sizes the Qdrant collection. Guessing
    # it (or reusing the hash dimension) creates a collection that then rejects
    # every upsert, because the vectors arrive at the model's real size.
    fallback_embed_model: str = "all-MiniLM-L6-v2"
    fallback_embed_dim: int = 384

    # --- Vector store ------------------------------------------------------
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "enterprise_rag"
    # Local (serverless) Qdrant path. This is what makes the benchmark suite
    # runnable with zero credentials: `qdrant local` needs no cluster, no API
    # key and no network.
    qdrant_local_path: str | None = None

    # --- Catalog (was: registry) -----------------------------------------
    catalog_db_url: str | None = None
    # LangGraph checkpoint store. Deliberately a *separate* field from
    # `catalog_db_url` even though both usually point at the same Postgres: one
    # holds the document catalog, the other holds conversation checkpoints. They
    # have different schemas, different retention and different blast radii, and
    # collapsing them into one variable is how a "clear the conversation
    # history" migration ends up deleting the document index. Falls back to the
    # catalog URL when unset, so a single-database deployment still works.
    checkpoint_db_url: str | None = None

    # --- Cache / rate limit ------------------------------------------------
    upstash_url: str | None = None
    upstash_token: str | None = None
    cache_ttl_s: int = 3600
    rate_limit_per_minute: int = 20
    # Trust X-Forwarded-For only when a proxy sits in front. Off by default:
    # the header is client-controlled, so trusting it hands the attacker the
    # rate-limit key.
    trust_forwarded_for: bool = False

    # --- Security ----------------------------------------------------------
    api_key: str | None = None
    max_query_chars: int = 4000
    max_context_chars: int = 24000
    request_timeout_s: float = 120.0

    # --- Client ------------------------------------------------------------
    # Where the UI and the evaluation harnesses send requests. It is a *client*
    # concern, not a server one, so it lives in the same object purely to keep a
    # single env contract; the server never reads it.
    backend_url: str = "http://127.0.0.1:8000"

    # --- Observability -----------------------------------------------------
    logfire_token: str | None = None
    logfire_base_url: str | None = None
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "enterprise-rag"
    # LLM-as-judge credentials for evaluation. Kept off the production key and
    # off the Portkey gateway on purpose: eval traffic should not exhaust the
    # serving budget, and routing it through a gateway makes eval latency and
    # cost non-comparable between runs.
    judge_api_key: str | None = None
    judge_model: str = "gpt-4o"
    # USD per 1M tokens, used only for *estimates*. Published list prices; see
    # docs/08_COST_MODEL.md for provenance and the caveat that they drift.
    cost_per_mtok_input: float = 0.15
    cost_per_mtok_output: float = 0.60

    # --- Retrieval ---------------------------------------------------------
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Non-fatal configuration problems, populated by `validate()`. See
    # RetrievalConfig.warnings for the rationale.
    warnings: tuple[str, ...] = ()
    _pending_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Validate on *every* construction path, not just from_env(). Tests,
        # the CLI and `with_overrides()` all build Settings directly, and
        # validation that only runs on one of those paths is validation that
        # quietly does not apply to the configuration under test.
        self.validate()

    # --- Derived capability flags -----------------------------------------
    @property
    def gateway_enabled(self) -> bool:
        return not _placeholder(self.portkey_api_key)

    @property
    def llm_enabled(self) -> bool:
        return self.gateway_enabled or not _placeholder(self.openai_api_key)

    @property
    def embeddings_enabled(self) -> bool:
        return not _placeholder(self.jina_api_key)

    @property
    def rerank_enabled(self) -> bool:
        return self.retrieval.rerank == "jina" and self.embeddings_enabled

    @property
    def vector_store_enabled(self) -> bool:
        return not _placeholder(self.qdrant_url) or bool(self.qdrant_local_path)

    @property
    def catalog_enabled(self) -> bool:
        return not _placeholder(self.catalog_db_url)

    @property
    def checkpoint_enabled(self) -> bool:
        return not _placeholder(self.checkpoint_db_url) or self.catalog_enabled

    @property
    def cache_enabled(self) -> bool:
        return not _placeholder(self.upstash_url) and not _placeholder(self.upstash_token)

    @property
    def auth_enabled(self) -> bool:
        return not _placeholder(self.api_key)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from `env` (defaults to `os.environ`)."""
        src = os.environ if env is None else env

        retrieval = RetrievalConfig(
            use_bm25=_get_bool(src, "RAG_USE_BM25", True),
            use_dense=_get_bool(src, "RAG_USE_DENSE", True),
            bm25_k1=_get_float(src, "RAG_BM25_K1", 1.5),
            bm25_b=_get_float(src, "RAG_BM25_B", 0.75),
            fusion=_get_str(src, "RAG_FUSION", "rrf") or "rrf",
            fusion_rrf_k=_get_int(src, "RAG_FUSION_RRF_K", 60),
            dense_weight=_get_float(src, "RAG_DENSE_WEIGHT", 1.0),
            sparse_weight=_get_float(src, "RAG_SPARSE_WEIGHT", 1.0),
            candidate_k=_get_int(src, "RAG_CANDIDATE_K", 30),
            final_k=_get_int(src, "RAG_FINAL_K", 5),
            rerank=_get_str(src, "RAG_RERANK", "jina") or "jina",
            adaptive=_get_bool(src, "RAG_ADAPTIVE", True),
            confidence_threshold=_get_float(src, "RAG_CONFIDENCE_THRESHOLD", 0.45),
            abstain_below=_get_float(src, "RAG_ABSTAIN_BELOW", 0.25),
            max_escalations=_get_int(src, "RAG_MAX_ESCALATIONS", 1),
            # Status policy is configurable because "which lifecycle states are
            # servable" is a deployment decision, not a code change. It was
            # previously hardcoded with no env override, so the documented
            # behaviour and the configurable behaviour were not the same thing.
            status_exclude=_get_list(
                src,
                "RAG_STATUS_EXCLUDE",
                ("superseded", "retired", "archived", "deprecated"),
            ),
            status_demote=_get_list(
                src,
                "RAG_STATUS_DEMOTE",
                ("draft", "proposed", "experimental", "unapproved"),
            ),
        )

        settings = cls(
            corpus_dir=_get_str(src, "RAG_CORPUS_DIR", "DATA/true_data"),
            max_chunk_chars=_get_int(src, "RAG_MAX_CHUNK_CHARS", 1200),
            chunk_overlap=_get_int(src, "RAG_CHUNK_OVERLAP", 120),
            openai_api_key=_get_str(src, "OPENAI_API_KEY"),
            openai_model=_get_str(src, "OPENAI_MODEL", "gpt-4o-mini"),
            openai_fallback_model=_get_str(src, "OPENAI_FALLBACK_MODEL", "gpt-4o"),
            llm_max_retries=_get_int(src, "LLM_MAX_RETRIES", 2),
            llm_timeout_s=_get_float(src, "LLM_TIMEOUT_S", 60.0),
            portkey_api_key=_get_str(src, "PORTKEY_API_KEY"),
            portkey_primary_slug=_get_str(src, "PORTKEY_PRIMARY_SLUG", "marathon-api"),
            portkey_fallback_slug=_get_str(src, "PORTKEY_FALLBACK_SLUG", "anthropic-fallback"),
            portkey_config_id=_get_str(src, "PORTKEY_PRIMARY_CONFIG_ID"),
            jina_api_key=_get_str(src, "JINA_API_KEY"),
            jina_base_url=_get_str(src, "JINA_BASE_URL", "https://api.jina.ai/v1"),
            jina_embed_model=_get_str(src, "JINA_EMBED_MODEL", "jina-embeddings-v3"),
            jina_rerank_model=_get_str(src, "JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual"),
            jina_embed_dim=_get_int(src, "JINA_EMBED_DIM", 1024),
            jina_embed_batch=_get_int(src, "JINA_EMBED_BATCH", 64),
            jina_timeout_s=_get_float(src, "JINA_TIMEOUT_S", 30.0),
            lsa_dim=_get_int(src, "RAG_LSA_DIM", 160),
            hash_embed_dim=_get_int(src, "FALLBACK_EMBED_DIM", 256),
            fallback_embed_model=_get_str(src, "FALLBACK_EMBED_MODEL", "all-MiniLM-L6-v2"),
            fallback_embed_dim=_get_int(src, "FALLBACK_EMBED_MODEL_DIM", 384),
            qdrant_url=_get_str(src, "QDRANT_CLUSTER_ENDPOINT"),
            qdrant_api_key=_get_str(src, "QDRANT_API_KEY"),
            qdrant_collection=_get_str(src, "QDRANT_COLLECTION", "enterprise_rag"),
            qdrant_local_path=_get_str(src, "QDRANT_LOCAL_PATH"),
            catalog_db_url=_get_str(src, "CATALOG_DB_URL") or _get_str(src, "NEON_DB_URL"),
            checkpoint_db_url=_get_str(src, "CHECKPOINT_DB_URL") or _get_str(src, "NEON_DB_URL"),
            upstash_url=(_get_str(src, "UPSTASH_REDIS_REST_URL") or "").rstrip("/") or None,
            upstash_token=_get_str(src, "UPSTASH_REDIS_REST_TOKEN"),
            cache_ttl_s=_get_int(src, "CACHE_TTL_SECONDS", 3600),
            rate_limit_per_minute=_get_int(src, "RATE_LIMIT_PER_MINUTE", 20),
            trust_forwarded_for=_get_bool(src, "TRUST_FORWARDED_FOR", False),
            api_key=_get_str(src, "RAG_API_KEY"),
            max_query_chars=_get_int(src, "MAX_QUERY_CHARS", 4000),
            max_context_chars=_get_int(src, "MAX_CONTEXT_CHARS", 24000),
            request_timeout_s=_get_float(src, "REQUEST_TIMEOUT_S", 120.0),
            backend_url=(_get_str(src, "BACKEND_URL", "http://127.0.0.1:8000") or "").rstrip("/"),
            logfire_token=_get_str(src, "LOGFIRE_TOKEN"),
            logfire_base_url=_get_str(src, "LOGFIRE_BASE_URL"),
            langsmith_tracing=_get_bool(src, "LANGSMITH_TRACING", False),
            langsmith_api_key=_get_str(src, "LANGSMITH_API_KEY"),
            langsmith_project=_get_str(src, "LANGSMITH_PROJECT", "enterprise-rag"),
            judge_api_key=_get_str(src, "JUDGE_OPENAI_API_KEY"),
            judge_model=_get_str(src, "JUDGE_OPENAI_MODEL", "gpt-4o"),
            cost_per_mtok_input=_get_float(src, "COST_PER_MTOK_INPUT", 0.15),
            cost_per_mtok_output=_get_float(src, "COST_PER_MTOK_OUTPUT", 0.60),
            retrieval=retrieval,
        )
        return settings

    def validate(self) -> None:
        """Cross-field checks that per-field parsing cannot catch."""
        if self.rate_limit_per_minute < 0:
            raise ConfigError("RATE_LIMIT_PER_MINUTE must be >= 0")
        if self.max_context_chars < 500:
            raise ConfigError("MAX_CONTEXT_CHARS must be >= 500 to hold any real context")
        if self.max_query_chars < 16:
            raise ConfigError("MAX_QUERY_CHARS must be >= 16 to hold a real question")
        if self.qdrant_url and self.qdrant_local_path:
            raise ConfigError(
                "set either QDRANT_CLUSTER_ENDPOINT or QDRANT_LOCAL_PATH, not both"
            )
        if self.trust_forwarded_for and not self.vector_store_enabled:
            # Rate limiting keys on the client IP, not on the vector store, so the
            # relevant counterpart is the shared counter, not Qdrant.
            if not self.cache_enabled:
                self._collect(
                    "TRUST_FORWARDED_FOR is set but Upstash is not configured: the "
                    "rate limiter will trust a client-controlled header with no shared "
                    "counter, so the limit is trivially bypassed"
                )
        if self.llm_max_retries < 0:
            raise ConfigError("LLM_MAX_RETRIES must be >= 0")
        if self.hash_embed_dim < 8:
            raise ConfigError("FALLBACK_EMBED_DIM must be >= 8 to be non-degenerate")
        if self.fallback_embed_dim < 8:
            raise ConfigError("FALLBACK_EMBED_MODEL_DIM must be >= 8 to be non-degenerate")

    def _collect(self, message: str) -> None:
        """Record a non-fatal configuration problem (see RetrievalConfig._collect)."""
        existing = list(getattr(self, "_pending_warnings", ()))
        existing.append(message)
        object.__setattr__(self, "_pending_warnings", existing)
        object.__setattr__(self, "warnings", tuple(existing))

    def with_overrides(self, **kwargs) -> "Settings":
        """Return a copy with fields replaced. Used by tests and the CLI."""
        return replace(self, **kwargs)

    def capabilities(self) -> dict[str, str]:
        """Human-readable capability map. Reports what is *available*, not what is set."""
        if self.gateway_enabled:
            llm = "portkey"
        elif not _placeholder(self.openai_api_key):
            llm = "openai"
        else:
            llm = "missing"
        if self.embeddings_enabled:
            embeddings = "jina"
        else:
            embeddings = "lsa"
        if self.qdrant_local_path:
            store = f"qdrant-local({self.qdrant_local_path})"
        elif self.vector_store_enabled:
            store = "qdrant-cloud"
        else:
            store = "in-memory"
        return {
            "llm": llm,
            "embeddings": embeddings,
            "vector_store": store,
            "catalog": "postgres" if self.catalog_enabled else "in-memory",
            "cache": "upstash" if self.cache_enabled else "in-memory",
            "auth": "enabled" if self.auth_enabled else "disabled",
        }

    def redacted(self) -> dict[str, str | int | float | bool | None]:
        """Config as a dict safe to log: secrets become `<set,len=N>` or None.

        The length is retained on purpose. "Is the key present?" is the question
        an operator actually has, and answering it does not require disclosing
        the key. `None` stays `None` so absence is still visible.
        """
        out: dict[str, str | int | float | bool | None] = {}
        for name in self.__slots__:
            value = getattr(self, name)
            if isinstance(value, RetrievalConfig):
                out[name] = repr(value)
            elif _is_secret_name(name):
                out[name] = f"<set,len={len(str(value))}>" if value else None
            else:
                out[name] = value
        return out


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings, built once."""
    global _settings
    if _settings is None:
        from dotenv import load_dotenv

        load_dotenv()
        _settings = Settings.from_env()
    return _settings


def set_settings(settings: Settings) -> None:
    """Install settings explicitly. The supported test/CLI injection point."""
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Drop the cached settings so the next `get_settings()` re-reads the env."""
    global _settings
    _settings = None
