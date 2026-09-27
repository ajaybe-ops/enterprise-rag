"""Configuration parsing, validation, and documentation-drift guards.

The drift tests matter more than they look. This project has twice shipped an
`.env.example` that documented variables the code never read -- once for six
names, including the corpus path. A template that names unset variables is
worse than no template: an operator sets `RAG_CORPUS_DIR`, sees the corpus they
configured ignored, and concludes the index is broken. These tests make that
class of bug fail CI instead.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from enterprise_rag.config.settings import (
    ConfigError,
    RetrievalConfig,
    Settings,
    get_settings,
    reset_settings,
    set_settings,
)

ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = ROOT / ".env.example"

# `_get_bool(src, "RAG_ADAPTIVE", True)` and friends. The first argument is the
# env mapping, the second is the variable name.
_READER_PREFIX = "_get_"
_READER_TYPES = {"_get_str", "_get_int", "_get_float", "_get_bool", "_get_list"}


def _read_keys_in(module: Path) -> set[str]:
    """Variable names passed to the `_get_*` readers in one module.

    Only the *second* positional argument is the variable name; later
    arguments are defaults, so walking every string argument would sweep up
    default values like `"evals/datasets/docs"` and report them as
    configuration.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in _READER_TYPES:
            continue
        if len(node.args) < 2:
            continue
        key = node.args[1]
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            found.add(key.value)
    return found


def _read_keys_via_os_environ(module: Path) -> set[str]:
    """Variable names read straight off ``os.environ``.

    `RAG_TRACE_EMIT` is read this way, from the tracing module, because a
    library must not perform import-time environment reads to configure its own
    output sink. So it will not appear in the `_get_*` set.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 1:
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Attribute) and func.attr == "get":
            name = "os.environ.get"
        elif isinstance(func, ast.Attribute) and func.attr == "getenv":
            name = "os.environ.getenv"
        if name and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str) and value.isupper():
                found.add(value)
    return found


def _env_example_keys() -> set[str]:
    """Variable names assigned in `.env.example` (comments ignored)."""
    keys: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Z][A-Z0-9_]*)=", stripped)
        if match:
            keys.add(match.group(1))
    return keys


def _all_read_keys() -> set[str]:
    package = ROOT / "src" / "enterprise_rag"
    keys: set[str] = set()
    keys |= _read_keys_in(package / "config" / "settings.py")
    keys |= _read_keys_via_os_environ(package / "observability" / "tracing.py")
    return keys


class TestEnvExampleDrift:
    def test_env_example_documents_exactly_the_read_keys(self) -> None:
        documented = _env_example_keys()
        read = _all_read_keys()

        undocumented = read - documented
        assert not undocumented, (
            "these variables are read by the code but absent from .env.example: "
            f"{sorted(undocumented)}"
        )

        phantom = documented - read
        assert not phantom, (
            "these variables appear in .env.example but nothing reads them: "
            f"{sorted(phantom)}"
        )

    def test_env_example_leaves_everything_unset(self) -> None:
        """An unedited template must configure nothing.

        The README claims the default configuration is an empty environment, so
        a value pre-filled in the template would silently falsify that claim.
        """
        filled = []
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                name, _, value = stripped.partition("=")
                if value.strip():
                    filled.append(name)
        assert not filled, f".env.example pre-fills values: {filled}"


class TestDefaults:
    def test_no_environment_means_offline_and_extractive(self) -> None:
        settings = Settings.from_env({})
        assert not settings.llm_enabled
        assert not settings.embeddings_enabled
        assert not settings.vector_store_enabled
        assert not settings.auth_enabled
        assert settings.retrieval.use_bm25
        assert settings.retrieval.use_dense

    def test_capabilities_report_the_offline_stack(self) -> None:
        caps = Settings.from_env({}).capabilities()
        assert caps["generation"] == "extractive"
        assert caps["embeddings"].startswith("lsa-")
        assert caps["vector_store"] == "in-memory"
        assert caps["auth"] == "disabled"

    def test_defaults_are_valid(self) -> None:
        Settings.from_env({}).validate()  # must not raise

    def test_chunk_settings_default_within_limits(self) -> None:
        settings = Settings.from_env({})
        assert 0 < settings.chunk_overlap < settings.max_chunk_chars


class TestProviderAvailability:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, False),
            ({"llm_api_key": "sk-real-value"}, True),
            ({"llm_api_key": "your-api-key-here"}, False),
            ({"llm_api_key": "<paste-key>"}, False),
            ({"llm_base_url": "http://localhost:11434/v1"}, True),
            ({"llm_base_url": "https://api.openai.com/v1/"}, False),
            (
                {
                    "llm_api_key": "your-api-key-here",
                    "llm_base_url": "http://localhost:11434/v1",
                },
                True,
            ),
        ],
    )
    def test_llm_enabled(self, kwargs: dict[str, object], expected: bool) -> None:
        """A key-less local endpoint counts as configured.

        Ollama, vLLM, llama.cpp and LM Studio all serve the OpenAI chat schema
        unauthenticated. Requiring a key made those deployments silently
        inoperative: `/health` advertised "extractive" and no call was ever
        attempted, so the configuration looked accepted while doing nothing.
        """
        assert Settings(**kwargs).llm_enabled is expected

    def test_embeddings_require_a_key(self) -> None:
        """Embeddings have no key-less local mode in this project.

        Unlike generation, there is no bundled local embedding server, so an
        unset key must leave the offline LSA leg in place.
        """
        assert not Settings().embeddings_enabled
        assert not Settings(embeddings_api_key="your-key-here").embeddings_enabled
        assert Settings(embeddings_api_key="jina-real").embeddings_enabled

    def test_rerank_needs_both_the_strategy_and_embeddings(self) -> None:
        base = Settings(retrieval=RetrievalConfig(rerank="jina"))
        assert not base.rerank_enabled
        assert Settings(
            retrieval=RetrievalConfig(rerank="jina"), embeddings_api_key="jina-real"
        ).rerank_enabled

    def test_vector_store_accepts_url_or_local_path_but_needs_either(self) -> None:
        assert not Settings().vector_store_enabled
        assert Settings(qdrant_local_path="./qdrant").vector_store_enabled
        assert not Settings(qdrant_url="your-qdrant-url-here").vector_store_enabled


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rate_limit_per_minute": -1},
            {"max_context_chars": 100},
            {"max_query_chars": 4},
            {"max_chunks": 0},
            {"llm_max_retries": -1},
            {"cache_max_entries": -1},
            {"embeddings_dim": 2},
            {"lsa_dim": 1},
            {"qdrant_url": "https://q.example", "qdrant_local_path": "./q"},
        ],
    )
    def test_rejects_nonsense(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ConfigError):
            Settings(**kwargs)

    def test_chunk_cap_above_final_k_is_a_warning_not_an_error(self) -> None:
        settings = Settings(max_chunks=50, retrieval=RetrievalConfig(final_k=5))
        assert any("MAX_CHUNKS" in w for w in settings.warnings)

    def test_malformed_scalar_raises_with_the_variable_name(self) -> None:
        with pytest.raises(ConfigError, match="RAG_FINAL_K"):
            Settings.from_env({"RAG_FINAL_K": "not-a-number"})

    def test_malformed_bool_raises_with_the_variable_name(self) -> None:
        with pytest.raises(ConfigError, match="RAG_ADAPTIVE"):
            Settings.from_env({"RAG_ADAPTIVE": "maybe"})

    @pytest.mark.parametrize("value", ["ture", "flase", "2", "y es"])
    def test_mistyped_boolean_is_rejected_not_silently_false(self, value: str) -> None:
        """A typo must not quietly disable a retrieval leg.

        This is the failure mode the strict parse exists to prevent:
        `RAG_USE_BM25=ture` used to read as `False`, changing retrieval quality
        with no warning anywhere.
        """
        with pytest.raises(ConfigError, match="RAG_USE_BM25"):
            Settings.from_env({"RAG_USE_BM25": value})

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "t", "y"])
    def test_accepted_true_spellings(self, value: str) -> None:
        assert Settings.from_env({"RAG_USE_BM25": value}).retrieval.use_bm25 is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "f", "n"])
    def test_accepted_false_spellings(self, value: str) -> None:
        assert Settings.from_env({"RAG_USE_BM25": value}).retrieval.use_bm25 is False

    def test_status_lists_parse_comma_separated(self) -> None:
        settings = Settings.from_env(
            {"RAG_STATUS_EXCLUDE": "old, retired", "RAG_STATUS_DEMOTE": "draft"}
        )
        assert settings.retrieval.status_exclude == ("old", "retired")
        assert settings.retrieval.status_demote == ("draft",)

    def test_explicitly_empty_status_exclude_excludes_nothing(self) -> None:
        """Empty must mean "no exclusions", not "fall back to the default".

        The alternative reading -- empty string means the default list -- makes
        it impossible to turn the policy off, which an operator disabling a
        filter entirely will reasonably expect to be able to do.
        """
        settings = Settings.from_env({"RAG_STATUS_EXCLUDE": ""})
        assert settings.retrieval.status_exclude == ()


class TestRedaction:
    @pytest.mark.parametrize(
        "field", ["llm_api_key", "embeddings_api_key", "qdrant_api_key", "api_key"]
    )
    def test_secrets_are_never_plaintext(self, field: str) -> None:
        settings = Settings.from_env({field: "super-secret-value"})
        dumped = repr(settings.redacted())
        assert "super-secret-value" not in dumped

    def test_absent_secrets_read_as_none(self) -> None:
        assert Settings.from_env({}).redacted()["llm_api_key"] is None

    def test_service_urls_survive_redaction(self) -> None:
        """A config dump that redacts URLs is useless for debugging.

        Base URLs carry no credential, so they are kept; only values whose field
        name suggests a secret are masked.
        """
        redacted = Settings.from_env(
            {"LLM_BASE_URL": "https://api.example-tenant.invalid/v1"}
        ).redacted()
        assert redacted["llm_base_url"] == "https://api.example-tenant.invalid/v1"


class TestSettingsLifecycle:
    def teardown_method(self) -> None:
        reset_settings()

    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()

    def test_set_settings_overrides_and_reset_restores(self) -> None:
        replacement = Settings.from_env({"RAG_FINAL_K": "9"})
        set_settings(replacement)
        assert get_settings().retrieval.final_k == 9

        reset_settings()
        assert get_settings() is not replacement

    def test_no_import_time_environment_reads(self) -> None:
        """Importing config must not capture the environment.

        Reading `os.environ` at import time makes configuration invisible to
        tests and to any caller that wants to supply its own mapping, and it
        makes `monkeypatch.setenv` silently ineffective.
        """
        import enterprise_rag.config.settings as module

        source = module.__file__
        assert source is not None
        text = Path(source).read_text(encoding="utf-8")
        # A module-level `os.environ[...]`/`get(...)` call, outside a function.
        assert not re.search(r"^_?\w*\s*=\s*os\.environ", text, re.MULTILINE)
        assert not re.search(r"^_?\w*\s*=\s*os\.getenv", text, re.MULTILINE)
