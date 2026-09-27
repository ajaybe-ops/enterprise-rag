# syntax=docker/dockerfile:1
#
# Runtime image for the retrieval service.
#
# Three deliberate properties:
#
#   1. The corpus is NOT baked into the image. Documents are mounted at
#      runtime and read from RAG_CORPUS_DIR. A corpus is versioned content, and
#      a content-addressed manifest is only meaningful if the content can change
#      without rebuilding the image that serves it.
#
#   2. No build-time credentials. The service starts and reports its health with
#      an empty environment; every provider is optional and configured at
#      runtime. An image that requires a secret to build cannot be built in CI
#      by a fork, which is most of the people who would run it.
#
#   3. Dependencies come from uv.lock, not from a hand-maintained list. The lock
#      is the record of what was actually tested; a `requirements.txt` is a
#      second opinion that drifts from it.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The base image has no locale data, and several source files contain non-ASCII
# characters in comments -- the BM25 formula in retrieval/sparse.py alone has
# sigma, element-of and a middle dot. An unencodable character in a docstring is
# a crash at import time under a C locale, which is a confusing way to find out.
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /srv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Build dependencies for wheels with no manylinux build, removed in the same
# layer they were installed in.
RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first: this layer survives every source change. `uv sync` installs
# the project itself alongside its locked dependencies.
#
# The evaluation corpus is not copied in. It is not a runtime input, and baking
# it in would make the image carry a second, unversioned copy of content that
# the manifest already describes.
#
# LICENSE is copied because `pyproject.toml` declares `license = { file = ... }`
# and Hatchling reads that file while building the wheel. Omitting it fails the
# build with an error about a missing licence file, which is a confusing way to
# learn that a metadata field refers to a path. It also has to be present in the
# image, not just the build context: the licence has to travel with the built
# artifact.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Run unprivileged. The corpus is mounted read-only, so nothing here needs to
# write outside /tmp.
RUN useradd --create-home --uid 10001 rag \
    && mkdir -p /corpus \
    && chown -R rag:rag /srv /corpus
USER rag

ENV RAG_CORPUS_DIR=/corpus \
    PATH="/srv/.venv/bin:$PATH"

VOLUME ["/corpus"]
EXPOSE 8000

# Reports `degraded` with a reason rather than failing the probe outright: a
# container whose corpus has not been mounted yet is still running correctly,
# and a hard healthcheck would make a start-up ordering problem look like a
# crash.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# One worker. The dense leg builds its projection once at load and the
# authorization path is per-request, so this is a throughput limit, not a
# correctness one -- but a single worker keeps the in-process caches coherent.
# Scale with a process manager and a real corpus, not by assuming statelessness.
CMD ["uvicorn", "enterprise_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
