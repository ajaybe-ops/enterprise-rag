"""Dense retrieval backends.

Two implementations, deliberately different in kind:

`LSARetriever`
    A classical dense baseline: TF-IDF over the term space, then truncated SVD
    to project documents and queries into a shared low-dimensional vector space
    (Latent Semantic Analysis). Fully reproducible from the corpus alone, needs
    no network, no model download and no API key — which is what makes the
    benchmark suite runnable in CI. It is a real dense retriever, not a stub,
    but it is a weak one: LSA captures co-occurrence structure, not semantics.

`JinaDenseRetriever`
    The production path: `jina-embeddings-v3` over the API, task-aware so
    passages and queries are embedded differently.

Having both is the point. It lets the benchmark answer "does the neural
embedding actually beat a classical dense baseline on *this* corpus?" — a
question that is usually asserted rather than measured, and the answer here is
not obvious in advance.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.config import Settings
from app.text import analyze
from app.types import Chunk

__all__ = [
    "LSARetriever",
    "JinaDenseRetriever",
    "Embedder",
    "cosine",
    "embeddings_mode",
]


# --------------------------------------------------------------------------- #
# Local dense: TF-IDF + truncated SVD (LSA)
# --------------------------------------------------------------------------- #


@dataclass
class LSARetriever:
    """TF-IDF + truncated SVD dense retriever.

    Why SVD and not, say, random projection: random projection preserves
    distances but not the term-weighting structure, so it is a weaker baseline
    than it looks. SVD gives the optimal rank-`d` approximation of the TF-IDF
    matrix under Frobenius loss, so the projection is doing the most it can.

    L2-normalised SVD components approximate cosine similarity in the projected
    space, so vectors are normalised at query time and the dot product is the
    score.
    """

    chunks: tuple[Chunk, ...]
    n_components: int = 160
    _matrix: np.ndarray | None = None
    _vocab: dict[str, int] | None = None
    _idf: np.ndarray | None = None
    _doc_vectors: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
        n_docs = len(self.chunks)
        if n_docs == 0:
            self._vocab, self._idf = {}, np.zeros(0)
            self._matrix = np.zeros((0, 0))
            self._doc_vectors = np.zeros((0, 0))
            return

        doc_tokens = [analyze(chunk.text) for chunk in self.chunks]
        vocab: dict[str, int] = {}
        for tokens in doc_tokens:
            for token in tokens:
                vocab.setdefault(token, len(vocab))
        self._vocab = vocab

        n_terms = len(vocab)
        counts = np.zeros((n_docs, n_terms), dtype=np.float32)
        for row, tokens in enumerate(doc_tokens):
            for token, tf in Counter(tokens).items():
                counts[row, vocab[token]] = tf

        # Sublinear TF damping, then L2 row norm. Standard, and it stops a
        # single repeated term from dominating a document's vector.
        counts = np.log1p(counts)
        row_norms = np.linalg.norm(counts, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        counts /= row_norms

        df = (counts > 0).sum(axis=0).astype(np.float32)
        self._idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        tfidf = counts * self._idf

        # n_components is capped at min(n_docs, n_terms): SVD cannot produce
        # more components than the smaller dimension, and asking for more
        # raises instead of degrading.
        k = max(1, min(self.n_components, n_docs, n_terms))
        # full_matrices=False is essential: it returns the k economy-size
        # factors instead of the full (n_docs x n_docs) matrix.
        _, _, vt = np.linalg.svd(tfidf, full_matrices=False)
        projection = vt[:k].T  # (n_terms, k)

        self._matrix = projection
        self._doc_vectors = self._normalize(tfidf @ projection)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def _vectorize(self, text: str) -> np.ndarray:
        if self._matrix is None or self._matrix.size == 0:
            return np.zeros((0,), dtype=np.float32)
        counts = np.zeros((len(self._vocab),), dtype=np.float32)
        for token, tf in Counter(analyze(text)).items():
            index = self._vocab.get(token)
            if index is not None:
                counts[index] = tf
        counts = np.log1p(counts)
        norm = np.linalg.norm(counts)
        if norm:
            counts /= norm
        return (counts * self._idf) @ self._matrix

    def scores(self, query: str) -> np.ndarray:
        if self._doc_vectors is None or self._doc_vectors.size == 0:
            return np.zeros(0, dtype=np.float32)
        query_vector = self._vectorize(query)
        if query_vector.size == 0:
            return np.zeros(len(self.chunks), dtype=np.float32)
        norm = np.linalg.norm(query_vector)
        query_vector = query_vector / norm if norm else query_vector
        return self._doc_vectors @ query_vector

    def search(self, query: str, top_k: int) -> list[Chunk]:
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            self.chunks[int(i)].with_scores(dense_score=float(scores[int(i)]), rank=rank)
            for rank, i in enumerate(order)
        ]


# --------------------------------------------------------------------------- #
# Remote dense: Jina
# --------------------------------------------------------------------------- #


class JinaDenseRetriever:
    """`jina-embeddings-v3` over the API, with retry and batching."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = None

    @property
    def dim(self) -> int:
        return self.settings.jina_embed_dim

    def _post(self, path: str, payload: dict) -> dict:
        import requests

        if self._session is None:
            self._session = requests.Session()
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self._session.post(
                    f"{self.settings.jina_base_url}{path}",
                    headers={
                        "Authorization": f"Bearer {self.settings.jina_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.settings.jina_timeout_s,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                text = str(exc).lower()
                retryable = any(t in text for t in ("429", "rate", "quota", "timeout", "503", "502"))
                if not retryable or attempt == 3:
                    break
                time.sleep(2**attempt * 0.5)
        raise RuntimeError(f"Jina {path} failed after retries: {last_error}") from last_error

    def embed(self, texts: Sequence[str], task: str) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.settings.jina_embed_model,
            "task": task,
            "dimensions": self.settings.jina_embed_dim,
            "input": list(texts),
        }
        data = self._post("/embeddings", payload)["data"]
        # The API does not guarantee ordering; sorting by index is mandatory,
        # not defensive — a shuffled response silently pairs every chunk with
        # the wrong vector and the index looks plausible.
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], task="retrieval.query")[0]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch = max(1, self.settings.jina_embed_batch)
        for start in range(0, len(texts), batch):
            vectors.extend(self.embed(texts[start : start + batch], task="retrieval.passage"))
        return vectors


# --------------------------------------------------------------------------- #
# Embedder: one object owns the vector space for a whole run
# --------------------------------------------------------------------------- #


@dataclass
class Embedder:
    """Turns text into vectors in *one* consistent space.

    This exists as an object rather than free functions for a concrete reason:
    an LSA query vector is only meaningful against the same fitted SVD basis
    that produced the document vectors. Free functions that rebuilt the index per
    call would put every query in a different basis, and the resulting scores
    would look plausible while being meaningless. The space is stateful, so its
    owner must be too.
    """

    settings: Settings
    lsa: LSARetriever | None = None
    fitted_dim: int = 0
    _jina: JinaDenseRetriever | None = None

    @property
    def mode(self) -> str:
        return "jina" if self.settings.embeddings_enabled else "lsa"

    @property
    def dim(self) -> int:
        return self.settings.jina_embed_dim if self.mode == "jina" else self.fitted_dim

    def fit(self, chunks: Sequence[Chunk]) -> "Embedder":
        """Fit the local space to a corpus. No-op for the remote backend."""
        if self.mode == "lsa":
            self.lsa = LSARetriever(chunks=tuple(chunks), n_components=self.settings.lsa_dim)
            if self.lsa._doc_vectors is not None and self.lsa._doc_vectors.size:
                self.fitted_dim = int(self.lsa._doc_vectors.shape[1])
        return self

    @property
    def _client(self) -> JinaDenseRetriever:
        if self._jina is None:
            self._jina = JinaDenseRetriever(self.settings)
        return self._jina

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if self.mode == "jina":
            return np.asarray(self._client.embed_passages(texts), dtype=np.float32)
        if self.lsa is None:
            raise RuntimeError("LSARetriever must be fit() on a corpus before use")
        matrix = np.asarray(
            [
                self.lsa._vectorize(text)
                for text in texts
            ],
            dtype=np.float32,
        )
        return self._normalize_rows(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        if self.mode == "jina":
            vector = np.asarray(self._client.embed_query(text), dtype=np.float32)
        else:
            if self.lsa is None:
                raise RuntimeError("LSARetriever must be fit() on a corpus before use")
            vector = np.asarray(self.lsa._vectorize(text), dtype=np.float32)
        norm = np.linalg.norm(vector)
        return (vector / norm).astype(np.float32) if norm else vector

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    def search(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[Chunk]:
        """Brute-force cosine over `chunks`.

        Identical candidate sets across scorers is what makes the benchmark a
        controlled comparison rather than a comparison of two different systems.
        """
        if not chunks:
            return []
        if self.mode == "jina":
            matrix = self._normalize_rows(
                np.asarray(
                    self._client.embed_passages([c.text for c in chunks]), dtype=np.float32
                )
            )
        else:
            if self.lsa is None:
                raise RuntimeError("LSARetriever must be fit() on a corpus before use")
            matrix = self._normalize_rows(
                np.asarray([self.lsa._vectorize(c.text) for c in chunks], dtype=np.float32)
            )
        if matrix.size == 0:
            return []
        scores = matrix @ self.embed_query(query)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            chunks[int(i)].with_scores(dense_score=float(scores[int(i)]), rank=rank)
            for rank, i in enumerate(order)
        ]


def embeddings_mode(settings: Settings) -> str:
    return "jina" if settings.embeddings_enabled else "lsa"


def cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

