"""Tokenisation, stopwords and light stemming.

One implementation shared by BM25, the lexical reranker and the query
transformer. Duplicating tokenisation across those three is how lexical
retrieval quietly rots: the index is built with one analyser and queried with
another, and recall drops for reasons that look like a ranking problem.

The stemmer is a deliberately small suffix-stripping Porter-style step, not full
Porter. It handles the inflections that matter for a technical corpus (plurals,
-ing, -ed, -s, -es) and nothing else. Aggressive stemming in a domain full of
identifiers is a liability: `k8s` and `s` collapse, `bus` and `bu` collapse, and
exact-match recall on flags gets worse, not better.
"""

from __future__ import annotations

import itertools
import re
from functools import lru_cache

__all__ = ["STOPWORDS", "analyze", "bigrams", "normalize", "stem", "tokenize"]

# Deliberately small. A large stopword list hurts a technical corpus: "no",
# "not" and "can" carry meaning in troubleshooting text, and "all"/"only" are
# meaningful in "only when the metrics server is available".
STOPWORDS = frozenset(
    """
    a an and are as at be been being by for from has have how i in is it its of on or
    that the their then there these this to was were what when where which who why will
    with you your
    """.split()
)

# Tokens that must survive stopword removal even though they look like filler.
_KEEP_ALWAYS = frozenset({"not", "no", "can", "should", "do", "does", "if", "all", "only"})

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace. Preserves identifiers like `kubectl`."""
    return _WS_RE.sub(" ", text.strip().lower())


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Split normalised text into tokens.

    The `[._-/]` alternation keeps compound identifiers intact
    (`sidecar-containers`, `enterprise_rag.kubernetes.io`, `kubectl_logs`, `rollout/restart`)
    instead of fragmenting them into pieces that match nothing.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(normalize(text))
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS or t in _KEEP_ALWAYS]


@lru_cache(maxsize=8192)
def stem(token: str) -> str:
    """Strip common English inflections. Idempotent on already-stemmed tokens."""
    if len(token) <= 3 or token.isdigit():
        return token

    # Possessives before plural: "pods'" -> "pods" -> "pod"
    if token.endswith("'s") or token.endswith("s'"):
        token = token[:-2]
    elif token.endswith("'"):
        token = token[:-1]

    if len(token) <= 3:
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("ss"):
        return token
    if token.endswith("s") and not token.endswith("us") and len(token) > 3:
        token = token[:-1]

    if len(token) <= 4:
        return token
    if token.endswith("ing") and len(token) > 5:
        base = token[:-3]
        # "monitoring" -> "monitor", but "sting" -> "sting" (no real base)
        return base if (len(base) >= 3 and base[-1] not in "aeiou") or base.endswith("at") else token
    if token.endswith("ed") and len(token) > 4:
        base = token[:-2]
        if len(base) >= 3 and base[-1] not in "aeiou":
            return base
        if base.endswith(("at", "iz", "us", "iv")):
            return base + "e"

    return token


def analyze(text: str, *, stem_tokens: bool = True, keep_stopwords: bool = False) -> list[str]:
    """Full analysis chain: normalize -> tokenize -> (optionally) stem."""
    tokens = tokenize(text, keep_stopwords=keep_stopwords)
    return [stem(t) for t in tokens] if stem_tokens else tokens


def bigrams(tokens: list[str]) -> list[str]:
    """Adjacent token pairs, for phrase-ish matching and query features."""
    return [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
