"""Structured tracing with no vendor dependency.

Why this exists instead of a tracing SDK
---------------------------------------
The previous implementation wrapped every stage in an OpenTelemetry-backed SDK
and shipped a commercial exporter as a hard dependency. That bought one thing:
spans forwarded to a hosted dashboard.

It cost more than it bought:

* A hard dependency on a vendor and an account, for a reference implementation
  that has to run on a laptop with `pip install -r requirements.txt` and nothing
  else. Anyone cloning this to read it had to sign up before they could run the
  tests.
* Exporter network calls on the request path, which is a latency cost and a
  failure mode, in exchange for observability data that is only useful
  *after* something has already gone wrong.
* A `LogfireNotConfiguredWarning` on every single invocation of a module that
  has nothing to do with tracing -- a warning that trains people to ignore
  warnings, which is worse than having no warning.

So tracing here is a plain in-process span recorder: no network, no
configuration, no account, and it produces the same information in a form that
can be printed, asserted on in a test, or shipped to whatever collector the
deployment actually uses. `drain()` is the seam for that -- point it at OTLP, a
file, or stdout and nothing above this module changes.

What it is not
--------------
Not sampled, not aggregated across processes, and not durable. It is a
per-request trace held in memory, which is exactly what the API needs in order
to explain a single response. Durable tracing is infrastructure, and
infrastructure belongs in the deployment, not in the library.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Trace", "configure", "drain", "install", "log", "span"]

# Emitted when true. Off by default: writing a JSON line to stderr on every
# request in a test run is noise, and noise in a test run is how a real warning
# stops being read.
_EMIT = os.environ.get("RAG_TRACE_EMIT", "").lower() in {"1", "true", "yes"}


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    fields: dict = field(default_factory=dict)
    parent: str | None = None
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000.0

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent": self.parent,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 3),
            "fields": self.fields,
        }


class Trace:
    """A per-thread span stack. Thread-local because a FastAPI worker handles
    concurrent requests on one event loop, and a shared stack would interleave
    two users' traces into nonsense -- a tracing system that cannot attribute a
    span to a request is worse than no tracing, because it is confidently wrong.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    @property
    def spans(self) -> list[Span]:
        if not hasattr(self._local, "spans"):
            self._local.spans = []
        # `self._local` is untyped, so the attribute read is `Any`. Copied into a
        # new list rather than returned directly: this is a live list that
        # callers iterate while spans are still being pushed, and handing out the
        # internal object means a caller mutating the result mutates the tracer.
        return list(self._local.spans)

    def reset(self) -> None:
        self._local.spans = []

    def snapshot(self) -> list[Span]:
        return list(self.spans)

    def by_name(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]


_TRACE = Trace()


def install() -> None:
    """Reset the current thread's trace. Call at the start of a request."""
    _TRACE.reset()


def drain() -> list[dict]:
    """Take and clear the recorded spans.

    Named `drain` rather than `get` because it consumes. A collector that can
    only be read accumulates spans forever, and a long-running process then leaks
    memory proportional to request count -- the kind of bug that only appears
    under production load.
    """
    spans = [s.to_dict() for s in _TRACE.snapshot()]
    _TRACE.reset()
    return spans


@contextlib.contextmanager
def span(name: str, **fields: Any) -> Iterator[Span]:
    """Time a block and record it. Never raises, never blocks.

    `fields` are captured eagerly at entry, so a caller that passes a mutable
    object sees the value as of the start of the stage. That is intentional:
    recording it at exit would capture post-mutation state, and a trace that
    disagrees with the code that produced it is actively harmful during an
    incident.
    """
    recorded = Span(name=name, start=time.perf_counter(), fields=dict(fields))
    stack = _TRACE.spans
    recorded.parent = stack[-1].span_id if stack else None
    stack.append(recorded)
    try:
        yield recorded
    finally:
        recorded.end = time.perf_counter()
        if _EMIT:
            sys.stderr.write(json.dumps(recorded.to_dict(), default=str) + "\n")


def log(level: str, message: str, **fields: Any) -> dict:
    """Emit a structured event into the current trace.

    A no-op unless `RAG_TRACE_EMIT` is set. Library code that writes to stdout
    unconditionally corrupts the output of every tool that imports it, so
    logging is opt-in and structured.
    """
    event = {"level": level, "message": message, "fields": dict(fields)}
    if _EMIT:
        sys.stderr.write(json.dumps(event, default=str) + "\n")
    return event


def info(message: str, **fields: Any) -> dict:
    return log("info", message, **fields)


def warning(message: str, **fields: Any) -> dict:
    return log("warning", message, **fields)


def error(message: str, **fields: Any) -> dict:
    return log("error", message, **fields)


def configure(*_args: Any, **_kwargs: Any) -> None:
    """No-op kept so call sites that configured the previous SDK still import.

    Present deliberately: a no-op named `configure` is the cheapest possible
    migration, and removing it would mean touching every call site for no
    behavioural gain.
    """


def as_logfire() -> Any:
    """Return this module under the previous SDK's name.

    A transitional shim so that a module can be migrated in one step rather than
    in two, and so that `logfire.span(...)` call sites keep working while the
    import is being replaced. Every such site is marked with a TODO in review;
    the point of the shim is to make the remaining work greppable, not permanent.
    """
    return sys.modules[__name__]
