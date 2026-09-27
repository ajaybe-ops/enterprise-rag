# ADR 0005 — Metadata propagates to chunks; the defaults are fail-open on purpose

**Status:** accepted

Referenced from `app/retrieval/corpus.py`.

## Context

Access control needs to know which tenant owns a chunk. The obvious
implementation is to re-derive ownership at query time from the document path or
filename. That is wrong in a way that does not announce itself: a future code
path that filters chunks directly will not remember to reconstruct the
authorisation context, and the failure mode is a data leak rather than an error.

The obvious alternative — treat a missing `tenant` as invisible — is safe but
makes a plain directory of notes return nothing, and an empty result set is
extremely easy to mistake for "no relevant documents".

## Decision

1. Front matter is parsed once at load time and copied onto every chunk. A chunk
   carries its own `tenant` and `status`, so authorization never depends on
   re-deriving context.
2. `Principal.may_read` fails closed on every ambiguous input: no tenant on the
   document, no tenants on the principal, or `authenticated=False` all deny.
3. `DocumentRecord.tenant` defaults to `public` and `status` defaults to
   `current`. These two are deliberately fail-**open**, in the direction of
   serving.
4. Ingest writes both fields explicitly, and the evaluation corpus declares
   `tenant` on every document.

## Consequences

- Point 3 is the sharp edge. Forgetting `tenant:` on a restricted document
  publishes it. This is documented at the definition and pinned by tests so the
  trade-off cannot be changed silently.
- The argument for it: an empty default corpus is a worse failure mode than a
  wrong one, because silence looks like correct behaviour.
- `status: superseded` is the field that must be set to withhold a document. The
  asymmetry is deliberate — a stale document wrongly served is confusing, a
  current document wrongly withheld is a support ticket.
- A production deployment should add a corpus lint that fails any document
  missing `tenant` or `status`. That is the actual fix; the default exists only
  so the project is explorable without it.
