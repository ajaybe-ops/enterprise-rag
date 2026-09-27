True corpus
===========

Files in this directory are the authoritative enterprise documentation: the
Kubernetes, Databricks and infrastructure documents the assistant is expected to
answer from.

Files in `../noisy_data/` are unrelated third-party material — academic papers
and tutorials on unrelated subjects. They are ingested into the same collection
but tagged with `source_type="noisy"`, and the query-time payload filter in
`app/services/retrieval/qdrant_service.py` excludes them. That makes it possible
to measure the retrieval quality difference by flipping the filter.
