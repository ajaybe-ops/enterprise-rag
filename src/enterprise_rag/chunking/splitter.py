"""Paragraph-aware chunker with a hard size cap and optional overlap.

One chunker, not two. The retrieval package could have shipped its own splitter,
but then the benchmark would chunk documents differently from the ingest path
and the measured numbers would describe a system nobody ships. Both paths call
`chunk_text`, so a change here moves both.

Why paragraph alignment: splitting mid-paragraph tends to cut a definition away
from its explanation, which hurts both embedding quality and reranking. The
chunker prefers paragraph boundaries and only hard-splits a paragraph that is
itself larger than the cap.

Why overlap, and why it defaults to 0: without it, a fact that straddles a chunk
boundary is unretrievable — neither half matches the query well, so the passage
that actually answers the question may not be in the top K at all. That is the
single most common cause of "the document clearly says this but the bot missed
it". Overlap costs storage and dilutes BM25 term frequency, so it is opt-in:
the legacy ingest path keeps the old behaviour, and the retrieval path turns it
on. A non-zero overlap is a *character* overlap, not a token or sentence
boundary, which is crude but predictable and cannot loop.
"""

from __future__ import annotations

from typing import List

from enterprise_rag.observability import tracing as logfire  # vendor-free tracing shim

__all__ = ["chunk_text"]


def _hard_split(paragraph: str, chunk_size: int) -> list[str]:
    """Split a paragraph that is larger than the cap.

    Tries to break on a sentence boundary first so the two halves are not
    guaranteed to end mid-word; falls back to a raw cut only when a single
    "sentence" is itself oversized (minified JSON, a base64 blob).
    """
    pieces: list[str] = []
    remaining = paragraph
    while len(remaining) > chunk_size:
        window = remaining[:chunk_size]
        cut = max(window.rfind(". "), window.rfind(".\n"), window.rfind("; "))
        # Only honour a break in the last third: a sentence boundary right at
        # the start of the window would produce a uselessly small chunk.
        if cut > chunk_size // 3:
            pieces.append(remaining[: cut + 1].strip())
            remaining = remaining[cut + 1 :].strip()
        else:
            pieces.append(window.strip())
            remaining = remaining[chunk_size:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 0) -> List[str]:
    """Split text into paragraph-aligned chunks no larger than `chunk_size`.

    `overlap` is the number of trailing characters of each chunk repeated at the
    start of the next one. Overlap is applied between *emitted* chunks, and is
    capped at half the chunk size: a larger overlap would not leave room for new
    content and the loop would stop making progress.
    """
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    overlap = min(overlap, chunk_size // 2)
    # The separator counts against the cap. Forgetting this is how a chunk ends
    # up two characters over the limit, gets caught by the safety trim below,
    # and silently loses those two characters.
    sep = "\n\n"
    sep_len = len(sep)

    with logfire.span("✂️ Text Chunking", text_length=len(text), chunk_size=chunk_size, overlap=overlap):
        if not text.strip():
            return []

        paragraphs = [p for p in text.split(sep) if p.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                if current.strip():
                    chunks.append(current.strip())
                    current = ""
                for piece in _hard_split(paragraph, max(1, chunk_size - overlap - sep_len)):
                    tail = current.strip()[-overlap:] if overlap else ""
                    if current.strip():
                        chunks.append(current.strip())
                        current = ""
                    head = tail + sep if tail else ""
                    current = head + piece + sep
                continue

            if len(current) + len(paragraph) < chunk_size:
                current += paragraph + sep
                continue

            # The paragraph does not fit alongside what is accumulated, so close
            # the current chunk. This append is load-bearing: without it the
            # accumulated text is overwritten and disappears from the corpus
            # entirely, with no error and no empty chunk to show for it.
            if current.strip():
                chunks.append(current.strip())

            tail = current.strip()[-overlap:] if overlap else ""
            head = tail + sep if tail else ""
            combined = head + paragraph
            if len(combined) <= chunk_size:
                current = combined + sep
            else:
                # The overlap does not fit alongside this paragraph. Give up the
                # overlap for this chunk rather than truncating the paragraph:
                # the overlapped text is *already* present in the chunk just
                # emitted, so dropping the overlap loses nothing, whereas
                # truncating here would delete content permanently and silently.
                current = paragraph + sep

        if current.strip():
            chunks.append(current.strip())

        valid = [chunk for chunk in chunks if chunk.strip()]
        # A hard cap is a contract the vector store relies on: an oversized
        # chunk is silently truncated by some embedders and rejected by others.
        oversized = [c for c in valid if len(c) > chunk_size]
        if oversized:
            logfire.warning(
                "Chunk exceeded chunk_size after overlap; trimming",
                max_len=max(len(c) for c in oversized),
            )
            valid = [c[:chunk_size] for c in valid]
        logfire.info(f"✅ Generated {len(valid)} chunks")
        return valid
