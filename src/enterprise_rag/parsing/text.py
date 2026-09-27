"""Plain-text loader."""

from pathlib import Path

from enterprise_rag.observability import tracing as logfire  # vendor-free tracing shim


def parse_text(file_path: str) -> str:
    """Read a text file, trying UTF-8 first and falling back to Latin-1."""
    with logfire.span("→ Text Parsing", filename=file_path):
        try:
            for encoding in ("utf-8", "latin-1"):
                try:
                    with Path(file_path).open(encoding=encoding) as handle:
                        text = handle.read()
                    logfire.info(f"Read {len(text)} characters ({encoding}).")
                    return text
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {file_path}")
        except Exception as exc:
            logfire.error(f"❌ Text parse failed: {exc}")
            raise
