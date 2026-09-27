"""HTML text extraction with BeautifulSoup."""

from pathlib import Path

from bs4 import BeautifulSoup

from enterprise_rag.observability import tracing as logfire  # vendor-free tracing shim


def parse_html(file_path: str) -> str:
    """Strip scripts/styles/metadata and return readable text."""
    with logfire.span("📄 HTML Parsing", filename=file_path):
        try:
            with Path(file_path).open(encoding="utf-8", errors="ignore") as handle:
                content = handle.read()

            soup = BeautifulSoup(content, "html.parser")

            for junk in soup(["script", "style", "meta", "noscript", "nav", "footer"]):
                junk.decompose()

            text = soup.get_text(separator="\n")

            lines = (line.strip() for line in text.splitlines())
            phrases = (phrase.strip() for line in lines for phrase in line.split("  "))
            return "\n".join(chunk for chunk in phrases if chunk)

        except Exception as exc:
            logfire.error(f"❌ HTML parse failed: {exc}")
            raise
