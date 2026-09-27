"""Local PDF text extraction (pypdf, with a pdfplumber fallback)."""

from app import observability as logfire  # vendor-free tracing shim
from pypdf import PdfReader


def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF locally.

    pypdf first; any page it returns blank for (image-only, unusual encoding)
    is retried with pdfplumber. No external OCR service is involved.
    """
    with logfire.span("PDF Parsing (local)", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"PDF has {total_pages} pages.")

            text_parts: list[str] = []
            blank_pages: list[int] = []

            for index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    text_parts.append(text)
                else:
                    blank_pages.append(index + 1)

            if blank_pages:
                logfire.info(
                    f"pypdf returned blank on pages {blank_pages} — retrying with pdfplumber."
                )
                try:
                    import pdfplumber

                    with pdfplumber.open(file_path) as pdf:
                        for page_number in blank_pages:
                            fallback_text = pdf.pages[page_number - 1].extract_text() or ""
                            if fallback_text.strip():
                                text_parts.append(fallback_text)
                except Exception as plumber_err:
                    logfire.warning(f"pdfplumber fallback failed: {plumber_err}")

            full_text = "\n".join(text_parts)

            if not full_text.strip():
                logfire.warning(f"No text extracted from {file_path} — likely fully image-based.")
            else:
                logfire.info(f"Extracted {len(full_text)} characters from {file_path}.")

            return full_text

        except Exception as exc:
            logfire.error(f"PDF parse failed for {file_path}: {exc}")
            raise
