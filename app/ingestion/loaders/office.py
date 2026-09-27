"""Office document parsing (.docx / .pptx) — local, no external service."""

from app import observability as logfire  # vendor-free tracing shim


def parse_office(file_path: str) -> str:
    """Extract text from Word or PowerPoint files.

    python-docx / python-pptx are used directly instead of a heavyweight
    partitioning library: these formats are structured, so walking paragraphs
    and slide shapes is both faster and easier to reason about.
    """
    with logfire.span("📄 Office Document Parsing", filename=file_path):
        try:
            lowered = file_path.lower()
            if lowered.endswith(".docx"):
                text = _parse_docx(file_path)
            elif lowered.endswith(".pptx"):
                text = _parse_pptx(file_path)
            else:
                raise ValueError(f"Unsupported office format: {file_path}")

            if not text.strip():
                logfire.warning(f"⚠️ No text extracted from {file_path}")
            else:
                logfire.info(f"✅ Parsed {len(text)} characters")
            return text

        except Exception as exc:
            logfire.error(f"❌ Office parse failed: {exc}")
            raise


def _parse_docx(file_path: str) -> str:
    from docx import Document

    document = Document(file_path)
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _parse_pptx(file_path: str) -> str:
    from pptx import Presentation

    presentation = Presentation(file_path)
    parts: list[str] = []
    for index, slide in enumerate(presentation.slides, start=1):
        slide_parts = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                slide_parts.append(shape.text_frame.text.strip())
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        slide_parts.append(" | ".join(cells))
        if slide_parts:
            parts.append(f"Slide {index}\n" + "\n".join(slide_parts))
    return "\n\n".join(parts)
