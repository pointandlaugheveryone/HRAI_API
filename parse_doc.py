from io import BytesIO
import re

from pypdf import PdfReader


_SKILLS_SECTION_RE = re.compile(r"\b(dovednosti|schopnosti|skills)\b", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    # Collapse whitespace and keep a readable single-line string for downstream matching.
    return " ".join(text.split())


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    return _normalize_text(_extract_pdf_text(file_bytes))


def extract_text_from_pdf_skills_section(file_bytes: bytes) -> str:
    """
    Optimized PDF extractor that trims content to the skills section when present.
    Looks for headings: dovednosti | schopnosti | skills (case-insensitive).
    """
    raw_text = _extract_pdf_text(file_bytes)
    if not raw_text:
        return ""

    lines = raw_text.splitlines()
    for idx, line in enumerate(lines):
        if _SKILLS_SECTION_RE.search(line):
            # Return content after the heading line for tighter matching.
            return _normalize_text("\n".join(lines[idx + 1 :]))

    return _normalize_text(raw_text)


def extract_text(file_bytes: bytes, filename: str = '') -> str:
    fn = filename.lower()
    if fn.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    else:
        raise RuntimeError(
            "Unsupported file type. Supported format: pdf. Use /text"
        )