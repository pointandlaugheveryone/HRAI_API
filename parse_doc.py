from io import BytesIO

from pypdf import PdfReader
from docx import Document as DocxDocument
from odfdo import Document as OdfDocument


extensions = {'.pdf', '.docx', '.odt'}
class UnsupportedFileTypeError(Exception):
    pass


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return '\n'.join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(BytesIO(file_bytes))
    return '\n'.join(para.text for para in doc.paragraphs if para.text)


def extract_text_from_odt(file_bytes: bytes) -> str:
    doc = OdfDocument(BytesIO(file_bytes))
    body = doc.body
    return body.get_formatted_text() if body else ''


def extract_text(filename: str, file_bytes: bytes) -> str:
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    elif filename_lower.endswith('.odt'):
        return extract_text_from_odt(file_bytes)

    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type. Supported formats: pdf, docx, odt. Use /text"
        )
