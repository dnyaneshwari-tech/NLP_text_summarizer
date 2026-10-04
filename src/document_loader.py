from __future__ import annotations

from pathlib import Path
from typing import Union

import fitz
from docx import Document

from config import MAX_UPLOAD_BYTES

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def validate_upload_file(file_path: str, filename: str | None = None) -> bool:
    """Return True when the file type is supported and within size limits."""
    path = Path(file_path)
    candidate_name = (filename or path.name).lower()
    extension = Path(candidate_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return False

    if path.suffix.lower() != extension:
        return False

    if path.exists() and path.stat().st_size > MAX_UPLOAD_BYTES:
        return False

    return True


def extract_txt_text(file_path: str) -> str:
    """Read UTF-8 text from a TXT file, falling back gracefully on decode issues."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def extract_pdf_text(file_path: str) -> str:
    """Extract selectable text from a PDF. If a PDF has no text, raise a clear error."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = fitz.open(path)
    chunks: list[str] = []
    for page in doc:
        page_text = page.get_text("text")
        if page_text and page_text.strip():
            chunks.append(page_text)

    doc.close()

    text = "\n\n".join(chunks).strip()
    if not text:
        raise ValueError(
            "No selectable text was found in this PDF. It may be a scanned image-based PDF, "
            "which requires OCR and is not included in this version."
        )
    return text


def extract_docx_text(file_path: str) -> str:
    """Extract paragraphs from a DOCX file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = Document(str(path))
    paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return "\n\n".join(paragraphs).strip()


def extract_document_text(file_path: str, filename: str | None = None) -> str:
    """Extract content from a supported document file."""
    candidate_name = (filename or Path(file_path).name).lower()
    extension = Path(candidate_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {candidate_name}. Supported types are TXT, PDF, and DOCX.")

    if Path(file_path).exists() and Path(file_path).stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"The uploaded file is larger than the allowed {MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB limit."
        )

    if extension == ".txt":
        return extract_txt_text(file_path)
    if extension == ".pdf":
        return extract_pdf_text(file_path)
    if extension == ".docx":
        return extract_docx_text(file_path)

    raise ValueError(f"Unsupported file type: {candidate_name}")


def load_uploaded_document(uploaded_file) -> tuple[str, str]:
    """Read an uploaded Streamlit file object into text and its original filename."""
    if uploaded_file is None:
        raise ValueError("No document was uploaded.")

    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"The uploaded file exceeds the {MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB size limit."
        )

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

    file_bytes = uploaded_file.getvalue()
    suffix_name = uploaded_file.name
    temp_path = Path("temp_upload")
    temp_path.mkdir(exist_ok=True)
    destination = temp_path / suffix_name
    destination.write_bytes(file_bytes)

    try:
        text = extract_document_text(str(destination), uploaded_file.name)
    finally:
        if destination.exists():
            destination.unlink()

    return text, uploaded_file.name
