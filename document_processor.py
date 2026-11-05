# document_processor.py — Handles PDF, DOCX, and TXT document parsing and chunking

import io
import re
from typing import List, Dict, Any

import PyPDF2
import docx
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF file provided as raw bytes."""
    text_parts: List[str] = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF: {exc}") from exc
    return "\n\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract plain text from a DOCX file provided as raw bytes."""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        return "\n\n".join(paragraphs)
    except Exception as exc:
        raise ValueError(f"Failed to parse DOCX: {exc}") from exc


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode a plain-text file, trying UTF-8 then latin-1 as fallback."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot decode text file with supported encodings.")


# ---------------------------------------------------------------------------
# Public entry-point: process a single uploaded file
# ---------------------------------------------------------------------------

def process_uploaded_file(file_name: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Dispatch to the correct extractor based on file extension, then
    chunk the text for vector storage.

    Returns a dict with:
      - ``filename``: original file name
      - ``raw_text``: full extracted text
      - ``chunks``: list of text chunk strings ready for embedding
      - ``metadata``: list of per-chunk metadata dicts
    """
    extension = file_name.rsplit(".", 1)[-1].lower()

    if extension == "pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    elif extension == "docx":
        raw_text = extract_text_from_docx(file_bytes)
    elif extension == "txt":
        raw_text = extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{extension}")

    raw_text = _clean_text(raw_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)

    metadata = [
        {
            "source": file_name,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    return {
        "filename": file_name,
        "raw_text": raw_text,
        "chunks": chunks,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise whitespace and remove non-printable characters."""
    # Remove null bytes and control characters (keep newlines/tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", " ", text)
    # Collapse runs of blank lines to two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def get_document_summary(raw_text: str, max_chars: int = 500) -> str:
    """Return a short preview of the extracted text for display purposes."""
    preview = raw_text[:max_chars].strip()
    if len(raw_text) > max_chars:
        preview += "…"
    return preview
