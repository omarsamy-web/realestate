"""Unified PDF extraction service (Phase 2).

Merges the PyMuPDF page-aware extractor from the research project with the
existing PyPDF loader pipeline.  Both code paths are available:

- ``extract_pdf(file_bytes, filename)`` → rich ``ExtractedDocument`` with
  per-page text and page numbers (used by the Research Lab).
- ``load_pdfs_from_bytes(file_bytes, filename)`` → list of LangChain
  ``Document`` chunks (used when feeding uploaded PDFs into the RAG index).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:
    try:
        import fitz  # PyMuPDF legacy import
    except ImportError:
        fitz = None


# ---------------------------------------------------------------------------
# Rich extraction (Research Lab)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedDocument:
    """Result of a page-aware PDF extraction."""
    filename: str
    page_count: int
    pages: list[dict]

    @property
    def text(self) -> str:
        return "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in self.pages)


class PDFExtractionError(ValueError):
    """Raised when an uploaded file is not a usable PDF."""


def extract_pdf(file_bytes: bytes, filename: str = "uploaded.pdf") -> ExtractedDocument:
    """Extract page-numbered text from a PDF using PyMuPDF.

    Falls back to PyPDF if PyMuPDF is not installed.
    """
    if not file_bytes:
        raise PDFExtractionError("The uploaded file is empty.")
    if not filename.lower().endswith(".pdf"):
        raise PDFExtractionError("Please upload a PDF file.")

    if fitz is not None:
        return _extract_with_pymupdf(file_bytes, filename)
    return _extract_with_pypdf(file_bytes, filename)


def _extract_with_pymupdf(file_bytes: bytes, filename: str) -> ExtractedDocument:
    if not file_bytes.startswith(b"%PDF-"):
        raise PDFExtractionError("This file does not have a valid PDF header.")
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            if pdf.is_encrypted:
                raise PDFExtractionError("This PDF is password-protected.")
            pages = [
                {"page": i + 1, "text": page.get_text("text").strip()}
                for i, page in enumerate(pdf)
            ]
    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError("The PDF could not be opened.") from exc

    if not pages:
        raise PDFExtractionError("The PDF contains no pages.")
    if not any(p["text"] for p in pages):
        raise PDFExtractionError(
            "No selectable text was found. This may be a scanned PDF; "
            "OCR support is not included."
        )
    return ExtractedDocument(filename=filename, page_count=len(pages), pages=pages)


def _extract_with_pypdf(file_bytes: bytes, filename: str) -> ExtractedDocument:
    """Fallback using pypdf when PyMuPDF is not installed."""
    import io
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise PDFExtractionError("The PDF could not be opened.") from exc

    pages = [
        {"page": i + 1, "text": (page.extract_text() or "").strip()}
        for i, page in enumerate(reader.pages)
    ]
    if not pages:
        raise PDFExtractionError("The PDF contains no pages.")
    if not any(p["text"] for p in pages):
        raise PDFExtractionError("No selectable text was found.")
    return ExtractedDocument(filename=filename, page_count=len(pages), pages=pages)


# ---------------------------------------------------------------------------
# LangChain Document chunking (for RAG ingestion of uploaded PDFs)
# ---------------------------------------------------------------------------

def load_pdfs_from_bytes(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Extract text from uploaded PDF bytes and return chunked LangChain Documents."""
    doc = extract_pdf(file_bytes, filename)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    chunks: list[Document] = []
    for page in doc.pages:
        if not page["text"]:
            continue
        for chunk in splitter.split_text(page["text"]):
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "page": page["page"],
                    "type": "pdf",
                },
            ))
    return chunks


# ---------------------------------------------------------------------------
# Text chunking helper (for LLM context-window splitting)
# ---------------------------------------------------------------------------

def chunk_text(text: str, limit: int = config.LLM_CHUNK_CHAR_LIMIT) -> list[str]:
    """Split long text on paragraph boundaries for sequential LLM processing."""
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        while len(paragraph) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:limit])
            paragraph = paragraph[limit:]
        candidate = f"{current}\n\n{paragraph}".strip()
        if len(candidate) > limit and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]
