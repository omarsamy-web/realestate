"""Daily documents: every file sent to the bot is kept and becomes knowledge.

For each incoming file (PDF, Excel, CSV, Word, text, Markdown, JSON):
  1. It is saved under data/documents/<date>/ and recorded in the `documents` table
     (the same file sent twice is recognised by its hash and skipped).
  2. A unit sheet (has a unit column and a price column) updates the units table.
     Any other file is scanned for unit price lines ("PR-101 3,350,000"),
     so a PDF or Word price list updates prices too.
  3. The RAG index is rebuilt in the background, so the chat can answer from it.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config, price_ingest, radar

SHEETS = (".xlsx", ".xls", ".csv")
TEXT_FILES = (".txt", ".md", ".json")
SUPPORTED = (".pdf", ".docx", *SHEETS, *TEXT_FILES)
MAX_SHEET_ROWS_AS_TEXT = 5000


class UnsupportedDocument(ValueError):
    pass


@dataclass
class DocumentResult:
    filename: str
    kind: str
    path: str = ""
    duplicate: bool = False
    chars: int = 0
    prices: price_ingest.IngestResult | None = None
    notes: list = field(default_factory=list)

    def summary(self, lang: str = "ar") -> str:
        if self.duplicate:
            return (f"📄 {self.filename}: الملف ده اتبعت قبل كده." if lang == "ar"
                    else f"📄 {self.filename}: already received.")
        head = (f"📄 اتحفظ {self.filename} واتضاف لقاعدة المعرفة."
                if lang == "ar" else f"📄 Saved {self.filename} to the knowledge base.")
        lines = [head]
        p = self.prices
        if p and (p.applied or p.added or p.held):
            lines.append(p.summary(lang))
        elif p and p.rejected and self.kind == "sheet":
            lines.append(p.summary(lang))
        lines += self.notes
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"filename": self.filename, "kind": self.kind, "path": self.path, "duplicate": self.duplicate,
                "chars": self.chars, "prices": self.prices.as_dict() if self.prices else None, "notes": self.notes}


# --------------------------------------------------------------------------- text extraction
def kind_of(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise UnsupportedDocument(f"Unsupported file type {ext or '(none)'}. Send one of: {', '.join(SUPPORTED)}")
    return {".pdf": "pdf", ".docx": "word"}.get(ext, "sheet" if ext in SHEETS else "text")


def _read_sheet(data: bytes, filename: str):
    import pandas as pd
    if filename.lower().endswith(".csv"):
        return {"Sheet1": pd.read_csv(io.StringIO(data.decode("utf-8-sig", errors="replace")))}
    return pd.read_excel(io.BytesIO(data), sheet_name=None)


def extract_pages(data: bytes, filename: str) -> list[dict]:
    """Return [{"page": n, "text": ...}] — pages for PDFs, sheets for workbooks, one block otherwise."""
    kind = kind_of(filename)
    if kind == "pdf":
        from .pdf_service import extract_pdf
        return [p for p in extract_pdf(data, filename).pages if p.get("text", "").strip()]
    if kind == "word":
        import docx
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for table in d.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return [{"page": 1, "text": "\n".join(parts)}]
    if kind == "sheet":
        pages = []
        for i, (name, frame) in enumerate(_read_sheet(data, filename).items(), start=1):
            frame = frame.dropna(how="all").head(MAX_SHEET_ROWS_AS_TEXT)
            rows = ["; ".join(f"{c}: {v}" for c, v in r.items() if str(v) != "nan") for _, r in frame.iterrows()]
            pages.append({"page": i, "text": f"[sheet {name}]\n" + "\n".join(rows)})
        return pages
    text = data.decode("utf-8-sig", errors="replace")
    if filename.lower().endswith(".json"):
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=1)
        except json.JSONDecodeError:
            pass
    return [{"page": 1, "text": text}]


def _is_unit_sheet(data: bytes, filename: str) -> bool:
    if not filename.lower().endswith(SHEETS):
        return False
    try:
        price_ingest.parse_sheet(data, filename)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- storage
def _docs_dir() -> Path:
    return Path(config.DOCS_DIR)


def _safe_name(filename: str) -> str:
    name = re.sub(r"[^\w.\- ()؀-ۿ]", "_", Path(filename).name).strip() or "document"
    return name[:120]


def _ensure_table():
    with radar.connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL, kind TEXT NOT NULL,
            source TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL, chars INTEGER, units_changed INTEGER,
            created_at TEXT NOT NULL)""")


def list_documents(limit: int = 50) -> list[dict]:
    _ensure_table()
    return radar.rows("SELECT * FROM documents ORDER BY id DESC LIMIT ?", (limit,))


def ingest_document(data: bytes, filename: str, source: str = "manual", message_key: str | None = None,
                    llm=None, rebuild: bool = True) -> DocumentResult:
    if not data:
        raise ValueError("The file is empty.")
    if len(data) > config.MAX_DOCUMENT_BYTES:
        raise ValueError(f"File limit is {config.MAX_DOCUMENT_BYTES // 1_000_000} MB.")
    kind = kind_of(filename)
    result = DocumentResult(filename=filename, kind=kind)
    _ensure_table()
    digest = hashlib.sha256(data).hexdigest()
    if radar.rows("SELECT 1 FROM documents WHERE sha256=?", (digest,)):
        result.duplicate = True
        return result

    # 1) prices
    if _is_unit_sheet(data, filename):
        result.prices = price_ingest.ingest_file(data, filename, source=source, message_key=message_key, rebuild=False)
        pages = []
    else:
        pages = extract_pages(data, filename)
        text = "\n".join(p["text"] for p in pages)
        result.chars = len(text)
        if not text.strip():
            result.notes.append("⚠️ مافيش نص يتقري في الملف ده (ممكن يكون صور متصورة).")
        else:
            # documents only update unit lines; a "10%" in a brochure must never reprice a project
            result.prices = price_ingest.ingest_text(text, source=f"{source}:{filename}", message_key=message_key,
                                                     llm=None, rebuild=False, project_changes=False)

    # 2) keep the file for the knowledge base
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = _docs_dir() / day
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / _safe_name(filename)
    if path.exists():
        path = folder / f"{path.stem}_{digest[:8]}{path.suffix}"
    path.write_bytes(data)
    result.path = str(path)

    changed = 0
    if result.prices:
        changed = len(result.prices.applied) + len(result.prices.added)
    with radar.connect() as db:
        db.execute("""INSERT INTO documents(filename,path,kind,source,sha256,chars,units_changed,created_at)
                      VALUES(?,?,?,?,?,?,?,?)""",
                   (filename, str(path), kind, source, digest, result.chars, changed, radar.now()))
        db.execute("INSERT INTO events(kind,details,created_at) VALUES(?,?,?)",
                   ("Document received", f"{source}: {filename} ({kind}), {changed} units changed", radar.now()))

    # 3) refresh the RAG index (live prices never wait for this)
    if rebuild:
        price_ingest.schedule_index_rebuild()
    return result


# --------------------------------------------------------------------------- RAG loading
def load_documents(docs_dir: str | None = None) -> list[Document]:
    """Chunks of every saved daily document, with file + page metadata for citations."""
    root = Path(docs_dir or config.DOCS_DIR)
    if not root.exists():
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        if path.suffix.lower() in SHEETS and _is_unit_sheet(path.read_bytes(), path.name):
            continue  # unit sheets live in the units table; indexing them would serve stale prices
        try:
            pages = extract_pages(path.read_bytes(), path.name)
        except Exception:
            continue
        kind = "pdf" if path.suffix.lower() == ".pdf" else "doc"
        for page in pages:
            for piece in splitter.split_text(page["text"]):
                chunks.append(Document(page_content=piece, metadata={
                    "source": path.name, "page": page["page"], "type": "pdf",
                    "kind": kind, "received": path.parent.name}))
    return chunks
