"""Step 1 - Data integration: PDFs + Google Sheets -> LangChain Documents (plain text)."""
import re
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


# ---------- PDFs (unstructured) ----------
def load_pdfs(pdf_dir: str = config.PDF_DIR) -> list[Document]:
    """Read every PDF, split into overlapping chunks, keep file + page as metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks: list[Document] = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        pages = PyPDFLoader(str(pdf)).load()
        for chunk in splitter.split_documents(pages):
            chunk.metadata["source"] = pdf.name
            chunk.metadata["page"] = int(chunk.metadata.get("page", 0)) + 1
            chunk.metadata["type"] = "pdf"
            chunks.append(chunk)
    return chunks


# ---------- Google Sheets / CSV / Excel (structured) ----------
def _sheet_to_dataframe(source: str, max_rows: int = config.MAX_RAG_SHEET_ROWS) -> pd.DataFrame:
    """Accepts a Google Sheets link, a local .csv, or a local .xlsx."""
    match = re.search(r"docs\.google\.com/spreadsheets/d/([\w-]+)", source)
    if match:
        sheet_id = match.group(1)
        gid = re.search(r"gid=(\d+)", source)
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        if gid:
            url += f"&gid={gid.group(1)}"
        return pd.read_csv(url, nrows=max_rows)  # sheet must be shared as "Anyone with the link can view"
    if source.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(source, nrows=max_rows)
    return pd.read_csv(source, nrows=max_rows)


def load_sheet(source: str = config.SHEET_SOURCE, max_rows: int = config.MAX_RAG_SHEET_ROWS) -> list[Document]:
    """Turn each row into one self-contained text record.

    Example: "Unit ID: PR-101; Project: Palm Residence; Bedrooms: 2; Price EGP: 3,450,000; ..."
    One row = one document, so a unit's facts are never split across chunks.
    """
    df = _sheet_to_dataframe(source, max_rows=max_rows).dropna(how="all")
    name = "Google Sheet" if "docs.google.com" in source else Path(source).name
    docs = []
    for i, row in df.iterrows():
        parts = []
        for col, val in row.items():
            if pd.isna(val):
                continue
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            if isinstance(val, int) and abs(val) >= 10000:
                val = f"{val:,}"
            parts.append(f"{col}: {val}")
        docs.append(
            Document(
                page_content="; ".join(parts),
                metadata={"source": name, "row": int(i) + 2, "type": "sheet"},  # +2 = sheet row number
            )
        )
    return docs


def load_all() -> list[Document]:
    docs = load_pdfs()
    csv_file = Path(config.SHEET_SOURCE)
    if csv_file.exists():
        docs += load_sheet(str(csv_file))

    # Also load any additional spreadsheets in data/xlsx
    xlsx_dir = Path(config.ROOT_DIR) / "data" / "xlsx"
    if xlsx_dir.exists():
        for f in sorted(xlsx_dir.glob("*.xlsx")):
            if f.name != csv_file.name:
                try:
                    docs += load_sheet(str(f))
                except Exception:
                    pass
        for f in sorted(xlsx_dir.glob("*.csv")):
            if f.name != csv_file.name:
                try:
                    docs += load_sheet(str(f))
                except Exception:
                    pass

    if not docs:
        raise RuntimeError("No documents found. Check PDF_DIR and SHEET_SOURCE.")
    return docs
