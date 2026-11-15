"""Tests for Phase 2 Research Lab and unified services."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src import config
from src.pdf_service import chunk_text, PDFExtractionError, extract_pdf
from src.research_workflow import build_research_graph, run_research_workflow
from src.report_generator import save_report, list_reports


def test_chunk_text():
    text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
    chunks = chunk_text(text, limit=20)
    assert len(chunks) >= 2
    assert "Paragraph 1" in chunks[0]


def test_chunk_text_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="positive"):
        chunk_text("Some text", limit=0)


def test_chunk_text_empty_input():
    assert chunk_text("") == []


def test_extract_pdf_empty_bytes():
    with pytest.raises(PDFExtractionError, match="empty"):
        extract_pdf(b"", "empty.pdf")


def test_extract_pdf_non_pdf():
    with pytest.raises(PDFExtractionError, match="upload a PDF"):
        extract_pdf(b"not a pdf", "file.txt")


def test_research_graph_compilation():
    graph = build_research_graph()
    assert graph is not None


def test_report_save_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", str(tmp_path))
    report_file = save_report("sample.pdf", "# Test Report\n\nThis is a test report.")
    assert report_file.exists()
    assert "Test Report" in report_file.read_text(encoding="utf-8")

    reports = list_reports()
    assert len(reports) == 1
    assert reports[0]["filename"] == report_file.name

