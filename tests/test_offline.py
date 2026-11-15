"""Smoke tests that run with no API key and no model download:  pytest -q"""
from src import config
from src.indexer import build_index
from src.loaders import load_all, load_sheet
from src.memory import BufferMemory, SummaryBufferMemory
from src.offline import OfflineLLM
from src.rag import RealEstateAssistant


def test_sheet_rows_become_text():
    docs = load_sheet()
    assert docs and "Unit ID: PR-101" in docs[0].page_content
    assert "3,200,000" in docs[0].page_content


def test_sheet_loading_respects_row_cap(tmp_path):
    source = tmp_path / "many_rows.csv"
    source.write_text("id,value\n" + "\n".join(f"{i},row-{i}" for i in range(20)), encoding="utf-8")
    docs = load_sheet(str(source), max_rows=3)
    assert len(docs) == 3


def test_retrieval_finds_right_source(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path))
    store = build_index(load_all(), offline=True)
    bot = RealEstateAssistant(store, OfflineLLM(), BufferMemory())
    _, docs, _ = bot.ask("reservation fee refundable")
    assert any(d.metadata["source"] == "Sales_Policy.pdf" for d in docs)


def test_memory_keeps_history_and_summarises():
    mem = SummaryBufferMemory(OfflineLLM(), max_turns=2, keep_turns=1)
    for i in range(4):
        mem.add_turn(f"q{i}", f"a{i}")
    assert len(mem.history()) <= 2 * 2       # never more than max_turns kept verbatim
    assert mem.summary()                     # older turns folded into the summary
