"""Daily documents: any file sent to the bot is stored, indexed, and scanned for prices."""
import io

import docx
import pymupdf

from src import config, documents, telegram_bot
from src.loaders import load_all


def make_pdf(lines):
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(lines):
        page.insert_text((50, 72 + 20 * i), line)
    return doc.tobytes()


def make_docx(lines):
    d = docx.Document()
    for line in lines:
        d.add_paragraph(line)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def price(radar, unit_id):
    return radar.rows("SELECT price, status FROM units WHERE unit_id=?", (unit_id,))[0]


def test_pdf_brochure_is_saved_and_indexed(temp_db):
    pdf = make_pdf(["Sunrise Gardens brochure", "Clubhouse opens in March 2027",
                    "Palm Residence 10% down payment over 8 years"])
    r = documents.ingest_document(pdf, "sunrise.pdf", source="test")
    assert r.path and r.chars > 0 and not r.duplicate
    assert price(temp_db, "PR-101")["price"] == 3_200_000  # "10% down payment" must not reprice
    chunks = [d for d in load_all() if d.metadata["source"] == "sunrise.pdf"]
    assert chunks and "Clubhouse" in chunks[0].page_content


def test_pdf_price_list_updates_units(temp_db):
    pdf = make_pdf(["Daily price list", "PR-101   3,300,000", "NV-103 available"])
    r = documents.ingest_document(pdf, "prices today.pdf", source="test")
    assert price(temp_db, "PR-101")["price"] == 3_300_000
    assert price(temp_db, "NV-103")["status"] == "Available"
    assert len(r.prices.applied) == 2


def test_word_file_updates_units(temp_db):
    r = documents.ingest_document(make_docx(["تحديث", "NV-101 2.8 مليون"]), "update.docx", source="test")
    assert price(temp_db, "NV-101")["price"] == 2_800_000
    assert "NV-101" in r.summary("ar")


def test_unit_sheet_goes_to_units_table_not_index(temp_db):
    csv = b"Unit ID,Price (EGP),Status\nPR-201,14800000,Available\n"
    r = documents.ingest_document(csv, "units today.csv", source="test")
    assert r.kind == "sheet" and r.prices.applied
    assert not [d for d in load_all() if d.metadata["source"] == "units today.csv"]


def test_other_sheet_is_indexed_as_knowledge(temp_db):
    csv = "Agent,Target\nMona,12 deals\n".encode()
    documents.ingest_document(csv, "targets.csv", source="test")
    assert any("Mona" in d.page_content for d in load_all() if d.metadata["source"] == "targets.csv")


def test_text_file_and_duplicates(temp_db):
    data = "Meeting notes: launch on Sunday".encode()
    assert not documents.ingest_document(data, "notes.txt", source="test").duplicate
    assert documents.ingest_document(data, "notes copy.txt", source="test").duplicate
    assert len(documents.list_documents()) == 1


def test_unsupported_type_is_refused(temp_db):
    try:
        documents.ingest_document(b"x", "image.png", source="test")
    except documents.UnsupportedDocument:
        pass
    else:
        raise AssertionError("png should be refused")


class FakeTelegram:
    def __init__(self, files=None):
        self.replies, self.files = [], files or {}

    def reply(self, chat_id, message_id, text):
        self.replies.append(text)

    def download(self, file_id):
        return self.files[file_id]


def test_telegram_any_file_and_commands(temp_db, monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_PRICE_CHAT_IDS", {"-100"})
    tg = FakeTelegram({"p": make_pdf(["PR-102 5,200,000"]), "x": b"..."})
    chat = {"id": -100}
    r = telegram_bot.handle_message(tg, {"message_id": 1, "chat": chat,
                                         "document": {"file_id": "p", "file_name": "list.pdf", "file_size": 900}})
    assert r.prices.applied and "list.pdf" in tg.replies[-1]
    telegram_bot.handle_message(tg, {"message_id": 2, "chat": chat,
                                     "document": {"file_id": "x", "file_name": "clip.mp4"}})
    assert "مش بقرا" in tg.replies[-1]
    telegram_bot.handle_message(tg, {"message_id": 3, "chat": chat, "text": "/status"})
    assert "list.pdf" in tg.replies[-1]
    telegram_bot.handle_message(tg, {"message_id": 4, "chat": chat, "text": "/help@radar_bot"})
    assert "PDF" in tg.replies[-1]
