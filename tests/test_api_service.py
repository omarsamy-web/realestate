"""Phase 1: HTTP API smoke tests (offline LLM, temp database)."""
import pytest
from fastapi.testclient import TestClient

from src import api, config, telegram_bot
from src.indexer import build_index
from src.loaders import load_all


@pytest.fixture
def client(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OFFLINE_MODE", True)
    monkeypatch.setattr(config, "API_KEY", "")
    monkeypatch.setattr(config, "INGEST_WEBHOOK_TOKEN", "secret")
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path / "index"))
    build_index(load_all(), offline=True)
    api.reset_chat_cache()
    with TestClient(api.app) as c:
        yield c
    api.reset_chat_cache()


def test_health_and_units(client):
    h = client.get("/health").json()
    assert h["ok"] and h["units"] == 13
    units = client.get("/units", params={"location": "New Cairo", "bedrooms": 2}).json()
    assert {u["unit_id"] for u in units} == {"PR-102"}


def test_webhook_ingest_then_history(client):
    r = client.post("/ingest/webhook", json={"text": "PR-101 3,300,000", "source": "whatsapp", "message_id": "1"},
                    headers={"X-Ingest-Token": "wrong"})
    assert r.status_code == 401
    r = client.post("/ingest/webhook", json={"text": "PR-101 3,300,000", "source": "whatsapp", "message_id": "1"},
                    headers={"X-Ingest-Token": "secret"})
    assert r.json()["applied"][0]["new_price"] == 3_300_000
    unit = client.get("/units/PR-101").json()
    assert unit["price"] == 3_300_000 and unit["price_history"][0]["source"] == "whatsapp"


def test_chat_answers_with_live_price(client):
    client.post("/ingest/text", json={"text": "PR-102 5,250,000"})
    r = client.post("/chat", json={"message": "What is the price of PR-102?"}).json()
    assert "5,250,000" in r["answer"] and r["live_units"][0]["unit_id"] == "PR-102"
    follow = client.post("/chat", json={"message": "and NV-101?", "session_id": r["session_id"]}).json()
    assert follow["session_id"] == r["session_id"]


def test_leads_and_pending_flow(client):
    lead = client.post("/leads", json={"name": "Test Buyer", "phone": "+201000000001", "budget": 6000000,
                                       "area": "New Cairo", "bedrooms": 2}).json()
    assert client.get(f"/leads/{lead['id']}").json()["name"] == "Test Buyer"
    client.post("/ingest/text", json={"text": "PR-101 9,000,000"})
    pending = client.get("/ingest/pending").json()
    assert len(pending) == 1
    client.post(f"/ingest/pending/{pending[0]['id']}/decision", json={"approve": False})
    assert client.get("/units/PR-101").json()["price"] == 3_200_000


def test_api_key_is_enforced(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "k")
    assert client.get("/units").status_code == 401
    assert client.get("/units", headers={"X-API-Key": "k"}).status_code == 200
    assert client.get("/health").status_code == 200


class FakeTelegram:
    def __init__(self, files=None):
        self.replies, self.files = [], files or {}

    def reply(self, chat_id, message_id, text):
        self.replies.append(text)

    def download(self, file_id):
        return self.files[file_id]


def test_telegram_group_message(client, monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_PRICE_CHAT_IDS", {"-100"})
    tg = FakeTelegram()
    msg = {"message_id": 7, "chat": {"id": -100, "title": "Prices"}, "text": "NV-101 2,700,000"}
    result = telegram_bot.handle_message(tg, msg)
    assert result.applied and "NV-101" in tg.replies[0]
    assert telegram_bot.handle_message(tg, msg).duplicate  # same message again is ignored
    other = {"message_id": 8, "chat": {"id": -200}, "text": "NV-101 2,900,000"}
    assert telegram_bot.handle_message(tg, other) is None  # not an allowed group


def test_telegram_sheet_file(client, monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_PRICE_CHAT_IDS", {"-100"})
    csv = b"Unit ID,Price (EGP),Status\nPR-201,14800000,Available\n"
    tg = FakeTelegram({"f1": csv})
    msg = {"message_id": 9, "chat": {"id": -100}, "document": {"file_id": "f1", "file_name": "today.csv"}}
    assert telegram_bot.handle_message(tg, msg).applied
