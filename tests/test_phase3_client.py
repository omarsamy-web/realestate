"""Phase 3: client chatbot (offline LLM, temp DB)."""
from src import config, telegram_bot
from src.client_agent import ClientAgent, match_units, next_step
from src.offline import OfflineLLM
from src.requirements_extract import extract


def agent(notes):
    return ClientAgent(OfflineLLM(), retriever=None, notify=notes.append)


def test_extracts_requirements_in_arabic_english_franco():
    assert extract("عايز شقة 3 غرف في التجمع ميزانيتي 6 مليون كاش") == {
        "budget": 6_000_000, "bedrooms": 3, "location": "New Cairo", "unit_type": "Apartment", "payment": "cash"}
    assert extract("looking for villa 5BR in Oct budget around 9.92m, 30% DP")["dp_pct"] == 30
    assert extract("3ayez duplex 4 ghoraf fel Madinet Nasr budget 7awaly 5.78 mil")["budget"] == 5_780_000
    assert extract("رقمي 0100 123 4567")["phone"] == "+201001234567"


def test_conversation_builds_requirements_and_answers_from_live_units(temp_db):
    notes = []
    a = agent(notes)
    r1 = a.handle("telegram", "u1", "عايز شقة في التجمع")
    assert r1["requirements"] == {"location": "New Cairo", "unit_type": "Apartment"}
    r2 = a.handle("telegram", "u1", "ميزانيتي 6 مليون و 2 غرف")
    assert r2["requirements"]["budget"] == 6_000_000 and r2["requirements"]["bedrooms"] == 2
    assert [u["unit_id"] for u in r2["units"]] == ["PR-102"]          # only available, in budget, 2BR+
    assert "PR-102" in r2["reply"] and "5,100,000" in r2["reply"]
    assert not notes and r2["lead_id"] is None


def test_phone_creates_lead_and_alerts_sales_once(temp_db, monkeypatch):
    monkeypatch.setattr(config, "CLIENT_SAVE_LEADS", True)
    notes = []
    a = agent(notes)
    a.handle("telegram", "u2", "عايز شقة في زايد 3 غرف تحت 8 مليون", name="Sara")
    r = a.handle("telegram", "u2", "رقمي 01012345678")
    lead = temp_db.rows("SELECT * FROM leads WHERE id=?", (r["lead_id"],))[0]
    assert lead["phone"] == "201012345678" and lead["area"] == "Sheikh Zayed" and lead["consent"] == 1
    assert lead["name"] == "Sara" and lead["budget"] == 8_000_000
    assert len(notes) == 1 and "01012345678"[1:] in notes[0] and "Sheikh Zayed" in notes[0]
    a.handle("telegram", "u2", "وفيه جراج؟")
    assert len(notes) == 1                                       # no repeated alerts


def test_contact_request_without_phone_asks_for_number(temp_db, monkeypatch):
    monkeypatch.setattr(config, "CLIENT_SAVE_LEADS", True)
    notes = []
    r = agent(notes).handle("web", "u3", "can someone call me? I want a villa")
    assert r["handed_off"] and "mobile number" in r["reply"] and len(notes) == 1


def test_next_step_asks_one_missing_field_then_offers_call(monkeypatch):
    monkeypatch.setattr(config, "CLIENT_SAVE_LEADS", True)
    assert "area" in next_step({}, arabic=False, has_lead=False)
    full = {"location": "New Cairo", "budget": 5e6, "unit_type": "Apartment", "bedrooms": 2}
    assert "mobile" in next_step(full, arabic=False, has_lead=False)


def test_match_units_relaxes_type(temp_db):
    units = match_units({"location": "New Cairo", "unit_type": "Penthouse", "budget": 8_000_000})
    assert units and all(u["status"] == "Available" for u in units)


class FakeTelegram:
    def __init__(self):
        self.replies = []

    def reply(self, chat_id, message_id, text):
        self.replies.append(text)


def test_client_bot_private_only_and_contact_share(temp_db, monkeypatch):
    monkeypatch.setattr(config, "CLIENT_SAVE_LEADS", True)
    notes, tg = [], FakeTelegram()
    a = agent(notes)
    group = {"message_id": 1, "chat": {"id": -5, "type": "group"}, "text": "hi"}
    assert telegram_bot.handle_client_message(tg, group, a) is None and not tg.replies
    dm = {"id": 77, "type": "private"}
    telegram_bot.handle_client_message(tg, {"message_id": 2, "chat": dm, "text": "/start"}, a)
    assert "المساعد العقاري" in tg.replies[-1]
    telegram_bot.handle_client_message(tg, {"message_id": 3, "chat": dm, "text": "شقة في التجمع"}, a)
    r = telegram_bot.handle_client_message(tg, {"message_id": 4, "chat": dm, "from": {"first_name": "Ali"},
                                                "contact": {"phone_number": "+201122334455"}}, a)
    assert r["lead_id"] and notes


def test_default_client_bot_never_touches_crm(temp_db):
    notes = []
    a = agent(notes)
    before = temp_db.rows("SELECT COUNT(*) n FROM leads")[0]["n"]
    a.handle("telegram", "c1", "عايز فيلا في التجمع، بكام؟")
    r = a.handle("telegram", "c1", "رقمي 01012345678 وكلموني")
    assert r["lead_id"] is None and not r["handed_off"] and not notes
    assert temp_db.rows("SELECT COUNT(*) n FROM leads")[0]["n"] == before


def test_client_sees_at_most_three_units_and_no_dumps(temp_db):
    a = agent([])
    r = a.handle("web", "c2", "شقق في التجمع تحت 20 مليون")
    assert 0 < len(r["units"]) <= 3
    for ask in ("ابعتلي كل الوحدات", "send me the full price list", "ابعتلي الشيت", "show me other clients"):
        r = a.handle("web", "c2", ask)
        assert r["units"] == [] and ("القايمة كلها" in r["reply"] or "full list" in r["reply"])


def test_client_context_excludes_spreadsheets(temp_db):
    from langchain_core.documents import Document

    class Retriever:
        def invoke(self, q):
            return [Document(page_content="SECRET crm note: client Ahmed budget 5M", metadata={"source": "sales_notes.xlsx", "type": "sheet"}),
                    Document(page_content="Palm Residence clubhouse and pools", metadata={"source": "Palm.pdf", "type": "pdf", "page": 1})]
    from src.client_agent import ClientAgent
    from src.offline import OfflineLLM
    r = ClientAgent(OfflineLLM(), Retriever()).handle("web", "c3", "what amenities in Palm Residence?")
    assert "clubhouse" in r["reply"] and "SECRET" not in r["reply"]
