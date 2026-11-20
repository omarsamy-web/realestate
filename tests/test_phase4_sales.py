"""Phase 4: lead profiles, clustering, priority and sales recommendations."""
import pandas as pd

from src import config, lead_clustering, price_ingest, sales_agent, telegram_bot


def notes_file(tmp_path):
    rows = []
    for i in range(30):
        loc, beds, budget = [("التجمع", 2, "6 مليون"), ("زايد", 3, "8 مليون"), ("الساحل", 2, "5 مليون")][i % 3]
        rows.append({"note_id": f"N{i}a", "lead_id": 1000 + i, "agent": "Mona", "created_at": "2026-11-10 10:00",
                     "channel": "call", "note_text": f"عايز شقة {beds} غرف في {loc} الميزانية {budget}"})
        rows.append({"note_id": f"N{i}b", "lead_id": 1000 + i, "agent": "Mona", "created_at": "2026-11-15 10:00",
                     "channel": "visit", "note_text": "After site visit: شايف السعر غالي" if i % 3 == 0 else "مردش"})
    rows.append({"note_id": "Nx", "lead_id": 999, "agent": "Ali", "created_at": "2026-01-01",
                 "channel": "call", "note_text": "bought from another broker"})
    path = tmp_path / "notes.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return str(path)


def test_clustering_profiles_and_priority(temp_db, tmp_path):
    result = lead_clustering.run_clustering(notes_file(tmp_path), k=3)
    assert result["leads"] == 31 + 6 and result["k"] == 3
    labels = " | ".join(c["label"] for c in result["clusters"])
    assert "New Cairo" in labels and all(c["playbook"] for c in result["clusters"])
    p = lead_clustering.get_profile("1000")
    assert p["location"] == "New Cairo" and p["budget"] == 6_000_000 and p["bedrooms"] == 2
    assert "price_high" in p["objections"] and p["matching_units"] >= 1
    assert lead_clustering.get_profile("999")["priority"] == 0           # lost
    top = lead_clustering.priority_list(5)
    assert top[0]["priority"] >= top[-1]["priority"] and all(not t["lost"] for t in top)


def test_recommendations_explain_and_draft(temp_db, tmp_path):
    lead_clustering.run_clustering(notes_file(tmp_path), k=3)
    price_ingest.ingest_text("PR-102 4,900,000", source="test")              # a recent price drop
    rec = sales_agent.recommend_for("1000")
    ids = [u["unit_id"] for u in rec["units"]]
    assert ids[0] == "PR-102"
    assert any("price dropped" in w for w in rec["units"][0]["why"])
    assert any("السعر" in t for t in rec["units"][0]["talking_points"])  # answers the price objection
    assert "4,900,000" in rec["whatsapp_draft"]
    text = sales_agent.format_recommendation(rec)
    assert "PR-102" in text and "واتساب" in text


def test_recommend_for_unprofiled_crm_lead(temp_db):
    rec = sales_agent.recommend_for("Mariam")
    assert rec["lead"]["location"] == "New Cairo" and rec["units"]


def test_relaxes_budget_when_nothing_fits(temp_db):
    rec = sales_agent.recommend({"location": "New Cairo", "bedrooms": 3, "budget": 7_000_000, "objections": []})
    assert rec["relaxed_budget"] and rec["units"][0]["unit_id"] == "PR-104"


class FakeTelegram:
    def __init__(self):
        self.replies = []

    def reply(self, chat_id, message_id, text):
        self.replies.append(text)


def test_sales_bot_commands(temp_db, tmp_path, monkeypatch):
    lead_clustering.run_clustering(notes_file(tmp_path), k=3)
    monkeypatch.setattr(config, "TELEGRAM_SALES_CHAT_IDS", {"-9"})
    tg, chat = FakeTelegram(), {"id": -9, "type": "group"}
    for i, cmd in enumerate(["/priority 3", "/clusters", "/lead 1000", "/recommend 1000", "/newleads", "/status"]):
        assert telegram_bot.handle_internal(tg, {"message_id": i, "chat": chat, "text": cmd}) is None
    priority, clusters, lead, rec, new, status = tg.replies
    assert "كلّم دول" in priority and "#" in clusters and "New Cairo" in lead
    assert "PR-" in rec and "مافيش" in new and "الوحدات المتاحة" in status
    # ordinary chatter in the sales group is ignored (not treated as prices or questions)
    assert telegram_bot.handle_internal(tg, {"message_id": 99, "chat": chat, "text": "صباح الخير"}) is None
    assert len(tg.replies) == 6
