"""Phase 2: parsing and applying daily price messages (no network, no API key)."""
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src import price_ingest
from src.inventory import extract_filters, live_context, parse_amount


def price(radar, unit_id):
    return radar.rows("SELECT price, status FROM units WHERE unit_id=?", (unit_id,))[0]


def test_parse_amount_formats():
    assert parse_amount("3,450,000") == 3_450_000
    assert parse_amount("3.45M") == 3_450_000
    assert parse_amount("٣٫٤٥ مليون".replace("٫", ".")) == 3_450_000
    assert parse_amount("3450 الف") == 3_450_000


def test_english_and_arabic_lines_update_prices(temp_db):
    msg = """تحديث أسعار النهاردة
    PR-101 3,350,000
    NV-102 : 3.7 مليون
    PR-102 محجوزة
    random chatter 12
    """
    result = price_ingest.ingest_text(msg, source="test")
    assert price(temp_db, "PR-101")["price"] == 3_350_000
    assert price(temp_db, "NV-102")["price"] == 3_700_000
    assert price(temp_db, "PR-102")["status"] == "Reserved"
    assert len(result.applied) == 3
    assert any("random chatter" in r["line"] for r in result.rejected)


def test_area_and_bedrooms_are_not_taken_as_price(temp_db):
    price_ingest.ingest_text("PR-104 3 bedrooms 165 sqm 8,100,000", source="test")
    assert price(temp_db, "PR-104")["price"] == 8_100_000


def test_price_history_is_recorded(temp_db):
    price_ingest.ingest_text("PR-101 3,300,000", source="telegram:prices")
    hist = temp_db.price_history("PR-101")
    assert hist[0]["old_price"] == 3_200_000 and hist[0]["new_price"] == 3_300_000
    assert hist[0]["source"] == "telegram:prices"


def test_big_jump_is_held_until_approved(temp_db):
    result = price_ingest.ingest_text("PR-101 9,000,000", source="test")
    assert price(temp_db, "PR-101")["price"] == 3_200_000
    assert len(result.held) == 1
    pending = price_ingest.list_pending()
    price_ingest.decide_pending(pending[0]["id"], approve=True)
    assert price(temp_db, "PR-101")["price"] == 9_000_000
    assert price_ingest.list_pending() == []


def test_unknown_unit_is_rejected(temp_db):
    result = price_ingest.ingest_text("ZZ-999 4,000,000", source="test")
    assert result.rejected and result.rejected[0]["reason"] == "unknown unit"


def test_same_message_is_processed_once(temp_db):
    first = price_ingest.ingest_text("PR-101 3,300,000", source="t", message_key="t:1")
    again = price_ingest.ingest_text("PR-101 3,400,000", source="t", message_key="t:1")
    assert first.applied and again.duplicate
    assert price(temp_db, "PR-101")["price"] == 3_300_000


def test_project_wide_percentage(temp_db):
    before = price(temp_db, "NV-101")["price"]
    sold = price(temp_db, "NV-103")["price"]
    price_ingest.ingest_text("زيادة 5% على Nile View Towers", source="test")
    assert price(temp_db, "NV-101")["price"] == round(before * 1.05)
    assert price(temp_db, "NV-103")["price"] == sold  # sold units are left alone


def test_sheet_upload_updates_and_adds_units(temp_db):
    csv = ("Unit ID,Project,Location,Type,Bedrooms,Area (sqm),Price (EGP),Status,Delivery Year\n"
           "PR-101,Palm Residence,New Cairo,Apartment,1,65,3300000,Available,2028\n"
           "PR-999,Palm Residence,New Cairo,Apartment,2,120,6000000,Available,2029\n")
    result = price_ingest.ingest_file(csv.encode(), "prices.csv", source="test")
    assert price(temp_db, "PR-101")["price"] == 3_300_000
    assert price(temp_db, "PR-999")["price"] == 6_000_000
    assert len(result.added) == 1 and len(result.applied) == 1


def test_llm_fallback_cannot_invent_units(temp_db):
    fake = RunnableLambda(lambda _: AIMessage(
        '[{"unit_id": "PR-101", "price": 3300000}, {"unit_id": "XX-1", "price": 5000000}]'))
    updates = price_ingest.parse_text_with_llm("الوحدة PR-101 بقت بتلاتة مليون وتلتمية", fake)
    assert [u.unit_id for u in updates] == ["PR-101"]


def test_summary_mentions_changes(temp_db):
    result = price_ingest.ingest_text("PR-101 3,300,000", source="test")
    assert "PR-101" in result.summary("ar") and "3,300,000" in result.summary("en")


def test_live_context_uses_current_price(temp_db):
    price_ingest.ingest_text("PR-102 5,250,000", source="test")
    text, units = live_context("كام سعر PR-102؟")
    assert "5,250,000" in text and units[0]["unit_id"] == "PR-102"
    f = extract_filters("عايز شقة 2 غرف في التجمع تحت 6 مليون")
    assert f == {"location": "New Cairo", "type": "Apartment", "bedrooms": 2, "max_price": 6_000_000}
