"""Live unit inventory: structured lookups straight from the units table.

Prices change every day, so the chat never reads prices from the vector index.
Instead, each question is matched against the units table and the matching rows
are injected into the prompt as "live inventory" with their last-updated time.
"""
from __future__ import annotations

import re
import sqlite3

from . import radar

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

LOCATION_ALIASES = {
    "new cairo": "New Cairo", "التجمع": "New Cairo", "القاهرة الجديدة": "New Cairo",
    "sheikh zayed": "Sheikh Zayed", "زايد": "Sheikh Zayed", "الشيخ زايد": "Sheikh Zayed",
    "mostakbal": "Mostakbal City", "المستقبل": "Mostakbal City",
    "assiut": "Assiut", "اسيوط": "Assiut", "أسيوط": "Assiut",
}

TYPE_ALIASES = {
    "apartment": "Apartment", "شقة": "Apartment", "شقه": "Apartment", "شقق": "Apartment",
    "villa": "Villa", "فيلا": "Villa", "فلل": "Villa", "townhouse": "Townhouse", "تاون": "Townhouse",
    "studio": "Studio", "استوديو": "Studio", "ستوديو": "Studio",
    "duplex": "Duplex", "دوبلكس": "Duplex", "chalet": "Chalet", "شاليه": "Chalet", "شاليهات": "Chalet",
}


def normalize_digits(text: str) -> str:
    return (text or "").translate(ARABIC_DIGITS)


def parse_amount(raw: str) -> float | None:
    """'3,450,000' / '3.45M' / '3.45 مليون' / '3450k' -> 3450000.0"""
    if raw is None:
        return None
    text = normalize_digits(str(raw)).strip().lower().replace("٬", ",").replace("،", ",")
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(m\b|mil\b|mn\b|million|مليون|k\b|الف|ألف)?", text)
    if not m:
        return None
    number = float(m.group(1).replace(",", ""))
    unit = m.group(2) or ""
    if unit in ("m", "mil", "mn", "million", "مليون"):
        number *= 1_000_000
    elif unit in ("k", "الف", "ألف"):
        number *= 1_000
    return number


def extract_filters(question: str) -> dict:
    """Pull simple, explicit filters out of a free-text question (Arabic or English)."""
    q = normalize_digits(question).lower()
    filters: dict = {}

    projects = [r["project"] for r in radar.rows("SELECT DISTINCT project FROM units WHERE project IS NOT NULL")]
    hits = [p for p in projects if p and p.lower() in q]
    if hits:
        filters["projects"] = hits

    unit_ids = [r["unit_id"] for r in radar.rows("SELECT unit_id FROM units")]
    ids = [u for u in unit_ids if re.search(rf"(?<![\w-]){re.escape(u.lower())}(?![\w-])", q)]
    if ids:
        filters["unit_ids"] = ids

    for alias, loc in LOCATION_ALIASES.items():
        if alias in q:
            filters["location"] = loc
            break
    for alias, typ in TYPE_ALIASES.items():
        if alias in q:
            filters["type"] = typ
            break

    beds = re.search(r"(\d+)\s*(?:bed|bedroom|br\b|غرف|غرفة|اوض|أوض)", q)
    if beds:
        filters["bedrooms"] = int(beds.group(1))

    budget = re.search(
        r"(?:budget|under|below|max|up to|ميزانية|ميزانيتي|تحت|اقل من|أقل من|لحد|حدود)\s*(?:of|is|:)?\s*"
        r"(\d[\d,.]*\s*(?:m\b|million|مليون|k\b|الف|ألف)?)", q)
    if budget:
        filters["max_price"] = parse_amount(budget.group(1))
    return filters


def search_units(projects=None, unit_ids=None, location=None, type=None, bedrooms=None,
                 max_price=None, min_price=None, status="Available", limit=10) -> list[dict]:
    sql, args = "SELECT * FROM units WHERE 1=1", []
    if unit_ids:
        sql += f" AND unit_id IN ({','.join('?' * len(unit_ids))})"
        args += list(unit_ids)
    if projects:
        sql += f" AND project IN ({','.join('?' * len(projects))})"
        args += list(projects)
    if location:
        sql += " AND location=?"
        args.append(location)
    if type:
        sql += " AND type=?"
        args.append(type)
    if bedrooms is not None:
        sql += " AND bedrooms=?"
        args.append(int(bedrooms))
    if max_price:
        sql += " AND price<=?"
        args.append(float(max_price))
    if min_price:
        sql += " AND price>=?"
        args.append(float(min_price))
    if status and not unit_ids:
        sql += " AND status=?"
        args.append(status)
    sql += " ORDER BY price ASC LIMIT ?"
    args.append(int(limit))
    return radar.rows(sql, tuple(args))


def format_unit(u: dict) -> str:
    beds = "Studio" if not u.get("bedrooms") else f"{u['bedrooms']} bedrooms"
    return (f"Unit {u['unit_id']} | {u['project']} | {u['location']} | {u['type']} | {beds} | "
            f"{u['area']:g} sqm | Price EGP {u['price']:,.0f} | Status {u['status']} | "
            f"Delivery {u.get('delivery_year') or '-'} | price updated {u['updated_at']}")


def live_context(question: str, limit: int = 10) -> tuple[str, list[dict]]:
    """Return (prompt text, rows) with the units relevant to the question."""
    try:
        filters = extract_filters(question)
    except sqlite3.OperationalError:  # database not initialised yet
        return "", []
    if not filters:
        return "", []
    units = search_units(**filters, limit=limit)
    if not units:
        return f"No units in the live inventory match: {filters}.", []
    lines = [f"[live inventory] {format_unit(u)}" for u in units]
    return "\n".join(lines), units
