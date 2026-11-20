"""Sales bot: full access to the units inventory (and nothing else unless SALES_CRM_ACCESS=1).

Salespeople ask in plain words ("فلل التجمع تحت 15 مليون", "كام وحدة متاحة في Rai Valleys",
"أرخص 3 غرف في زايد", "تاريخ سعر PR-101") and get real rows from the units table:
counts, price ranges per project, the matching units, and price history.
"""
from __future__ import annotations

import re
import statistics

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import inventory, radar
from .requirements_extract import extract

MAX_ROWS = 40
STATUS_WORDS = {
    "Sold": ("sold", "مباع", "المباعة", "اتباعت"),
    "Reserved": ("reserved", "محجوز", "المحجوزة"),
    "all": ("all statuses", "كل الحالات", "any status", "including sold"),
}
SORT_EXPENSIVE = ("most expensive", "highest price", "أغلى", "اغلى")

SALES_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the internal inventory assistant for the sales team of an Egyptian real-estate company. "
     "The salesperson has full access to the units inventory.\n"
     "- Answer ONLY from the query results below; they come live from the units database.\n"
     "- Give exact numbers: counts, price ranges, unit codes, prices, sizes. Never invent anything.\n"
     "- When listing units, use one compact line per unit: code | project | type | bedrooms | sqm | price | status.\n"
     "- Reply in the salesperson's language (Egyptian Arabic if they write Arabic). Be direct.\n\n"
     "Filters understood from the question: {filters}\n\n"
     "Summary:\n{summary}\n\n"
     "Context:\n{rows}"),
    ("human", "{question}"),
])


def filters_from(question: str) -> dict:
    q = inventory.normalize_digits(question).lower()
    f = inventory.extract_filters(question)  # projects, unit ids, location, type, bedrooms, max_price
    req = extract(question)
    if req.get("location") and "location" not in f:
        f["location"] = req["location"]
    if req.get("unit_type") and "type" not in f:
        f["type"] = req["unit_type"]
    if req.get("bedrooms") is not None and "bedrooms" not in f:
        f["bedrooms"] = req["bedrooms"]
    if req.get("budget") and "max_price" not in f:
        f["max_price"] = req["budget"]
    if req.get("budget_min"):
        f["min_price"] = req["budget_min"]
    m = re.search(r"(?:over|above|more than|فوق|أكتر من|اكتر من|من)\s*(\d[\d,.]*\s*(?:m\b|million|مليون|k\b|الف|ألف)?)", q)
    if m and not re.search(r"(تحت|لحد|under|below|max)", q[:m.start()][-15:]):
        v = inventory.parse_amount(m.group(1))
        if v and v >= 100_000:
            f["min_price"] = v
            if f.get("max_price") == v:  # "فوق 7 مليون" was also read as a plain budget
                f.pop("max_price")
    f["status"] = "Available"
    for status, words in STATUS_WORDS.items():
        if any(w in q for w in words):
            f["status"] = None if status == "all" else status
    f["sort"] = "desc" if any(w in q for w in SORT_EXPENSIVE) else "asc"
    return f


def query(f: dict, limit: int = MAX_ROWS) -> tuple[list[dict], dict]:
    where, args = ["1=1"], []
    if f.get("unit_ids"):
        where.append(f"unit_id IN ({','.join('?' * len(f['unit_ids']))})")
        args += f["unit_ids"]
    if f.get("projects"):
        where.append(f"project IN ({','.join('?' * len(f['projects']))})")
        args += f["projects"]
    for col, key in (("location", "location"), ("type", "type")):
        if f.get(key):
            where.append(f"{col}=?")
            args.append(f[key])
    if f.get("bedrooms") is not None:
        where.append("bedrooms=?")
        args.append(int(f["bedrooms"]))
    if f.get("max_price"):
        where.append("price<=?")
        args.append(float(f["max_price"]))
    if f.get("min_price"):
        where.append("price>=?")
        args.append(float(f["min_price"]))
    if f.get("status") and not f.get("unit_ids"):
        where.append("status=?")
        args.append(f["status"])
    sql_where = " AND ".join(where)
    order = "DESC" if f.get("sort") == "desc" else "ASC"
    rows = radar.rows(f"SELECT * FROM units WHERE {sql_where} ORDER BY price {order} LIMIT ?", (*args, limit))
    all_prices = [r["price"] for r in radar.rows(f"SELECT price FROM units WHERE {sql_where}", tuple(args))]
    by_project = radar.rows(f"""SELECT project, COUNT(*) n, MIN(price) lo, MAX(price) hi FROM units
                                WHERE {sql_where} GROUP BY project ORDER BY n DESC LIMIT 15""", tuple(args))
    stats = {"count": len(all_prices),
             "min": min(all_prices) if all_prices else None,
             "median": statistics.median(all_prices) if all_prices else None,
             "max": max(all_prices) if all_prices else None,
             "by_project": by_project}
    return rows, stats


def summary_text(stats: dict) -> str:
    if not stats["count"]:
        return "0 units match."
    lines = [f"{stats['count']} units match. Price range EGP {stats['min']:,.0f} – {stats['max']:,.0f} "
             f"(median {stats['median']:,.0f})."]
    for p in stats["by_project"]:
        lines.append(f"- {p['project']}: {p['n']} units, EGP {p['lo']:,.0f} – {p['hi']:,.0f}")
    return "\n".join(lines)


def unit_line(u: dict) -> str:
    beds = "Studio" if not u.get("bedrooms") else f"{u['bedrooms']}BR"
    return (f"{u['unit_id']} | {u['project']} | {u['type']} | {beds} | {u['area']:g} sqm | "
            f"EGP {u['price']:,.0f} | {u['status']} | updated {str(u['updated_at'])[:10]}")


def unit_details(unit_id: str) -> str | None:
    rows = radar.rows("SELECT * FROM units WHERE unit_id=? COLLATE NOCASE", (unit_id.strip(),))
    if not rows:
        return None
    u = rows[0]
    lines = [unit_line(u), f"Location: {u['location']} | Delivery: {u.get('delivery_year') or '-'}"]
    hist = radar.price_history(u["unit_id"])[:10]
    if hist:
        lines.append("Price history:")
        for h in hist:
            old = f"{h['old_price']:,.0f}" if h["old_price"] else "-"
            lines.append(f"  {h['changed_at'][:10]}: {old} → {h['new_price']:,.0f} ({h['new_status']}) [{h['source']}]")
    return "\n".join(lines)


def projects_overview() -> str:
    rows = radar.rows("""SELECT project, location, COUNT(*) total, SUM(status='Available') available,
                                MIN(CASE WHEN status='Available' THEN price END) lo,
                                MAX(CASE WHEN status='Available' THEN price END) hi
                         FROM units GROUP BY project, location ORDER BY available DESC""")
    if not rows:
        return "No units yet."
    return "\n".join(f"{r['project']} ({r['location']}): {r['available']}/{r['total']} available"
                     + (f", EGP {r['lo']:,.0f} – {r['hi']:,.0f}" if r["lo"] else "") for r in rows)


def list_units(question: str, limit: int = 15) -> str:
    """No LLM: the raw matching rows, for /units."""
    f = filters_from(question)
    rows, stats = query(f, limit)
    out = [summary_text(stats)]
    out += [unit_line(u) for u in rows]
    if stats["count"] > len(rows):
        out.append(f"… +{stats['count'] - len(rows)} more. Narrow the search (area, type, bedrooms, budget).")
    return "\n".join(out)


class SalesUnitsAssistant:
    def __init__(self, llm):
        self.chain = SALES_PROMPT | llm | StrOutputParser()

    def answer(self, question: str) -> dict:
        ids = re.findall(r"\b[A-Za-z]{1,6}-\d{1,5}[A-Za-z]?\b", question)
        if ids and re.search(r"(history|تاريخ|اتغير|changes)", question, re.I):
            details = [unit_details(i) for i in ids]
            text = "\n\n".join(d for d in details if d) or "Unit not found."
            return {"answer": text, "units": [], "stats": None}
        f = filters_from(question)
        rows, stats = query(f)
        shown = {k: v for k, v in f.items() if v not in (None, "", []) and k != "sort"}
        answer = self.chain.invoke({
            "filters": shown or "(none — whole inventory)",
            "summary": summary_text(stats),
            "rows": "\n".join(unit_line(u) for u in rows) or "(no rows)",
            "question": question,
        }).strip()
        return {"answer": answer, "units": rows, "stats": stats}


def build_sales_assistant(offline: bool | None = None) -> SalesUnitsAssistant:
    from . import config
    from .llm import get_llm
    offline = config.OFFLINE_MODE if offline is None else offline
    return SalesUnitsAssistant(get_llm(offline))
