"""Daily price ingestion: turn group messages and sheets into unit price updates.

Sources: a Telegram group (see telegram_bot.py), a webhook that any WhatsApp
automation (n8n, Make, a forwarding number) can call, the API, or the CLI.

Pipeline for every message:
  parse  -> list of PriceUpdate (rule-based; optional LLM fallback for messy text)
  check  -> unknown units are rejected, big jumps are held for human approval
  apply  -> update units + price_history + CRM opportunities (via radar.update_unit)
  log    -> one row in ingest_runs, keyed by message so a re-sent message is ignored
  index  -> schedule a background rebuild of the RAG index
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field

from . import config, radar
from .inventory import normalize_digits, parse_amount

logger = logging.getLogger(__name__)

STATUS_WORDS = {
    "Sold": ("sold", "مباع", "مباعة", "اتباع", "اتباعت", "بيعت"),
    "Reserved": ("reserved", "hold", "محجوز", "محجوزة", "اتحجز", "اتحجزت"),
    "Available": ("available", "متاح", "متاحة", "متوفر", "متوفرة", "رجع", "back"),
}
UP_WORDS = ("increase", "up", "+", "زيادة", "زياده", "رفع", "ارتفاع")
DOWN_WORDS = ("decrease", "down", "discount", "-", "خصم", "تخفيض", "نزول", "نقص")
MIN_UNIT_PRICE = 100_000  # numbers below this are areas, floors, bedrooms...


@dataclass
class PriceUpdate:
    unit_id: str
    price: float | None = None
    status: str | None = None
    line: str = ""
    # full row, only for new units coming from a spreadsheet
    row: dict | None = None


@dataclass
class IngestResult:
    source: str
    received: int = 0
    applied: list = field(default_factory=list)
    held: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    added: list = field(default_factory=list)
    duplicate: bool = False

    def summary(self, lang: str = "ar") -> str:
        if self.duplicate:
            return "الرسالة دي اتسجلت قبل كده." if lang == "ar" else "This message was already processed."
        if lang == "ar":
            parts = [f"✅ اتحدّث {len(self.applied)} وحدة"]
            if self.added:
                parts.append(f"🆕 اتضاف {len(self.added)} وحدة")
            if self.held:
                parts.append(f"⏸️ {len(self.held)} تغيير كبير مستني موافقة")
            if self.rejected:
                parts.append(f"⚠️ {len(self.rejected)} سطر ما اتفهمش أو الوحدة مش موجودة")
        else:
            parts = [f"✅ {len(self.applied)} units updated"]
            if self.added:
                parts.append(f"🆕 {len(self.added)} units added")
            if self.held:
                parts.append(f"⏸️ {len(self.held)} large changes held for approval")
            if self.rejected:
                parts.append(f"⚠️ {len(self.rejected)} lines not understood or unknown unit")
        lines = [" · ".join(parts)]
        for a in self.applied[:15]:
            old = f"{a['old_price']:,.0f}" if a.get("old_price") else "-"
            new = f"{a['new_price']:,.0f}" if a.get("new_price") else "-"
            status = f" ({a['new_status']})" if a.get("new_status") != a.get("old_status") else ""
            lines.append(f"• {a['unit_id']}: {old} → {new}{status}")
        if len(self.applied) > 15:
            lines.append(f"… +{len(self.applied) - 15}")
        for h in self.held[:5]:
            lines.append(f"⏸️ {h['unit_id']}: {h['reason']}")
        for r in self.rejected[:5]:
            lines.append(f"⚠️ {r['line'][:80]} — {r['reason']}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"source": self.source, "received": self.received, "applied": self.applied,
                "held": self.held, "rejected": self.rejected, "added": self.added,
                "duplicate": self.duplicate}


# --------------------------------------------------------------------------- parsing
def _detect_status(text: str) -> str | None:
    t = text.lower()
    for status, words in STATUS_WORDS.items():
        if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", t) for w in words):
            return status
    return None


def _known_units() -> list[str]:
    return [r["unit_id"] for r in radar.rows("SELECT unit_id FROM units")]


def _find_unit(line: str, units_sorted: list[str]) -> str | None:
    low = line.lower()
    for uid in units_sorted:  # longest first, so "PR-1010" wins over "PR-101"
        if uid.lower() in low and re.search(rf"(?<![\w-]){re.escape(uid.lower())}(?![\w-])", low):
            return uid
    m = re.search(r"\b([A-Za-z]{1,6}-?\d{1,5}[A-Za-z]?)\b", line)
    return m.group(1).upper() if m else None


def _find_price(text: str) -> float | None:
    candidates = []
    for m in re.finditer(r"\d[\d,٬]*(?:\.\d+)?\s*(?:m\b|mn\b|million|مليون|k\b|الف|ألف)?", text.lower()):
        value = parse_amount(m.group(0))
        if value and value >= MIN_UNIT_PRICE:
            candidates.append(value)
    return max(candidates) if candidates else None


def _project_change(line: str, projects: list[str]) -> tuple[str, float] | None:
    """'Palm Residence +5%' / 'زيادة 5% على Palm Residence' -> (project, +5.0)"""
    m = re.search(r"([+-]?\s*\d+(?:\.\d+)?)\s*%", line)
    if not m:
        return None
    project = next((p for p in sorted(projects, key=len, reverse=True) if p and p.lower() in line.lower()), None)
    if not project:
        return None
    raw = m.group(1).replace(" ", "")
    pct = float(raw)
    low = line.lower()
    has_down = any(w in low for w in DOWN_WORDS if w not in ("-", "down"))
    has_up = any(w in low for w in UP_WORDS if w not in ("+", "up"))
    if not (raw[0] in "+-" or has_up or has_down):
        return None  # "10% down payment" is not a price change
    if pct > 0 and has_down and not raw.startswith("+"):
        pct = -pct
    return project, pct


def parse_text(text: str) -> tuple[list[PriceUpdate], list[dict], list[tuple[str, float, str]]]:
    """Rule-based parser. Returns (unit updates, rejected lines, project-wide % changes)."""
    text = normalize_digits(text or "")
    units_sorted = sorted(_known_units(), key=len, reverse=True)
    projects = [r["project"] for r in radar.rows("SELECT DISTINCT project FROM units")]
    updates, rejected, project_changes = [], [], []
    for raw in text.splitlines():
        line = raw.strip(" \t•*-–—|")
        if not line or not re.search(r"\d", line):
            continue
        unit = _find_unit(line, units_sorted)
        if not unit:
            pc = _project_change(line, projects)
            if pc:
                project_changes.append((pc[0], pc[1], line))
            else:
                rejected.append({"line": line, "reason": "no unit code found"})
            continue
        rest = re.sub(re.escape(unit), " ", line, flags=re.I)
        price, status = _find_price(rest), _detect_status(rest)
        if price is None and status is None:
            rejected.append({"line": line, "reason": "no price or status found"})
            continue
        updates.append(PriceUpdate(unit_id=unit, price=price, status=status, line=line))
    return updates, rejected, project_changes


LLM_EXTRACT_PROMPT = (
    "Extract real-estate unit price updates from the message below. It may be Arabic or English.\n"
    "Return ONLY a JSON list like [{{\"unit_id\": \"PR-101\", \"price\": 3450000, \"status\": \"Available\"}}].\n"
    "price is a full number in EGP (convert 'مليون'/'M'), or null. status is one of Available, Reserved, Sold, or null.\n"
    "Only use unit codes that appear in the message. If there are no updates, return [].\n\nMessage:\n{text}"
)


def parse_text_with_llm(text: str, llm) -> list[PriceUpdate]:
    """Fallback for free-form messages. Output is validated before anything is applied."""
    reply = llm.invoke(LLM_EXTRACT_PROMPT.format(text=text[:6000]))
    content = getattr(reply, "content", reply)
    m = re.search(r"\[.*\]", str(content), re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    updates = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not it.get("unit_id"):
            continue
        uid = str(it["unit_id"]).strip()
        if uid.lower() not in text.lower():  # the model must not invent unit codes
            continue
        price = it.get("price")
        status = it.get("status") if it.get("status") in STATUS_WORDS else None
        try:
            price = float(price) if price is not None else None
        except (TypeError, ValueError):
            price = None
        if price is not None and price < MIN_UNIT_PRICE:
            price = None
        if price is not None or status:
            updates.append(PriceUpdate(unit_id=uid, price=price, status=status, line=f"(llm) {uid}"))
    return updates


def parse_sheet(file_bytes: bytes, filename: str) -> list[PriceUpdate]:
    import pandas as pd
    from io import BytesIO, StringIO

    if len(file_bytes) > 50_000_000:
        raise ValueError("File limit is 50 MB.")
    name = filename.lower()
    if name.endswith(".csv"):
        frame = pd.read_csv(StringIO(file_bytes.decode("utf-8-sig")))
    elif name.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(BytesIO(file_bytes))
    else:
        raise ValueError("Send a .csv or .xlsx file.")
    frame = radar._normalize_unit_dataframe(frame)
    if "Unit ID" not in frame.columns or "Price (EGP)" not in frame.columns:
        raise ValueError("The sheet needs a unit column and a price column.")
    frame = frame.dropna(subset=["Unit ID", "Price (EGP)"]).drop_duplicates(subset=["Unit ID"], keep="last")
    updates = []
    for _, r in frame.iterrows():
        row = {k: (None if pd.isna(v) else v) for k, v in r.items()}
        updates.append(PriceUpdate(unit_id=str(row["Unit ID"]).strip(), price=float(row["Price (EGP)"]),
                                   status=str(row.get("Status") or "Available"), line=str(row["Unit ID"]), row=row))
    return updates


# --------------------------------------------------------------------------- applying
def _hold(db, unit, new_price, new_status, reason, source, result):
    db.execute("""INSERT INTO pending_price_updates(unit_id,old_price,new_price,new_status,reason,source,created_at)
                  VALUES(?,?,?,?,?,?,?)""", (unit["unit_id"], unit["price"], new_price, new_status, reason, source, radar.now()))
    result.held.append({"unit_id": unit["unit_id"], "old_price": unit["price"], "new_price": new_price, "reason": reason})


def _insert_unit(db, upd: PriceUpdate, source: str):
    r = upd.row or {}
    db.execute("""INSERT INTO units(unit_id,project,location,type,bedrooms,area,price,status,delivery_year,updated_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?)""",
               (upd.unit_id, str(r.get("Project") or ""), str(r.get("Location") or ""), str(r.get("Type") or "Apartment"),
                int(r.get("Bedrooms") or 0), float(r.get("Area (sqm)") or 0), float(upd.price),
                upd.status or "Available", int(r.get("Delivery Year") or 0) or None, radar.now()))
    radar.record_price_change(db, upd.unit_id, None, float(upd.price), None, upd.status or "Available", source)
    db.execute("INSERT INTO events(unit_id,kind,details,created_at) VALUES((SELECT id FROM units WHERE unit_id=?),?,?,?)",
               (upd.unit_id, "Unit added", f"{upd.unit_id} added from {source}", radar.now()))


def apply_updates(updates: list[PriceUpdate], source: str, result: IngestResult | None = None,
                  hold_pct: float | None = None) -> IngestResult:
    result = result or IngestResult(source=source)
    hold_pct = config.PRICE_CHANGE_HOLD_PCT if hold_pct is None else hold_pct
    result.received += len(updates)
    to_apply = []
    with radar.connect() as db:
        lookup = {r["unit_id"].lower(): dict(r) for r in db.execute("SELECT * FROM units")}
        for upd in updates:
            unit = lookup.get(upd.unit_id.lower())
            if not unit:
                if upd.row is not None and upd.price:
                    _insert_unit(db, upd, source)
                    result.added.append({"unit_id": upd.unit_id, "new_price": upd.price})
                else:
                    result.rejected.append({"line": upd.line or upd.unit_id, "reason": "unknown unit"})
                continue
            new_price = upd.price if upd.price is not None else unit["price"]
            new_status = upd.status or unit["status"]
            if new_price == unit["price"] and new_status == unit["status"]:
                continue
            if unit["price"] and abs(new_price - unit["price"]) / unit["price"] * 100 > hold_pct:
                change = (new_price - unit["price"]) / unit["price"] * 100
                _hold(db, unit, new_price, new_status, f"price change {change:+.1f}% > {hold_pct:g}%", source, result)
                continue
            to_apply.append((unit, new_price, new_status))
    # radar.update_unit opens its own connection and runs the CRM opportunity agents
    for unit, new_price, new_status in to_apply:
        radar.update_unit(unit["unit_id"], new_price, new_status, source=source)
        result.applied.append({"unit_id": unit["unit_id"], "old_price": unit["price"], "new_price": new_price,
                               "old_status": unit["status"], "new_status": new_status})
    return result


def apply_project_change(project: str, pct: float, source: str, result: IngestResult, line: str = "") -> None:
    units = radar.rows("SELECT * FROM units WHERE project=? AND status='Available'", (project,))
    if abs(pct) > config.PRICE_CHANGE_HOLD_PCT:
        with radar.connect() as db:
            for u in units:
                _hold(db, u, round(u["price"] * (1 + pct / 100)), u["status"],
                      f"project-wide change {pct:+g}% > {config.PRICE_CHANGE_HOLD_PCT:g}%", source, result)
        return
    updates = [PriceUpdate(unit_id=u["unit_id"], price=round(u["price"] * (1 + pct / 100)), line=line) for u in units]
    apply_updates(updates, source, result, hold_pct=abs(pct) + 0.01)


# --------------------------------------------------------------------------- entry points
def _already_processed(message_key: str | None) -> bool:
    if not message_key:
        return False
    return bool(radar.rows("SELECT 1 FROM ingest_runs WHERE message_key=?", (message_key,)))


def _log_run(result: IngestResult, message_key: str | None) -> None:
    with radar.connect() as db:
        db.execute("""INSERT OR IGNORE INTO ingest_runs(source,message_key,received,applied,held,rejected,details,created_at)
                      VALUES(?,?,?,?,?,?,?,?)""",
                   (result.source, message_key, result.received, len(result.applied) + len(result.added),
                    len(result.held), len(result.rejected), json.dumps(result.as_dict(), ensure_ascii=False, default=str),
                    radar.now()))
        db.execute("INSERT INTO events(kind,details,created_at) VALUES(?,?,?)",
                   ("Price ingest", f"{result.source}: {len(result.applied)} updated, {len(result.added)} added, "
                                    f"{len(result.held)} held, {len(result.rejected)} rejected", radar.now()))


def ingest_text(text: str, source: str = "manual", message_key: str | None = None, llm=None,
                rebuild: bool = True, project_changes: bool = True) -> IngestResult:
    result = IngestResult(source=source)
    if _already_processed(message_key):
        result.duplicate = True
        return result
    updates, rejected, changes = parse_text(text)
    if not project_changes:
        changes = []
    if not updates and not changes and llm is not None and re.search(r"\d", text or ""):
        updates = parse_text_with_llm(text, llm)
        if updates:
            rejected = []
    result.rejected.extend(rejected)
    apply_updates(updates, source, result)
    for project, pct, line in changes:
        apply_project_change(project, pct, source, result, line)
    _log_run(result, message_key)
    if rebuild and (result.applied or result.added):
        schedule_index_rebuild()
    return result


def ingest_file(file_bytes: bytes, filename: str, source: str = "manual", message_key: str | None = None,
                rebuild: bool = True) -> IngestResult:
    result = IngestResult(source=source)
    if _already_processed(message_key):
        result.duplicate = True
        return result
    apply_updates(parse_sheet(file_bytes, filename), f"{source}:{filename}", result)
    _log_run(result, message_key)
    if rebuild and (result.applied or result.added):
        schedule_index_rebuild()
    return result


def list_pending() -> list[dict]:
    return radar.rows("SELECT * FROM pending_price_updates WHERE status='Pending' ORDER BY created_at DESC")


def decide_pending(pending_id: int, approve: bool, actor: str = "manager") -> dict:
    rows = radar.rows("SELECT * FROM pending_price_updates WHERE id=?", (pending_id,))
    if not rows:
        raise ValueError("Pending update not found")
    p = rows[0]
    if p["status"] != "Pending":
        raise ValueError(f"Already {p['status']}")
    if approve:
        radar.update_unit(p["unit_id"], p["new_price"], p["new_status"], source=f"approved:{p['source']}")
        schedule_index_rebuild()
    with radar.connect() as db:
        db.execute("UPDATE pending_price_updates SET status=? WHERE id=?", ("Approved" if approve else "Rejected", pending_id))
        db.execute("INSERT INTO events(kind,details,created_at) VALUES(?,?,?)",
                   ("Price approval", f"{actor} {'approved' if approve else 'rejected'} {p['unit_id']} -> {p['new_price']:,.0f}", radar.now()))
    return {**p, "status": "Approved" if approve else "Rejected"}


# --------------------------------------------------------------------------- index refresh
_rebuild_lock = threading.Lock()
_rebuild_again = threading.Event()


def rebuild_index_now(offline: bool | None = None) -> int:
    from .indexer import build_index
    from .loaders import load_all

    offline = config.OFFLINE_MODE if offline is None else offline
    docs = load_all()
    build_index(docs, offline=offline)
    logger.info("RAG index rebuilt with %d chunks", len(docs))
    return len(docs)


def _rebuild_worker():
    while True:
        _rebuild_again.clear()
        try:
            rebuild_index_now()
        except Exception:
            logger.exception("Index rebuild failed")
        if not _rebuild_again.is_set():
            break


def schedule_index_rebuild() -> None:
    """Rebuild the RAG index in the background; coalesce bursts of messages into one rebuild.

    Live prices never depend on this: the chat reads prices from the units table.
    The rebuild only keeps document search in step with new units."""
    if not config.AUTO_REBUILD_INDEX:
        return
    _rebuild_again.set()
    if not _rebuild_lock.acquire(blocking=False):
        return

    def run():
        try:
            _rebuild_worker()
        finally:
            _rebuild_lock.release()

    threading.Thread(target=run, name="index-rebuild", daemon=True).start()


def result_json(result: IngestResult) -> str:
    return json.dumps(result.as_dict(), ensure_ascii=False, default=str, indent=2)


__all__ = ["PriceUpdate", "IngestResult", "parse_text", "parse_text_with_llm", "parse_sheet", "apply_updates",
           "ingest_text", "ingest_file", "list_pending", "decide_pending", "schedule_index_rebuild",
           "rebuild_index_now", "asdict"]
