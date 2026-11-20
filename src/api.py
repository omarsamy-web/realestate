"""RADAR HTTP API (Phase 1) — replaces the Streamlit dashboard.

Run:  python main.py serve        (or: uvicorn src.api:app --reload)
Docs: http://localhost:8000/docs  (interactive Swagger UI for every endpoint)

The same service will back the client chatbot and the sales assistant (phases 3-4).
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from . import config, inventory, price_ingest, radar

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- auth
def require_api_key(x_api_key: str | None = Header(default=None)):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(401, "Missing or wrong X-API-Key header")


@asynccontextmanager
async def lifespan(_app):
    radar.init_db()
    yield


app = FastAPI(title="RADAR Real Estate Agent API", version="3.0", lifespan=lifespan)
staff = [Depends(require_api_key)]


# --------------------------------------------------------------------------- models
class LeadIn(BaseModel):
    name: str
    phone: str = ""
    language: str = "Arabic"
    budget: float | None = None
    area: str | None = None
    bedrooms: int | None = None
    unit_type: str | None = None
    payment_preference: str | None = None
    monthly_capacity: float | None = None
    objection: str | None = None
    notes: str = ""
    status: str = "New"
    consent: bool = False


class LeadPatch(BaseModel):
    status: str | None = None
    consent: bool | None = None


class NoteIn(BaseModel):
    note: str


class UnitPatch(BaseModel):
    price: float | None = None
    status: str | None = None


class DecisionIn(BaseModel):
    decision: str = Field(description="Approved, Snoozed or Dismissed")
    reason: str = ""


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


class TextIngestIn(BaseModel):
    text: str
    source: str = "api"
    message_id: str | None = None


class WebhookIn(BaseModel):
    text: str = ""
    source: str = "webhook"
    message_id: str | None = None
    chat: str | None = None


class PendingDecisionIn(BaseModel):
    approve: bool


class SettingsIn(BaseModel):
    kill_switch: bool | None = None
    autonomy: int | None = None


# --------------------------------------------------------------------------- health
@app.get("/health")
def health():
    counts = radar.rows("""SELECT (SELECT COUNT(*) FROM units) units,
                                  (SELECT COUNT(*) FROM units WHERE status='Available') available,
                                  (SELECT COUNT(*) FROM leads) leads,
                                  (SELECT MAX(changed_at) FROM price_history) last_price_change""")[0]
    return {"ok": True, **counts}


# --------------------------------------------------------------------------- units
@app.get("/units", dependencies=staff)
def list_units(project: str | None = None, location: str | None = None, type: str | None = None,
               bedrooms: int | None = None, min_price: float | None = None, max_price: float | None = None,
               status: str | None = "Available", limit: int = Query(100, le=5000)):
    return inventory.search_units(projects=[project] if project else None, location=location, type=type,
                                  bedrooms=bedrooms, min_price=min_price, max_price=max_price,
                                  status=status, limit=limit)


@app.get("/units/{unit_id}", dependencies=staff)
def get_unit(unit_id: str):
    rows = radar.rows("SELECT * FROM units WHERE unit_id=?", (unit_id,))
    if not rows:
        raise HTTPException(404, "Unit not found")
    return {**rows[0], "price_history": radar.price_history(unit_id)}


@app.get("/units/{unit_id}/history", dependencies=staff)
def unit_history(unit_id: str):
    return radar.price_history(unit_id)


@app.patch("/units/{unit_id}", dependencies=staff)
def patch_unit(unit_id: str, body: UnitPatch):
    try:
        created = radar.update_unit(unit_id, body.price, body.status, source="api")
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"opportunities_created": created, **get_unit(unit_id)}


# --------------------------------------------------------------------------- leads
@app.get("/leads", dependencies=staff)
def list_leads(status: str | None = None):
    if status:
        return radar.rows("SELECT * FROM leads WHERE status=? ORDER BY id DESC", (status,))
    return radar.rows("SELECT * FROM leads ORDER BY id DESC")


@app.post("/leads", dependencies=staff, status_code=201)
def create_lead(body: LeadIn):
    try:
        lead_id = radar.add_lead(body.model_dump())
    except Exception as e:
        raise HTTPException(400, str(e))
    return get_lead(lead_id)


@app.get("/leads/{lead_id}", dependencies=staff)
def get_lead(lead_id: int):
    rows = radar.rows("SELECT * FROM leads WHERE id=?", (lead_id,))
    if not rows:
        raise HTTPException(404, "Lead not found")
    events = radar.rows("SELECT * FROM events WHERE lead_id=? ORDER BY created_at DESC", (lead_id,))
    return {**rows[0], "events": events}


@app.patch("/leads/{lead_id}", dependencies=staff)
def patch_lead(lead_id: int, body: LeadPatch):
    get_lead(lead_id)
    if body.status and body.status not in radar.STATUSES:
        raise HTTPException(400, f"status must be one of {radar.STATUSES}")
    with radar.connect() as db:
        if body.status:
            db.execute("UPDATE leads SET status=? WHERE id=?", (body.status, lead_id))
            db.execute("INSERT INTO events(lead_id,kind,details,created_at) VALUES(?,?,?,?)",
                       (lead_id, "Status changed", body.status, radar.now()))
        if body.consent is not None:
            db.execute("UPDATE leads SET consent=? WHERE id=?", (int(body.consent), lead_id))
    return get_lead(lead_id)


@app.post("/leads/{lead_id}/notes", dependencies=staff)
def add_note(lead_id: int, body: NoteIn):
    get_lead(lead_id)
    radar.add_note(lead_id, body.note)
    return get_lead(lead_id)


# --------------------------------------------------------------------------- opportunities
@app.get("/opportunities", dependencies=staff)
def list_opportunities(status: str | None = "Pending"):
    sql = """SELECT o.*, l.name lead_name, l.phone, l.consent, u.unit_id unit_code, u.project, u.price
             FROM opportunities o JOIN leads l ON l.id=o.lead_id JOIN units u ON u.id=o.unit_id"""
    if status:
        return radar.rows(sql + " WHERE o.status=? ORDER BY o.score DESC", (status,))
    return radar.rows(sql + " ORDER BY o.created_at DESC")


@app.post("/opportunities/scan", dependencies=staff)
def scan_opportunities():
    return {"created": radar.scan_dormant()}


@app.post("/opportunities/{opp_id}/decision", dependencies=staff)
def decide(opp_id: int, body: DecisionIn):
    try:
        radar.decide_opportunity(opp_id, body.decision, body.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": opp_id, "status": body.decision}


# --------------------------------------------------------------------------- chat (RAG + live prices)
_sessions: dict[str, tuple] = {}
_sessions_lock = threading.Lock()
_shared: dict = {}


def _shared_parts():
    if "llm" not in _shared:
        from .indexer import load_index
        from .llm import check_llm, get_llm
        offline = config.OFFLINE_MODE
        if not offline:
            check_llm()
        _shared["llm"] = get_llm(offline)
        _shared["store"] = load_index(offline)
    return _shared["store"], _shared["llm"]


def reset_chat_cache():
    _shared.clear()
    with _sessions_lock:
        _sessions.clear()


def _assistant(session_id: str):
    from .memory import BufferMemory
    from .rag import RealEstateAssistant
    now = time.time()
    ttl = config.CHAT_SESSION_TTL_MIN * 60
    with _sessions_lock:
        for sid in [s for s, (_, t) in _sessions.items() if now - t > ttl]:
            del _sessions[sid]
        if session_id not in _sessions:
            store, llm = _shared_parts()
            _sessions[session_id] = (RealEstateAssistant(store, llm, BufferMemory()), now)
        bot = _sessions[session_id][0]
        _sessions[session_id] = (bot, now)
    return bot


@app.post("/chat", dependencies=staff)
def chat(body: ChatIn):
    session_id = body.session_id or uuid.uuid4().hex
    try:
        bot = _assistant(session_id)
    except (FileNotFoundError, SystemExit) as e:
        raise HTTPException(503, f"Chat not ready: {e}. Run: python main.py build")
    answer, docs, standalone = bot.ask(body.message)
    sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page"), "row": d.metadata.get("row")}
               for d in docs]
    return {"session_id": session_id, "answer": answer, "searched_for": standalone,
            "sources": sources, "live_units": bot.last_units}


@app.delete("/chat/{session_id}", dependencies=staff)
def reset_chat(session_id: str):
    with _sessions_lock:
        _sessions.pop(session_id, None)
    return {"ok": True}


# --------------------------------------------------------------------------- price ingestion (Phase 2)
def _llm_for_ingest():
    if not config.INGEST_USE_LLM_FALLBACK or config.OFFLINE_MODE:
        return None
    try:
        return _shared_parts()[1]
    except BaseException:
        return None


@app.post("/ingest/text", dependencies=staff)
def ingest_text(body: TextIngestIn):
    key = f"{body.source}:{body.message_id}" if body.message_id else None
    result = price_ingest.ingest_text(body.text, source=body.source, message_key=key, llm=_llm_for_ingest())
    return {**result.as_dict(), "summary": result.summary("ar")}


@app.post("/ingest/file", dependencies=staff)
async def ingest_file(file: UploadFile = File(...), source: str = Form("upload")):
    data = await file.read()
    try:
        result = price_ingest.ingest_file(data, file.filename or "upload.xlsx", source=source)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {**result.as_dict(), "summary": result.summary("ar")}


@app.post("/ingest/webhook")
def ingest_webhook(body: WebhookIn, x_ingest_token: str | None = Header(default=None)):
    """For WhatsApp/Telegram automations (n8n, Make, Zapier, a forwarding bot).

    Send JSON {"text": "...", "source": "whatsapp", "message_id": "...", "chat": "..."}
    with header X-Ingest-Token equal to INGEST_WEBHOOK_TOKEN."""
    if not config.INGEST_WEBHOOK_TOKEN:
        raise HTTPException(503, "Set INGEST_WEBHOOK_TOKEN in .env to enable the webhook")
    if x_ingest_token != config.INGEST_WEBHOOK_TOKEN:
        raise HTTPException(401, "Wrong X-Ingest-Token")
    key = f"{body.source}:{body.chat or ''}:{body.message_id}" if body.message_id else None
    result = price_ingest.ingest_text(body.text, source=body.source, message_key=key, llm=_llm_for_ingest())
    return {**result.as_dict(), "summary": result.summary("ar")}


@app.post("/documents", dependencies=staff)
async def upload_document(file: UploadFile = File(...), source: str = Form("upload")):
    """Any daily file (PDF, Excel, CSV, Word, TXT, MD, JSON): saved, indexed, prices applied."""
    from . import documents
    data = await file.read()
    try:
        result = documents.ingest_document(data, file.filename or "document", source=source)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {**result.as_dict(), "summary": result.summary("ar")}


@app.get("/documents", dependencies=staff)
def list_documents(limit: int = 50):
    from . import documents
    return documents.list_documents(limit)


@app.get("/ingest/runs", dependencies=staff)
def ingest_runs(limit: int = 50):
    return radar.rows("SELECT id,source,message_key,received,applied,held,rejected,created_at FROM ingest_runs "
                      "ORDER BY id DESC LIMIT ?", (limit,))


@app.get("/ingest/pending", dependencies=staff)
def pending_updates():
    return price_ingest.list_pending()


@app.post("/ingest/pending/{pending_id}/decision", dependencies=staff)
def decide_pending(pending_id: int, body: PendingDecisionIn):
    try:
        return price_ingest.decide_pending(pending_id, body.approve)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/index/rebuild", dependencies=staff)
def rebuild_index():
    chunks = price_ingest.rebuild_index_now()
    reset_chat_cache()
    return {"chunks": chunks}


# --------------------------------------------------------------------------- client chatbot (Phase 3)
class ClientChatIn(BaseModel):
    user_id: str = Field(description="Your stable id for this client (website session, WhatsApp number...)")
    message: str
    channel: str = "web"
    name: str | None = None


_client_agent: dict = {}


def require_client_key(x_client_key: str | None = Header(default=None)):
    if config.CLIENT_API_KEY and x_client_key != config.CLIENT_API_KEY:
        raise HTTPException(401, "Missing or wrong X-Client-Key header")


def get_client_agent():
    if "agent" not in _client_agent:
        from .client_agent import build_client_agent
        from .telegram_bot import sales_notifier
        _client_agent["agent"] = build_client_agent(notify=sales_notifier() if config.CLIENT_SAVE_LEADS else None)
    return _client_agent["agent"]


@app.post("/client/chat", dependencies=[Depends(require_client_key)])
def client_chat(body: ClientChatIn):
    """Public chat for buyers (website widget, WhatsApp bridge...). Answers only from company data."""
    try:
        agent = get_client_agent()
    except SystemExit as e:
        raise HTTPException(503, str(e))
    return agent.handle(body.channel, body.user_id, body.message, name=body.name)


@app.delete("/client/chat/{channel}/{user_id}", dependencies=[Depends(require_client_key)])
def client_reset(channel: str, user_id: str):
    get_client_agent().reset(channel, user_id)
    return {"ok": True}


# --------------------------------------------------------------------------- sales assistant (Phase 4)
_clustering_state = {"running": False, "last": None, "error": None}


def _run_clustering(notes_path, k):
    from . import lead_clustering
    try:
        result = lead_clustering.run_clustering(notes_path, k)
        _clustering_state.update(last={"leads": result["leads"], "k": result["k"], "at": radar.now()}, error=None)
    except Exception as e:  # surfaced on GET /sales/clusters/status
        logger.exception("Clustering failed")
        _clustering_state["error"] = str(e)
    finally:
        _clustering_state["running"] = False


@app.post("/sales/clusters/run", dependencies=staff)
def run_clusters(include_notes_file: bool = True, k: int | None = None):
    """Rebuild lead profiles, clusters and priorities in the background (about 2 minutes for 100k notes)."""
    if _clustering_state["running"]:
        return {"started": False, "reason": "already running"}
    from pathlib import Path as _P
    notes = config.SALES_NOTES_FILE if include_notes_file and _P(config.SALES_NOTES_FILE).exists() else None
    _clustering_state["running"] = True
    threading.Thread(target=_run_clustering, args=(notes, k), daemon=True).start()
    return {"started": True, "notes_file": notes}


@app.get("/sales/clusters/status", dependencies=staff)
def clusters_status():
    return _clustering_state


@app.get("/sales/clusters", dependencies=staff)
def get_clusters():
    from . import lead_clustering
    return lead_clustering.list_clusters()


@app.get("/sales/priority", dependencies=staff)
def get_priority(limit: int = Query(20, le=500), cluster: int | None = None, agent: str | None = None):
    from . import lead_clustering
    return lead_clustering.priority_list(limit, cluster, agent)


@app.get("/sales/leads/{key}", dependencies=staff)
def get_lead_profile(key: str):
    from . import lead_clustering
    p = lead_clustering.get_profile(key)
    if not p:
        raise HTTPException(404, "Lead profile not found (run POST /sales/clusters/run first)")
    return p


@app.get("/sales/leads/{key}/recommendations", dependencies=staff)
def get_recommendations(key: str, k: int = Query(5, le=20)):
    from . import sales_agent
    try:
        return sales_agent.recommend_for(key, k)
    except ValueError as e:
        raise HTTPException(404, str(e))


class SalesAskIn(BaseModel):
    question: str
    session_id: str = "api"


_sales_units: dict = {}


@app.post("/sales/ask", dependencies=staff)
def sales_ask(body: SalesAskIn):
    """Sales inventory assistant: full access to the units table."""
    from .sales_units import build_sales_assistant
    if "bot" not in _sales_units:
        _sales_units["bot"] = build_sales_assistant()
    return _sales_units["bot"].answer(body.question, session=body.session_id)


@app.get("/sales/units", dependencies=staff)
def sales_units_search(q: str = "", limit: int = Query(50, le=500)):
    from . import sales_units
    rows, stats = sales_units.query(sales_units.filters_from(q), limit)
    return {"stats": stats, "units": rows}


# --------------------------------------------------------------------------- research lab
@app.post("/research", dependencies=staff)
async def research(file: UploadFile = File(...), task: str = Form("")):
    from .pdf_service import PDFExtractionError, extract_pdf
    from .report_generator import save_report
    from .research_workflow import run_research_workflow
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File limit is {config.MAX_UPLOAD_MB} MB")
    try:
        document = extract_pdf(data, file.filename or "upload.pdf")
    except PDFExtractionError as e:
        raise HTTPException(400, str(e))
    result = run_research_workflow(document, task)
    path = save_report(file.filename or "upload.pdf", result["report"], result.get("research_notes", ""),
                       result.get("analysis", ""), task)
    return {"report": result["report"], "report_path": str(path)}


@app.get("/reports", dependencies=staff)
def reports():
    from .report_generator import list_reports
    return list_reports()


# --------------------------------------------------------------------------- settings + audit
@app.get("/settings", dependencies=staff)
def get_settings():
    return radar.settings()


@app.put("/settings", dependencies=staff)
def put_settings(body: SettingsIn):
    if body.kill_switch is not None:
        radar.update_setting("kill_switch", int(body.kill_switch))
    if body.autonomy is not None:
        radar.update_setting("autonomy", body.autonomy)
    return radar.settings()


@app.get("/events", dependencies=staff)
def events(limit: int = 50):
    return radar.rows("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
