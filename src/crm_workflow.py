"""LangGraph orchestration for RADAR's evidence-first opportunity workflow.

Each node has one job and returns explicit state to the next node:
intake -> deterministic matching -> opportunity qualification -> recommendation -> persistence.
The matching rules and policy controls stay in Python so an LLM cannot override them.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import radar

logger = logging.getLogger(__name__)


class CRMState(TypedDict, total=False):
    db: Any
    unit_key: str | None
    trigger: str
    units: list[dict]
    leads: list[dict]
    matches: list[dict]
    qualified: list[dict]
    prepared: list[dict]
    created: int
    stopped: str


def _log(node: str, **fields: Any) -> None:
    logger.info("CRM agent node=%s details=%s", node, fields)


def intake_agent(state: CRMState) -> dict:
    """Load only active, available CRM records and honor the global kill switch."""
    db = state["db"]
    settings = {r["key"]: r["value"] for r in db.execute("SELECT * FROM settings")}
    if settings.get("kill_switch") == "1":
        _log("intake", result="stopped", reason="kill_switch")
        return {"units": [], "leads": [], "stopped": "Agent actions are paused by the kill switch."}
    sql = "SELECT * FROM units WHERE status='Available'"
    args: tuple = ()
    if state.get("unit_key"):
        sql += " AND unit_id=?"
        args = (state["unit_key"],)
    units = [dict(r) for r in db.execute(sql, args).fetchall()]
    leads = [dict(r) for r in db.execute("SELECT * FROM leads WHERE status NOT IN ('Closed','Lost')").fetchall()]
    _log("intake", available_units=len(units), active_leads=len(leads))
    return {"units": units, "leads": leads}


def matching_agent(state: CRMState) -> dict:
    """Apply eligibility filters and calculate a transparent deterministic fit score."""
    matches = []
    for unit in state.get("units", []):
        for lead in state.get("leads", []):
            location = (unit.get("location") or "").lower()
            desired_area = (lead.get("area") or "").lower().strip()
            if lead.get("budget") and unit.get("price", 0) > lead["budget"] * 1.10:
                continue
            if desired_area and desired_area not in location and not any(
                word in location for word in desired_area.split()
            ):
                continue
            if lead.get("bedrooms") is not None and unit.get("bedrooms") is not None and unit["bedrooms"] < lead["bedrooms"]:
                continue
            parts = {
                "budget": 30 if not lead.get("budget") or unit["price"] <= lead["budget"] else 18,
                "location": 25 if not desired_area or desired_area in location else 0,
                "bedrooms": 20 if lead.get("bedrooms") is None or unit.get("bedrooms", 0) >= lead["bedrooms"] else 0,
                "type": 10 if not lead.get("unit_type") or lead["unit_type"].lower() in (unit.get("type") or "").lower() else 5,
                "intent": 10 if lead.get("status") in ("Interested", "Viewing", "Contacted") else 5,
                "freshness": 5,
            }
            why = [
                f"Budget: EGP {unit['price']:,.0f} (lead budget EGP {lead['budget']:,.0f})" if lead.get("budget") else "Budget not recorded",
                f"Location: {unit.get('location') or 'not recorded'}",
                f"{unit.get('bedrooms') or 'Unspecified'} bedrooms, {unit.get('area') or 0:g} sqm",
            ]
            if lead.get("objection"):
                why.append(f"Buyer note: {lead['objection']}")
            matches.append({"lead": lead, "unit": unit, "score": min(100, sum(parts.values())), "factors": parts, "why": why})
    _log("matching", eligible_pairs=len(matches))
    return {"matches": matches}


def opportunity_agent(state: CRMState) -> dict:
    """Apply event-trigger rules, dormant-lead age checks, and stable deduplication keys."""
    db = state["db"]
    trigger = state.get("trigger", "scan")
    qualified = []
    # A dormant scan evaluates many unit/lead pairs. Fetch each lead's latest
    # activity once instead of issuing the same query for every matching unit.
    latest_by_lead = {}
    if trigger == "scan":
        latest_by_lead = {
            row["lead_id"]: row["latest"]
            for row in db.execute(
                "SELECT lead_id, MAX(created_at) AS latest FROM events "
                "WHERE lead_id IS NOT NULL GROUP BY lead_id"
            ).fetchall()
        }
    for match in state.get("matches", []):
        lead, unit = match["lead"], match["unit"]
        lead_trigger = trigger
        if trigger == "scan":
            latest = latest_by_lead.get(lead["id"])
            if latest and datetime.fromisoformat(latest) > datetime.now(timezone.utc) - timedelta(days=21):
                continue
            lead_trigger = "dormant_reactivation"
        reasons = {
            "factors": match["factors"],
            "why": match["why"],
            "evidence": lead.get("objection") or lead.get("notes") or "Lead profile",
            "unit_verified": unit.get("updated_at"),
        }
        version = f"{unit['unit_id']}:{unit['price']}:{unit['status']}:{unit.get('down_payment')}:{unit['updated_at']}"
        qualified.append({**match, "trigger": lead_trigger, "trigger_version": version, "reasons": reasons})
    _log("opportunity", qualified_pairs=len(qualified), trigger=trigger)
    return {"qualified": qualified}


def recommendation_agent(state: CRMState) -> dict:
    """Prepare a conservative, localized draft and suggested next step for human review."""
    prepared = []
    for item in state.get("qualified", []):
        lead, unit = item["lead"], item["unit"]
        if (lead.get("language") or "English").lower().startswith("arab"):
            draft = f"أهلاً {lead['name']}، بناءً على اهتمامك السابق، عندنا خيار مناسب في {unit['project']} بسعر {unit['price']:,.0f} جنيه. هل تحب نراجع التفاصيل؟"
        else:
            draft = f"Hi {lead['name']}, based on your earlier interest, there is a relevant option at {unit['project']} for EGP {unit['price']:,.0f}. Would you like me to share the details?"
        prepared.append({**item, "action": "Call" if lead.get("objection") else "Send brochure", "draft": draft})
    _log("recommendation", drafts_prepared=len(prepared))
    return {"prepared": prepared}


def persist_opportunities(state: CRMState) -> dict:
    """Save unique recommendations and append an audit event; never contact a lead."""
    db = state["db"]
    created = 0
    for item in state.get("prepared", []):
        lead, unit = item["lead"], item["unit"]
        cur = db.execute(
            """INSERT OR IGNORE INTO opportunities
            (lead_id,unit_id,trigger,trigger_version,score,reasons,action,draft,created_at)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (lead["id"], unit["id"], item["trigger"], item["trigger_version"], item["score"],
             json.dumps(item["reasons"], ensure_ascii=False), item["action"], item["draft"], radar.now()),
        )
        if cur.rowcount:
            created += 1
            db.execute(
                "INSERT INTO events(lead_id,unit_id,kind,details,created_at) VALUES(?,?,?,?,?)",
                (lead["id"], unit["id"], "Opportunity created", f"{item['trigger']}; fit score {item['score']}", radar.now()),
            )
    _log("persistence", created=created, duplicates_skipped=len(state.get("prepared", [])) - created)
    return {"created": created}


def build_opportunity_graph():
    """Build the compiled sequential multi-agent workflow."""
    graph = StateGraph(CRMState)
    graph.add_node("intake_agent", intake_agent)
    graph.add_node("matching_agent", matching_agent)
    graph.add_node("opportunity_agent", opportunity_agent)
    graph.add_node("recommendation_agent", recommendation_agent)
    graph.add_node("persist_opportunities", persist_opportunities)
    graph.add_edge(START, "intake_agent")
    graph.add_edge("intake_agent", "matching_agent")
    graph.add_edge("matching_agent", "opportunity_agent")
    graph.add_edge("opportunity_agent", "recommendation_agent")
    graph.add_edge("recommendation_agent", "persist_opportunities")
    graph.add_edge("persist_opportunities", END)
    return graph.compile()


_OPPORTUNITY_GRAPH = build_opportunity_graph()


def run_opportunity_workflow(db, unit_key: str | None = None, trigger: str = "scan") -> int:
    """Execute the graph on the caller's DB connection and return new-opportunity count."""
    result = _OPPORTUNITY_GRAPH.invoke({"db": db, "unit_key": unit_key, "trigger": trigger})
    if result.get("stopped"):
        logger.info("CRM workflow stopped: %s", result["stopped"])
    return result.get("created", 0)
