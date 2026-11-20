"""Phase 4: sales assistant — per-lead unit recommendations with reasons.

Recommendations are deterministic and explainable (like the CRM matching agent):
they use the lead profile (budget, area, bedrooms, type, objections) and the
live units table, then add talking points, a next action and a WhatsApp draft.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import lead_clustering, radar

TALKING_POINTS = {
    "price_high": "السعر: الوحدة دي من أرخص الاختيارات اللي تناسب طلبه{drop}.",
    "short_installments": "التقسيط: نظام سداد على {years} سنين.",
    "high_down_payment": "المقدم: ابدأ بأقل مقدم متاح ({dp:g}%).",
    "delivery_late": "الاستلام: {delivery}.",
    "comparing": "المقارنة: جهّز مقارنة بالمشروع التاني (السعر للمتر، الاستلام، السداد).",
    "developer_trust": "الثقة: اعرض مشاريع المطور اللي اتسلمت واعزمه على زيارة للموقع.",
    "maintenance": "الصيانة: وضّح وديعة الصيانة من الأول.",
}


def _recent_drops(days: int = 30) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    drops = {}
    for r in radar.rows("""SELECT unit_id, old_price, new_price FROM price_history
                           WHERE changed_at>=? AND old_price IS NOT NULL AND new_price<old_price""", (since,)):
        drops[r["unit_id"]] = max(drops.get(r["unit_id"], 0), r["old_price"] - r["new_price"])
    return drops


def _candidates(p: dict, budget_cap: float) -> list[dict]:
    sql, args = "SELECT * FROM units WHERE status='Available'", []
    if p.get("location"):
        sql += " AND location=?"
        args.append(p["location"])
    if p.get("bedrooms") is not None:
        sql += " AND bedrooms>=?"
        args.append(int(p["bedrooms"]))
    if p.get("budget"):
        sql += " AND price<=?"
        args.append(p["budget"] * budget_cap)
    return radar.rows(sql + " ORDER BY price LIMIT 400", tuple(args))


def _score(unit: dict, p: dict, drops: dict) -> tuple[int, list[str]]:
    why, score = [], 0
    budget = p.get("budget")
    if not budget:
        score += 15
        why.append("budget not recorded")
    elif unit["price"] <= budget:
        score += 30
        why.append(f"within budget (EGP {unit['price']:,.0f} of {budget:,.0f})")
    else:
        over = (unit["price"] / budget - 1) * 100
        score += 18 if over <= 10 else 8
        why.append(f"{over:.0f}% above budget")
    if p.get("location"):
        score += 25
        why.append(f"in {unit['location']}")
    beds = p.get("bedrooms")
    if beds is not None:
        if (unit["bedrooms"] or 0) == beds:
            score += 20
            why.append(f"{beds} bedrooms as requested")
        else:
            score += 12
            why.append(f"{unit['bedrooms']} bedrooms (asked {beds})")
    if p.get("unit_type") and p["unit_type"].lower() in (unit["type"] or "").lower():
        score += 10
        why.append(f"{unit['type']} as requested")
    if unit["unit_id"] in drops:
        score += 10
        why.append(f"price dropped EGP {drops[unit['unit_id']]:,.0f} in the last 30 days")
    objections = p.get("objections") or []
    if "short_installments" in objections and (unit.get("years") or 0) >= 8:
        score += 5
    if "high_down_payment" in objections and (unit.get("down_payment") or 100) <= 10:
        score += 5
    return min(100, score), why


def _talking_points(unit: dict, p: dict, drops: dict) -> list[str]:
    pts = []
    for obj in (p.get("objections") or [])[:3]:
        tpl = TALKING_POINTS.get(obj)
        if not tpl:
            continue
        drop = f"، ونزل {drops[unit['unit_id']]:,.0f} جنيه الشهر ده" if unit["unit_id"] in drops else ""
        pts.append(tpl.format(drop=drop, years=unit.get("years") or 8, dp=unit.get("down_payment") or 10,
                              delivery=f"تسليم {unit['delivery_year']}" if unit.get("delivery_year") else "اسأل عن ميعاد التسليم"))
    return pts


def next_action(p: dict) -> str:
    if p.get("lost"):
        return "Lost: only re-contact for a clearly better offer."
    if p.get("unreachable") and p.get("notes_count") and p["unreachable"] / p["notes_count"] > 0.5:
        return "Hard to reach: send the WhatsApp draft first, then call at a different time."
    if not p.get("budget") or not p.get("location"):
        return "Qualify first: confirm area and budget on a short call."
    if p.get("positive"):
        return "Warm lead: call today, send 2-3 units and book a site visit."
    if p.get("timing") == "later":
        return "Not in a hurry: send options now and schedule a follow-up next month."
    return "Call and share the top matching units."


def whatsapp_draft(p: dict, units: list[dict]) -> str:
    name = p.get("name") if p.get("name") and not str(p.get("name")).startswith("Lead ") else ""
    hi = f"أهلاً {name}،" if name else "أهلاً بحضرتك،"
    if not units:
        return f"{hi} حابين نطمن على طلبك. ممكن تأكدلي المنطقة والميزانية عشان أبعتلك أنسب الوحدات المتاحة؟"
    lines = [f"{hi} لقيتلك اختيارات مناسبة لطلبك:"]
    for u in units[:3]:
        beds = "استوديو" if not u["bedrooms"] else f"{u['bedrooms']} غرف"
        lines.append(f"• {u['project']} ({u['location']}) {beds}، {u['area']:g} م، بسعر {u['price']:,.0f} جنيه")
    lines.append("تحب أبعتلك التفاصيل وخطة السداد أو نحدد معاد معاينة؟")
    return "\n".join(lines)


def recommend(profile: dict, k: int = 5) -> dict:
    drops = _recent_drops()
    units = _candidates(profile, 1.10)
    relaxed = False
    if not units and profile.get("budget"):
        units, relaxed = _candidates(profile, 1.25), True
    scored = []
    for u in units:
        s, why = _score(u, profile, drops)
        scored.append({**u, "fit_score": s, "why": why, "talking_points": _talking_points(u, profile, drops)})
    scored.sort(key=lambda u: (-u["fit_score"], u["price"]))
    top = scored[:k]
    return {"lead": {k_: profile.get(k_) for k_ in ("lead_key", "name", "phone", "budget", "location", "bedrooms",
                                                   "unit_type", "payment", "objections", "priority", "cluster")},
            "units": top, "relaxed_budget": relaxed, "next_action": next_action(profile),
            "whatsapp_draft": whatsapp_draft(profile, top)}


def recommend_for(key: str, k: int = 5) -> dict:
    profile = lead_clustering.get_profile(key)
    if profile is None:
        # CRM lead that has not been profiled yet
        crm = radar.rows("SELECT * FROM leads WHERE id=? OR phone=? OR name LIKE ? LIMIT 1",
                         (key if str(key).isdigit() else -1, key, f"%{key}%"))
        if not crm:
            raise ValueError(f"No lead found for '{key}'")
        lead = crm[0]
        profile = {"lead_key": f"crm-{lead['id']}", "name": lead["name"], "phone": lead["phone"],
                   "budget": lead["budget"], "location": lead["area"], "bedrooms": lead["bedrooms"],
                   "unit_type": lead["unit_type"], "objections": [], "positive": 0, "notes_count": 0}
    return recommend(profile, k)


def format_recommendation(rec: dict) -> str:
    lead = rec["lead"]
    head = (f"👤 {lead.get('name') or lead.get('lead_key')} | {lead.get('location') or '?'} | "
            f"{lead.get('bedrooms') if lead.get('bedrooms') is not None else '?'} غرف | "
            f"ميزانية {lead['budget']:,.0f}" if lead.get("budget") else
            f"👤 {lead.get('name') or lead.get('lead_key')} | {lead.get('location') or '?'} | ميزانية غير معروفة")
    lines = [head]
    if lead.get("objections"):
        lines.append("اعتراضات: " + "، ".join(lead["objections"]))
    if rec["relaxed_budget"]:
        lines.append("⚠️ مافيش وحدات في حدود الميزانية، دي اختيارات أعلى شوية.")
    if not rec["units"]:
        lines.append("مافيش وحدات متاحة مناسبة دلوقتي.")
    for u in rec["units"]:
        lines.append(f"\n🏠 {u['unit_id']} — {u['project']} — {u['price']:,.0f} جنيه (fit {u['fit_score']})")
        lines.append("   " + "; ".join(u["why"]))
        lines += [f"   💬 {t}" for t in u["talking_points"]]
    lines.append(f"\n➡️ {rec['next_action']}")
    lines.append(f"\n✉️ رسالة واتساب مقترحة:\n{rec['whatsapp_draft']}")
    return "\n".join(lines)
