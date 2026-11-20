"""Phase 4: CRM lead profiles, clustering and call priority.

1. Profiles: every lead's notes (sales_notes.xlsx or any notes export, plus CRM
   leads and client-bot leads) are turned into structured fields: budget,
   location, bedrooms, type, payment, down payment, objections, engagement,
   last contact, and how many available units match right now.
2. Clusters: KMeans groups similar leads. Each cluster gets a readable label
   ("New Cairo · Apartment · 3BR · ~6.2M · installments") and a sales playbook.
3. Priority: a transparent 0-100 score for who sales should call first.

Run:  python main.py profile-leads data/xlsx/sales_notes.xlsx
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import radar
from .requirements_extract import extract, outcome_signals

logger = logging.getLogger(__name__)

PLAYBOOK = {
    "price_high": "Lead with recent price drops and the cheapest matching units; show total cost per sqm.",
    "short_installments": "Offer the longest payment plans first (8+ years) and quarterly options.",
    "high_down_payment": "Pitch low down-payment offers (5-10%) and spread-down-payment plans.",
    "delivery_late": "Prioritise ready-to-move or earliest-delivery units.",
    "maintenance": "Explain maintenance deposit terms upfront; compare with competitor fees.",
    "comparing": "Send a side-by-side comparison with the competing project; stress developer track record.",
    "developer_trust": "Share delivered projects, site-visit invitation, and contract guarantees.",
    "location": "Suggest closer alternatives or explain upcoming road/infrastructure access.",
}


def _ensure_tables():
    with radar.connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS lead_profiles (
          lead_key TEXT PRIMARY KEY, crm_lead_id INTEGER, name TEXT, phone TEXT, agent TEXT,
          budget REAL, budget_min REAL, location TEXT, bedrooms INTEGER, unit_type TEXT, payment TEXT,
          dp_pct INTEGER, installment_years INTEGER, finishing TEXT, objections TEXT, timing TEXT,
          notes_count INTEGER, positive INTEGER, unreachable INTEGER, lost INTEGER,
          last_contact TEXT, last_note TEXT, matching_units INTEGER DEFAULT 0,
          cluster INTEGER, priority INTEGER, priority_reasons TEXT, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS lead_profiles_priority ON lead_profiles(priority DESC);
        CREATE INDEX IF NOT EXISTS lead_profiles_cluster ON lead_profiles(cluster);
        CREATE TABLE IF NOT EXISTS lead_clusters (
          cluster INTEGER PRIMARY KEY, label TEXT NOT NULL, size INTEGER NOT NULL, summary TEXT NOT NULL,
          playbook TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        """)


# --------------------------------------------------------------------------- profiles
FIELDS = ["budget", "budget_min", "location", "bedrooms", "unit_type", "payment", "dp_pct",
          "installment_years", "finishing", "timing"]


def _profile_from_notes(notes: list[tuple]) -> dict:
    """notes: [(timestamp, text, agent)] oldest first. Later notes override earlier facts."""
    prof: dict = {f: None for f in FIELDS}
    objections: Counter = Counter()
    positive = unreachable = 0
    lost = False
    for _, text, _agent in notes:
        facts = extract(text)
        for f in FIELDS:
            if facts.get(f) is not None:
                prof[f] = facts[f]
        if facts.get("phone"):
            prof["phone"] = facts["phone"]
        objections.update(facts.get("objections", []))
        sig = outcome_signals(text)
        positive += sig["positive"]
        unreachable += sig["unreachable"]
        lost = lost or sig["lost"]
    last = notes[-1] if notes else (None, "", None)
    prof.update({
        "objections": [o for o, _ in objections.most_common()],
        "notes_count": len(notes), "positive": positive, "unreachable": unreachable, "lost": int(lost),
        "last_contact": last[0].isoformat() if isinstance(last[0], datetime) else last[0],
        "last_note": (last[1] or "")[:300], "agent": last[2],
    })
    return prof


def profiles_from_notes_file(path: str) -> pd.DataFrame:
    frame = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    cols = {c.lower(): c for c in frame.columns}
    lead_col = cols.get("lead_id") or cols.get("lead") or cols.get("client_id")
    text_col = cols.get("note_text") or cols.get("note") or cols.get("notes") or cols.get("text")
    if not lead_col or not text_col:
        raise ValueError("The notes file needs a lead_id column and a note_text column.")
    date_col = cols.get("created_at") or cols.get("date")
    agent_col = cols.get("agent")
    frame = frame.dropna(subset=[lead_col, text_col])
    frame["_ts"] = pd.to_datetime(frame[date_col], format="mixed", errors="coerce", utc=True) if date_col else pd.NaT
    frame = frame.sort_values("_ts", na_position="first")
    rows = []
    for lead, group in frame.groupby(lead_col, sort=False):
        notes = [(ts.floor('s').to_pydatetime() if pd.notna(ts) else None, str(t), (str(a) if agent_col and pd.notna(a) else None))
                 for ts, t, a in zip(group["_ts"], group[text_col], group[agent_col] if agent_col else [None] * len(group))]
        rows.append({"lead_key": f"notes-{lead}", "crm_lead_id": None, "name": f"Lead {lead}", **_profile_from_notes(notes)})
    return pd.DataFrame(rows)


def profiles_from_crm() -> pd.DataFrame:
    """CRM leads (including client-bot leads): structured fields + their notes/events."""
    rows = []
    for lead in radar.rows("SELECT * FROM leads"):
        events = radar.rows("SELECT created_at, details FROM events WHERE lead_id=? ORDER BY created_at", (lead["id"],))
        notes = [(datetime.fromisoformat(e["created_at"]), e["details"] or "", None) for e in events]
        prof = _profile_from_notes(notes)
        explicit = {"budget": lead["budget"], "location": lead["area"], "bedrooms": lead["bedrooms"],
                    "unit_type": lead["unit_type"]}
        prof.update({k: v for k, v in explicit.items() if v not in (None, "")})
        if lead["objection"]:
            prof["objections"] = list(dict.fromkeys(prof["objections"] + extract(lead["objection"]).get("objections", [])))
        if lead["status"] in ("Lost", "Closed"):
            prof["lost"] = 1
        if lead["status"] in ("Interested", "Viewing", "Reserved"):
            prof["positive"] += 1
        rows.append({"lead_key": f"crm-{lead['id']}", "crm_lead_id": lead["id"], "name": lead["name"],
                     **prof, "phone": lead["phone"] or prof.get("phone"),
                     "last_contact": lead["last_contact"] or prof["last_contact"] or lead["created_at"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- matching + priority
def _unit_index() -> dict:
    """{(location, min_bedrooms): sorted prices} for available units, for fast counting."""
    units = radar.rows("SELECT location, bedrooms, price FROM units WHERE status='Available'")
    index: dict = {}
    for loc in {u["location"] for u in units}:
        for beds in range(0, 9):
            prices = sorted(u["price"] for u in units if u["location"] == loc and (u["bedrooms"] or 0) >= beds)
            index[(loc, beds)] = np.array(prices)
    return index


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or v == ""


def _matching_count(index, location, bedrooms, budget) -> int:
    if _missing(location) or _missing(budget):
        return 0
    prices = index.get((location, 0 if _missing(bedrooms) else int(bedrooms)))
    if prices is None or not len(prices):
        return 0
    return int(np.searchsorted(prices, budget * 1.10, side="right"))


def _priority(p: dict, now: datetime) -> tuple[int, list[str]]:
    p = {k: (None if _missing(v) and not isinstance(v, list) else v) for k, v in p.items()}
    if p.get("lost"):
        return 0, ["Marked lost / bought elsewhere"]
    score, why = 0, []
    if p.get("positive"):
        pts = min(25, 10 + 5 * int(p["positive"]))
        score += pts
        why.append(f"+{pts} engaged ({p['positive']} visits/meetings/interest)")
    days = None
    if p.get("last_contact"):
        try:
            last = datetime.fromisoformat(str(p["last_contact"]).replace("Z", "+00:00"))
            last = last if last.tzinfo else last.replace(tzinfo=timezone.utc)
            days = max(0, (now - last).days)  # notes dated in the future count as today
        except ValueError:
            pass
    if days is not None and days <= 14:
        score += 20
        why.append(f"+20 contacted {days} days ago")
    elif days is not None and days <= 60:
        score += 10
        why.append(f"+10 contacted {days} days ago")
    elif days is not None and days > 180:
        score += 5
        why.append(f"+5 dormant {days} days, good for re-activation")
    if p.get("budget"):
        score += 10
        why.append("+10 budget known")
    if p.get("location"):
        score += 5
        why.append("+5 location known")
    if p.get("matching_units"):
        pts = min(25, 10 + int(math.log2(1 + p["matching_units"])) * 3)
        score += pts
        why.append(f"+{pts} {p['matching_units']} available units match now")
    if p.get("notes_count") and p.get("unreachable", 0) / max(1, p["notes_count"]) > 0.5:
        score -= 15
        why.append("-15 mostly unreachable")
    if p.get("timing") == "later":
        score -= 10
        why.append("-10 not in a hurry")
    return max(0, min(100, score)), why


# --------------------------------------------------------------------------- clustering
def _features(df: pd.DataFrame) -> np.ndarray:
    budget = df["budget"].astype(float)
    med = budget.median() if budget.notna().any() else 5_000_000
    X = pd.DataFrame({
        "budget": np.log10(budget.fillna(med).clip(lower=100_000)) * 2.0,
        "budget_known": budget.notna().astype(float),
        "bedrooms": df["bedrooms"].astype(float).fillna(df["bedrooms"].astype(float).median() if df["bedrooms"].notna().any() else 3),
        "cash": (df["payment"] == "cash").astype(float),
        "dp": df["dp_pct"].astype(float).fillna(0) / 10,
        "engaged": np.log1p(df["positive"].astype(float)),
        "unreachable": df["unreachable"].astype(float) / df["notes_count"].clip(lower=1).astype(float),
        "later": (df["timing"] == "later").astype(float),
    })
    top_locs = df["location"].value_counts().head(12).index
    for loc in top_locs:
        X[f"loc_{loc}"] = (df["location"] == loc).astype(float) * 1.5
    for typ in df["unit_type"].dropna().unique():
        X[f"type_{typ}"] = (df["unit_type"] == typ).astype(float)
    for obj in PLAYBOOK:
        X[f"obj_{obj}"] = df["objections"].apply(lambda o: obj in (o or [])).astype(float) * 0.7
    X = X.fillna(0)
    std = X.std().replace(0, 1)
    return ((X - X.mean()) / std).to_numpy()


def _choose_k(X: np.ndarray, k_range=range(5, 11)) -> int:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = len(X)
    if n < 10:
        return max(1, min(3, n))
    rng = np.random.default_rng(0)
    sample = X[rng.choice(n, size=min(n, 4000), replace=False)]
    best_k, best = min(k_range), -1.0
    for k in k_range:
        if k >= len(sample):
            break
        labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(sample)
        s = silhouette_score(sample, labels)
        if s > best:
            best_k, best = k, s
    return best_k


def _objection_rates(df: pd.DataFrame) -> dict:
    n = max(1, len(df))
    counts = Counter(o for lst in df["objections"] for o in (lst or []))
    return {o: c / n for o, c in counts.items()}


def _describe(group: pd.DataFrame, global_rates: dict) -> tuple[str, dict, str]:
    def mode(col, unknown, mixed=None):
        v = group[col].dropna()
        if len(v) < 0.5 * len(group):
            return unknown
        top = v.value_counts()
        if mixed and top.iloc[0] < 0.4 * len(group):
            return mixed
        return top.index[0]
    budget = group["budget"].dropna()
    rates = _objection_rates(group)
    # the objection that is unusually common in this cluster, not just common everywhere
    lift = sorted(((r / global_rates.get(o, r), o) for o, r in rates.items() if r >= 0.05), reverse=True)
    distinctive = [o for _, o in lift if _ > 1.15]
    beds = mode("bedrooms", None)
    parts = [mode("location", "Area unknown", "Mixed areas"), mode("unit_type", "Type unknown", "Mixed types")]
    if beds is not None:
        parts.append("Studio" if int(beds) == 0 else f"{int(beds)}BR")
    if len(budget) >= 0.3 * len(group):
        parts.append(f"~{budget.median() / 1e6:.1f}M")
    else:
        parts.append("budget unknown")
    cash_share = (group["payment"] == "cash").mean()
    parts.append("cash" if cash_share > 0.5 else "installments")
    engaged = (group["positive"] > 0).mean()
    summary = {
        "size": int(len(group)),
        "median_budget": float(budget.median()) if len(budget) else None,
        "budget_p25_p75": [float(budget.quantile(.25)), float(budget.quantile(.75))] if len(budget) else None,
        "location_known_share": round(float(group["location"].notna().mean()), 2),
        "top_locations": group["location"].value_counts().head(3).to_dict(),
        "top_types": group["unit_type"].value_counts().head(3).to_dict(),
        "cash_share": round(float(cash_share), 2),
        "engaged_share": round(float(engaged), 2),
        "lost_share": round(float(group["lost"].mean()), 2),
        "unreachable_share": round(float((group["unreachable"] > 0).mean()), 2),
        "objection_rates": {o: round(r, 3) for o, r in sorted(rates.items(), key=lambda x: -x[1])[:4]},
        "avg_priority": round(float(group["priority"].mean()), 1),
        "with_matching_units": int((group["matching_units"] > 0).sum()),
    }
    if distinctive:
        parts.append(f"objection: {distinctive[0].replace('_', ' ')}")
    tips = [PLAYBOOK[o] for o in distinctive[:2]]
    if summary["unreachable_share"] > 0.4:
        tips.append("Many are hard to reach: start with a WhatsApp message, then call at a different time of day.")
    if summary["location_known_share"] < 0.5 or len(budget) < 0.3 * len(group):
        tips.append("Requirements are incomplete: qualify area and budget on the next call before pitching.")
    if summary["with_matching_units"] == 0:
        tips.append("No current inventory matches: keep warm and alert them when matching units are added.")
    if not tips:
        tips.append("Share 2-3 matching units with prices and invite to a site visit.")
    return " · ".join(parts), summary, " ".join(tips)


def run_clustering(notes_path: str | None = None, k: int | None = None) -> dict:
    """Build profiles, cluster them, score priority, and save everything. Returns a summary."""
    from sklearn.cluster import KMeans

    radar.init_db()
    _ensure_tables()
    frames = [profiles_from_crm()]
    if notes_path:
        frames.append(profiles_from_notes_file(notes_path))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if df.empty:
        return {"leads": 0, "clusters": []}
    for col in ["phone", "budget_min", "installment_years", "finishing"]:
        if col not in df:
            df[col] = None
    index = _unit_index()
    df["matching_units"] = [_matching_count(index, loc, beds, bud)
                            for loc, beds, bud in zip(df["location"], df["bedrooms"], df["budget"])]
    now = datetime.now(timezone.utc)
    scored = [_priority(r, now) for r in df.to_dict("records")]
    df["priority"] = [s for s, _ in scored]
    df["priority_reasons"] = [json.dumps(w) for _, w in scored]

    X = _features(df)
    k = k or _choose_k(X)
    df["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X) if len(df) > k else range(len(df))

    stamp = radar.now()
    clusters = []
    with radar.connect() as db:
        db.execute("DELETE FROM lead_profiles")
        db.execute("DELETE FROM lead_clusters")
        cols = ["lead_key", "crm_lead_id", "name", "phone", "agent", "budget", "budget_min", "location", "bedrooms",
                "unit_type", "payment", "dp_pct", "installment_years", "finishing", "objections", "timing",
                "notes_count", "positive", "unreachable", "lost", "last_contact", "last_note", "matching_units",
                "cluster", "priority", "priority_reasons"]
        records = []
        for r in df.to_dict("records"):
            r["objections"] = json.dumps(r.get("objections") or [])
            vals = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, float) and math.isnan(v):
                    v = None
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                vals.append(v)
            records.append((*vals, stamp))
        db.executemany(f"INSERT INTO lead_profiles({','.join(cols)},updated_at) VALUES({','.join('?' * (len(cols) + 1))})",
                       records)
        global_rates = _objection_rates(df)
        for c, group in df.groupby("cluster"):
            label, summary, playbook = _describe(group, global_rates)
            db.execute("INSERT INTO lead_clusters VALUES(?,?,?,?,?,?)",
                       (int(c), label, int(len(group)), json.dumps(summary, ensure_ascii=False, default=str), playbook, stamp))
            clusters.append({"cluster": int(c), "label": label, "playbook": playbook, **summary})
        db.execute("INSERT INTO events(kind,details,created_at) VALUES(?,?,?)",
                   ("Lead clustering", f"{len(df)} leads in {k} clusters", stamp))
    clusters.sort(key=lambda c: -c["size"])
    return {"leads": int(len(df)), "k": int(k), "clusters": clusters}


# --------------------------------------------------------------------------- queries
def _decode(p: dict) -> dict:
    p = dict(p)
    p["objections"] = json.loads(p.get("objections") or "[]")
    p["priority_reasons"] = json.loads(p.get("priority_reasons") or "[]")
    return p


def list_clusters() -> list[dict]:
    _ensure_tables()
    out = []
    for c in radar.rows("SELECT * FROM lead_clusters ORDER BY size DESC"):
        out.append({**c, "summary": json.loads(c["summary"])})
    return out


def priority_list(limit: int = 20, cluster: int | None = None, agent: str | None = None) -> list[dict]:
    _ensure_tables()
    sql, args = "SELECT * FROM lead_profiles WHERE lost=0", []
    if cluster is not None:
        sql += " AND cluster=?"
        args.append(cluster)
    if agent:
        sql += " AND agent=?"
        args.append(agent)
    sql += " ORDER BY priority DESC, last_contact DESC LIMIT ?"
    args.append(limit)
    return [_decode(p) for p in radar.rows(sql, tuple(args))]


def get_profile(key: str) -> dict | None:
    """Find a lead by profile key, CRM id, notes lead id, phone or name."""
    _ensure_tables()
    key = str(key).strip()
    candidates = [("lead_key=?", key), ("lead_key=?", f"notes-{key}"), ("lead_key=?", f"crm-{key}"),
                  ("phone=?", key), ("name LIKE ?", f"%{key}%")]
    for where, val in candidates:
        rows = radar.rows(f"SELECT * FROM lead_profiles WHERE {where} LIMIT 1", (val,))
        if rows:
            return _decode(rows[0])
    return None
