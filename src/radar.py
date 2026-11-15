"""RADAR CRM domain services and deterministic multi-agent workflow."""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config

DB_PATH = Path(config.ROOT_DIR) / "data" / "radar.db"
STATUSES = ("New", "Contacted", "Interested", "Viewing", "Reserved", "Closed", "Lost")


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_unit_dataframe(frame):
    col_map = {
        "Unit Name": "Unit ID",
        "Unit": "Unit ID",
        "unit_id": "Unit ID",
        "Nominal Price": "Price (EGP)",
        "Price": "Price (EGP)",
        "BUA": "Area (sqm)",
        "Area": "Area (sqm)",
        "No. of Bedrooms": "Bedrooms",
        "Usage Type": "Type",
    }
    for old, new in col_map.items():
        if old in frame.columns and new not in frame.columns:
            frame[new] = frame[old]

    project_locations = {
        "rai valleys": "Mostakbal City",
        "talala": "New Cairo",
        "the butterfly": "Mostakbal City",
        "sarai": "New Cairo",
        "étaje": "Sheikh Zayed",
        "etaje": "Sheikh Zayed",
        "elm tree park": "New Cairo",
        "club views": "New Cairo",
        "d2n": "New Cairo",
        "d2n2": "New Cairo",
        "esse residence": "New Cairo",
        "zahw assuit": "Assiut",
        "klok": "New Cairo",
        "sheya residence": "New Cairo",
        "origami": "Mostakbal City",
        "ville avenue": "New Cairo",
        "kinda res.": "New Cairo",
        "elan": "New Cairo",
        "palm residence": "New Cairo",
        "nile view towers": "Sheikh Zayed",
    }
    if "Location" not in frame.columns:
        def infer_loc(p):
            p_str = str(p).lower()
            for k, loc in project_locations.items():
                if k in p_str:
                    return loc
            return "New Cairo"
        frame["Location"] = frame["Project"].apply(infer_loc) if "Project" in frame.columns else "New Cairo"

    if "Status" not in frame.columns:
        frame["Status"] = "Available"
    if "Delivery Year" not in frame.columns:
        frame["Delivery Year"] = 2027
    if "Bedrooms" not in frame.columns:
        frame["Bedrooms"] = 0
    else:
        frame["Bedrooms"] = frame["Bedrooms"].fillna(0).astype(int)
    if "Type" not in frame.columns:
        frame["Type"] = "Apartment"
    return frame


def init_db():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS leads (
          id INTEGER PRIMARY KEY, name TEXT NOT NULL, phone TEXT, language TEXT DEFAULT 'English',
          budget REAL, area TEXT, bedrooms INTEGER, unit_type TEXT, payment_preference TEXT,
          monthly_capacity REAL, objection TEXT, notes TEXT DEFAULT '', status TEXT DEFAULT 'New',
          consent INTEGER DEFAULT 0, last_contact TEXT, created_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS lead_phone_unique ON leads(phone) WHERE phone IS NOT NULL AND phone != '';
        CREATE TABLE IF NOT EXISTS units (
          id INTEGER PRIMARY KEY, unit_id TEXT UNIQUE NOT NULL, project TEXT, location TEXT,
          type TEXT, bedrooms INTEGER, area REAL, price REAL, status TEXT DEFAULT 'Available',
          down_payment REAL DEFAULT 10, years INTEGER DEFAULT 8, payment TEXT DEFAULT '',
          delivery_year INTEGER, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS opportunities (
          id INTEGER PRIMARY KEY, lead_id INTEGER NOT NULL REFERENCES leads(id), unit_id INTEGER NOT NULL REFERENCES units(id),
          trigger TEXT NOT NULL, trigger_version TEXT NOT NULL, score INTEGER NOT NULL,
          reasons TEXT NOT NULL, action TEXT NOT NULL, draft TEXT NOT NULL,
          status TEXT DEFAULT 'Pending', created_at TEXT NOT NULL, UNIQUE(lead_id,unit_id,trigger,trigger_version)
        );
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY, lead_id INTEGER REFERENCES leads(id), unit_id INTEGER REFERENCES units(id),
          kind TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, lead_id INTEGER REFERENCES leads(id),
          field TEXT, value TEXT, confidence REAL, evidence TEXT, source TEXT, created_at TEXT NOT NULL);
        """)
        db.execute("INSERT OR IGNORE INTO settings VALUES ('kill_switch','0')")
        db.execute("INSERT OR IGNORE INTO settings VALUES ('autonomy','1')")

        import pandas as pd

        # 1. Ingest base units.csv
        csv = Path(config.SHEET_SOURCE)
        if csv.exists():
            for _, row in pd.read_csv(csv).iterrows():
                db.execute("""INSERT OR IGNORE INTO units(unit_id,project,location,type,bedrooms,area,price,status,delivery_year,updated_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?)""", (str(row['Unit ID']), str(row['Project']), str(row['Location']), str(row['Type']),
                  int(row['Bedrooms']), float(row['Area (sqm)']), float(row['Price (EGP)']), str(row['Status']), int(row['Delivery Year']), now()))

        # 2. Ingest additional Excel units if present in data/xlsx
        xlsx_dir = Path(config.ROOT_DIR) / "data" / "xlsx"
        if xlsx_dir.exists():
            for xfile in xlsx_dir.glob("*.xlsx"):
                try:
                    df = pd.read_excel(xfile)
                    df = _normalize_unit_dataframe(df)
                    df = df.dropna(subset=["Unit ID", "Price (EGP)", "Project"])
                    for _, row in df.iterrows():
                        db.execute("""INSERT OR IGNORE INTO units(unit_id,project,location,type,bedrooms,area,price,status,delivery_year,updated_at)
                          VALUES(?,?,?,?,?,?,?,?,?,?)""", (
                            str(row["Unit ID"]), str(row["Project"]), str(row["Location"]), str(row["Type"]),
                            int(row["Bedrooms"]), float(row["Area (sqm)"]), float(row["Price (EGP)"]),
                            str(row["Status"]), int(row["Delivery Year"]), now()
                        ))
                except Exception as e:
                    pass

        # 3. Seed initial buyer leads
        if db.execute("SELECT COUNT(*) FROM leads").fetchone()[0] == 0:
            examples = [
              ("Mariam Hassan", "+201001234567", "Arabic", 6000000, "New Cairo", 2, "Apartment", "low down payment", 45000, "Down payment is too high; prefers quarterly payments.", 1),
              ("Omar Khaled", "+201112345678", "English", 8000000, "New Cairo", 3, "Apartment", "flexible installments", 65000, "Visited a unit; wants a larger balcony and a lower total price.", 1),
              ("Nour Adel", "+201223456789", "Arabic", 4000000, "Sheikh Zayed", 1, "Apartment", "monthly installments", 35000, "Asked to be contacted next month. Interested in finished units.", 0),
              ("Tarek Mansour", "+201009876543", "Arabic", 20000000, "Mostakbal City", 5, "Villa", "installments over 8 years", 120000, "Interested in standalone villas in Rai Valleys or The Butterfly.", 1),
              ("Youssef El-Sayed", "+201119876543", "English", 12000000, "New Cairo", 3, "Apartment", "quarterly installments", 85000, "Looking for 3-bedroom apartment in TALALA or Sarai.", 1),
              ("Dina Farouk", "+201229876543", "English", 10000000, "Sheikh Zayed", 3, "Apartment", "semi-finished", 70000, "Prefers ground floor with garden in ÉTAJE.", 1),
            ]
            for name, phone, lang, budget, area, beds, typ, payment, cap, note, consent in examples:
                db.execute("""INSERT INTO leads(name,phone,language,budget,area,bedrooms,unit_type,payment_preference,monthly_capacity,notes,consent,created_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (name,phone,lang,budget,area,beds,typ,payment,cap,note,consent,now()))
                lead_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
                db.execute("INSERT INTO events(lead_id,kind,details,created_at) VALUES(?,?,?,?)", (lead_id,"Note added",note,now()))
                db.execute("INSERT INTO facts(lead_id,field,value,confidence,evidence,source,created_at) VALUES(?,?,?,?,?,?,?)",
                    (lead_id,"objection",note,0.91,note,"Seed note",now()))


def rows(query, args=()):
    with connect() as db:
        return [dict(r) for r in db.execute(query,args).fetchall()]


def add_lead(data):
    phone = re.sub(r"\D", "", data.get("phone", ""))
    if phone and len(phone) < 8:
        raise ValueError("Enter a valid phone number.")
    with connect() as db:
        cur = db.execute("""INSERT INTO leads(name,phone,language,budget,area,bedrooms,unit_type,payment_preference,monthly_capacity,objection,notes,status,consent,created_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (data["name"],phone,data.get("language","English"),data.get("budget"),data.get("area"),data.get("bedrooms"),data.get("unit_type"),data.get("payment_preference"),data.get("monthly_capacity"),data.get("objection"),data.get("notes",""),data.get("status","New"),int(data.get("consent",False)),now()))
        lid=cur.lastrowid
        db.execute("INSERT INTO events(lead_id,kind,details,created_at) VALUES(?,?,?,?)",(lid,"Lead created",data.get("notes", "Lead added"),now()))
    return lid


def import_units(file_bytes, filename):
    import pandas as pd
    from io import BytesIO, StringIO
    if len(file_bytes) > 50_000_000: raise ValueError("File limit is 50 MB.")
    if filename.lower().endswith(".csv"): frame = pd.read_csv(StringIO(file_bytes.decode("utf-8-sig")))
    elif filename.lower().endswith((".xlsx", ".xls")): frame = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    else: raise ValueError("Upload a .csv or .xlsx file.")

    frame = _normalize_unit_dataframe(frame)
    required = {"Unit ID", "Project", "Location", "Type", "Bedrooms", "Area (sqm)", "Price (EGP)", "Status", "Delivery Year"}
    missing = required - set(frame.columns)
    if missing: raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    frame = frame[list(required)].dropna(subset=["Unit ID", "Price (EGP)"])
    frame = frame.drop_duplicates(subset=["Unit ID"], keep="last")
    changed = 0
    with connect() as db:
        for _, r in frame.iterrows():
            key = str(r["Unit ID"])
            old = db.execute("SELECT * FROM units WHERE unit_id=?", (key,)).fetchone()
            vals = (str(r["Project"]), str(r["Location"]), str(r["Type"]), int(r["Bedrooms"]), float(r["Area (sqm)"]), float(r["Price (EGP)"]), str(r["Status"]), int(r["Delivery Year"]))
            if old:
                prev = (old["project"], old["location"], old["type"], old["bedrooms"], old["area"], old["price"], old["status"], old["delivery_year"])
                if prev != vals:
                    db.execute("UPDATE units SET project=?,location=?,type=?,bedrooms=?,area=?,price=?,status=?,delivery_year=?,updated_at=? WHERE unit_id=?", (*vals, now(), key))
                    db.execute("INSERT INTO events(unit_id,kind,details,created_at) VALUES((SELECT id FROM units WHERE unit_id=?),'Unit updated',?,?,?)", (key, f"{key} updated from import", now()))
                    create_opportunities(db, key, "unit_updated")
                    changed += 1
            else:
                db.execute("INSERT INTO units(unit_id,project,location,type,bedrooms,area,price,status,delivery_year,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)", (key, *vals, now()))
                db.execute("INSERT INTO events(unit_id,kind,details,created_at) VALUES((SELECT id FROM units WHERE unit_id=?),'Unit added',?,?,?)", (key, f"{key} added from import", now()))
                create_opportunities(db, key, "new_unit")
                changed += 1
    return len(frame), changed


def create_opportunities(db, unit_key=None, trigger="scan"):
    """Run the specialized CRM agents through the LangGraph orchestrator."""
    from .crm_workflow import run_opportunity_workflow

    return run_opportunity_workflow(db, unit_key=unit_key, trigger=trigger)


def scan_dormant():
    with connect() as db:
        return create_opportunities(db)


def decide_opportunity(opp_id, decision, reason=""):
    if decision not in ("Approved","Snoozed","Dismissed"): raise ValueError("Invalid decision")
    if decision=="Dismissed" and not reason: raise ValueError("Choose a dismissal reason")
    with connect() as db:
        o=db.execute("SELECT o.*,l.consent,l.name FROM opportunities o JOIN leads l ON l.id=o.lead_id WHERE o.id=?",(opp_id,)).fetchone()
        if not o: raise ValueError("Opportunity not found")
        if decision=="Approved" and not o["consent"]: raise ValueError("Blocked: this lead has no recorded contact consent.")
        db.execute("UPDATE opportunities SET status=? WHERE id=?",(decision,opp_id))
        db.execute("INSERT INTO events(lead_id,unit_id,kind,details,created_at) VALUES(?,?,?,?,?)",(o["lead_id"],o["unit_id"],decision,f"Opportunity {opp_id}; {reason or 'Human approval recorded'}",now()))


def update_setting(key,value):
    with connect() as db: db.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)",(key,str(value)))


def settings(): return {r["key"]:r["value"] for r in rows("SELECT * FROM settings")}


def add_note(lead_id,note):
    # Intake Agent: preserve the evidence and use conservative, explicit signal extraction.
    text=note.strip()
    if not text: return
    with connect() as db:
        db.execute("UPDATE leads SET notes=CASE WHEN notes='' THEN ? ELSE notes||char(10)||? END,last_contact=? WHERE id=?",(text,text,now(),lead_id))
        db.execute("INSERT INTO events(lead_id,kind,details,created_at) VALUES(?,?,?,?)",(lead_id,"Note added",text,now()))
        patterns=[("budget",r"(?:budget|ميزانية)\D{0,12}([\d,]+)"),("bedrooms",r"(\d+)\s*(?:bed|bedroom|غرف)")]
        for field,pattern in patterns:
            m=re.search(pattern,text,re.I)
            if m:
                value=re.sub(r",","",m.group(1))
                db.execute("INSERT INTO facts(lead_id,field,value,confidence,evidence,source,created_at) VALUES(?,?,?,?,?,?,?)",(lead_id,field,value,0.82,m.group(0),"Lead note",now()))
                col=field
                db.execute(f"UPDATE leads SET {col}=? WHERE id=?",(float(value) if field=="budget" else int(value),lead_id))


def update_unit(unit_key, price, status):
    with connect() as db:
        u=db.execute("SELECT * FROM units WHERE unit_id=?",(unit_key,)).fetchone()
        if not u: raise ValueError("Unit not found")
        changes=[]
        if price is not None and float(price)!=u["price"]: changes.append(("price",u["price"],float(price)))
        if status and status!=u["status"]: changes.append(("status",u["status"],status))
        if not changes:return 0
        db.execute("UPDATE units SET price=?,status=?,updated_at=? WHERE unit_id=?",(float(price) if price is not None else u["price"],status or u["status"],now(),unit_key))
        db.execute("INSERT INTO events(unit_id,kind,details,created_at) VALUES(?,?,?,?)",(u["id"],"Unit changed","; ".join(f"{k}: {a} -> {b}" for k,a,b in changes),now()))
        return create_opportunities(db,unit_key,"price_drop" if any(k=="price" and b<a for k,a,b in changes) else "unit_updated")
