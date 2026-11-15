"""
=============================================================================
◉ RADAR PLATFORM — ALL-IN-ONE ENTRYPOINT (PHASE 2)
=============================================================================
Unified Real Estate CRM · RAG Chat Assistant · Multi-Agent Research Lab

Run the interactive Streamlit dashboard:
    streamlit run main.py

Or use the CLI:
    python main.py build               # Build vector index
    python main.py chat                # Terminal chat with memory
    python main.py chat --summary      # Terminal chat with summary-buffer memory
    python main.py ask "your question" # Single query
=============================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config, radar
from src.indexer import build_index, load_index
from src.llm import check_llm, get_llm
from src.loaders import load_all
from src.memory import BufferMemory, SummaryBufferMemory
from src.pdf_service import extract_pdf, PDFExtractionError
from src.rag import RealEstateAssistant
from src.report_generator import list_reports, save_report
from src.research_workflow import run_research_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# =============================================================================
# CLI HANDLERS
# =============================================================================

def cmd_build(args):
    docs = load_all()
    pdf_count = sum(d.metadata["type"] == "pdf" for d in docs)
    print(f"Loaded {len(docs)} chunks ({pdf_count} from PDFs, {len(docs) - pdf_count} from sheet).")
    build_index(docs, offline=args.offline)
    print("Vector index saved successfully.")


def make_assistant(args):
    if not args.offline:
        check_llm()
    llm = get_llm(args.offline)
    memory = SummaryBufferMemory(llm) if getattr(args, "summary", False) else BufferMemory()
    return RealEstateAssistant(load_index(args.offline), llm, memory)


def show_chat_turn(answer, docs, standalone, question, sources=True):
    print(f"\nAssistant: {answer}\n")
    if sources:
        if standalone != question:
            print(f"  (searched for: {standalone})")
        for d in docs:
            m = d.metadata
            where = f"p.{m['page']}" if m["type"] == "pdf" else f"row {m['row']}"
            print(f"  - {m['source']} {where}")
        print()


def cmd_chat(args):
    bot = make_assistant(args)
    print("Real-estate assistant ready. Commands: /reset  /sources  /quit")
    show_src = True
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/reset":
            bot.memory.clear()
            print("Memory cleared.")
            continue
        if q == "/sources":
            show_src = not show_src
            print(f"Sources {'on' if show_src else 'off'}.")
            continue
        show_chat_turn(*bot.ask(q), q, show_src)


def cmd_ask(args):
    bot = make_assistant(args)
    show_chat_turn(*bot.ask(args.question), args.question)


def run_cli():
    ap = argparse.ArgumentParser(description="RADAR Platform CLI", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="no API key / no downloads (testing only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    c = sub.add_parser("chat")
    c.add_argument("--summary", action="store_true")
    c.set_defaults(fn=cmd_chat)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.set_defaults(fn=cmd_ask, summary=False)
    args = ap.parse_args()
    args.fn(args)


# =============================================================================
# STREAMLIT DASHBOARD (UI)
# =============================================================================

def run_streamlit():
    st.set_page_config(page_title="RADAR Platform", page_icon="◉", layout="wide")
    radar.init_db()

    st.markdown("""<style>
    .stApp{background:#f5f7fb;color:#172033}.block-container{padding-top:1.5rem;max-width:1450px}
    [data-testid=stMetric]{background:white;border:1px solid #e5eaf1;padding:14px;border-radius:14px}
    .hero{background:linear-gradient(115deg,#102a43,#176b87);padding:25px 30px;border-radius:18px;color:#fff;margin-bottom:18px}
    .hero h1{margin:0;color:white}.hero p{margin:8px 0 0;color:#d7e7f0}
    </style>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="hero"><h1>◉ RADAR Platform</h1>'
        '<p>Real estate CRM · RAG chat assistant · Multi-agent research lab · Phase 2 Unified</p></div>',
        unsafe_allow_html=True,
    )

    S = radar.settings()
    if S.get("kill_switch") == "1":
        st.error("AGENT PAUSED · Kill switch active")
    else:
        st.success("Agent active · customer messages require human approval")

    leads = radar.rows("SELECT * FROM leads ORDER BY id DESC")
    units = radar.rows("SELECT * FROM units ORDER BY project,unit_id")
    opps = radar.rows(
        "SELECT o.*,l.name lead_name,l.language,l.consent,u.unit_id,u.project,"
        "u.location,u.type,u.bedrooms,u.area,u.price,u.status unit_status "
        "FROM opportunities o JOIN leads l ON o.lead_id=l.id "
        "JOIN units u ON o.unit_id=u.id ORDER BY o.created_at DESC"
    )
    pending = [o for o in opps if o["status"] == "Pending"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Open leads", sum(x["status"] not in ("Closed", "Lost") for x in leads))
    m2.metric("Available units", sum(x["status"] == "Available" for x in units))
    m3.metric("Needs review", len(pending))
    m4.metric("Closed / reserved", sum(x["status"] in ("Closed", "Reserved") for x in leads))
    m5.metric("Research reports", len(list_reports()))

    tabs = st.tabs([
        "Overview",
        "💬 Chat",
        "Leads",
        "Opportunities",
        "Inventory",
        "📄 Research Lab",
        "Insights & Imports",
        "Agent controls",
        "Architecture",
    ])

    # 1. Overview
    with tabs[0]:
        x, y = st.columns([1.2, 1])
        with x:
            st.subheader("Opportunity inbox")
            for o in pending[:5]:
                st.markdown(
                    f"**{o['lead_name']} → {o['project']} / {o['unit_id']}** · Fit **{o['score']}/100**\n\n"
                    f"Why now: {json.loads(o['reasons'])['why'][0]} · Action: {o['action']}"
                )
            if not pending:
                st.info("No pending opportunities. Change inventory or scan dormant leads.")
        with y:
            st.subheader("Multi-agent workflow")
            st.markdown(
                "**LangGraph orchestration** runs a visible sequence of specialized nodes:\n\n"
                "1. **Intake Agent** loads active leads and verified inventory.\n"
                "2. **Matching Agent** applies deterministic budget, location and bedroom rules.\n"
                "3. **Opportunity Agent** checks event relevance and duplicate keys.\n"
                "4. **Recommendation Agent** prepares a bilingual draft and evidence.\n"
                "5. **Persistence** records the result for human review.\n\n"
                "The policy gate remains backend code: the kill switch stops the graph, "
                "and consent is required before approval."
            )
            if st.button("Run dormant lead scan", type="primary"):
                st.success(f"Scan complete · {radar.scan_dormant()} new opportunities")
                st.rerun()

    # 2. Chat (RAG)
    with tabs[1]:
        st.subheader("💬 Real Estate Chat Assistant")
        st.caption("Ask questions about projects, units, brochures and policies.")

        if "chat_bot" not in st.session_state:
            try:
                offline_chat = config.LLM_PROVIDER != "ollama" and not config.LLM_API_KEY
                llm = get_llm(offline=offline_chat)
                store = load_index()
                st.session_state["chat_bot"] = RealEstateAssistant(store, llm, BufferMemory())
                st.session_state["chat_history"] = []
                if offline_chat:
                    st.info("Chat is using local offline mode because no LLM API key is configured.")
            except FileNotFoundError:
                try:
                    store = build_index(load_all(), offline=True)
                    offline_chat = config.LLM_PROVIDER != "ollama" and not config.LLM_API_KEY
                    llm = get_llm(offline=offline_chat)
                    st.session_state["chat_bot"] = RealEstateAssistant(store, llm, BufferMemory())
                    st.session_state["chat_history"] = []
                    if offline_chat:
                        st.info("Chat is using local offline mode because no LLM API key is configured.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not initialize chat assistant: {exc}")
                    st.session_state["chat_bot"] = None
                    st.session_state["chat_history"] = []

        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for s_item in msg["sources"]:
                            st.caption(s_item)

        if prompt := st.chat_input("Ask about real estate projects, units, or policies…"):
            bot = st.session_state.get("chat_bot")
            if bot is None:
                st.error("Chat assistant is unavailable.")
            else:
                st.session_state["chat_history"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Thinking…"):
                            answer, docs, standalone = bot.ask(prompt)
                    except Exception as exc:
                        st.error(f"Chat request failed: {exc}")
                        answer, docs, standalone = "", [], prompt
                    if not answer:
                        st.stop()
                    st.markdown(answer)
                    sources = []
                    for d in docs:
                        m = d.metadata
                        where = f"p.{m['page']}" if m["type"] == "pdf" else f"row {m['row']}"
                        sources.append(f"{m['source']} {where}")
                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                st.caption(s)

                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("Clear chat"):
                bot = st.session_state.get("chat_bot")
                if bot:
                    bot.memory.clear()
                st.session_state["chat_history"] = []
                st.rerun()

    # 3. Leads
    with tabs[2]:
        st.subheader("Lead workspace")
        q = st.text_input("Search leads")
        status = st.selectbox("Filter status", ["All", *radar.STATUSES])
        show = [r for r in leads if (status == "All" or r["status"] == status) and (not q or q.lower() in str(r).lower())]
        if show:
            st.dataframe(pd.DataFrame(show)[["id", "name", "phone", "status", "budget", "area", "bedrooms", "objection", "consent"]], use_container_width=True, hide_index=True)

        with st.expander("＋ Add lead"):
            with st.form("add_lead_form"):
                c1, c2, c3 = st.columns(3)
                name = c1.text_input("Name *")
                phone = c2.text_input("Phone")
                lang = c3.selectbox("Draft language", ["English", "Arabic"])
                budget = c1.number_input("Budget (EGP)", min_value=0, step=100000)
                area = c2.text_input("Area")
                beds = c3.number_input("Bedrooms", min_value=0, max_value=10, value=1)
                kind = c1.selectbox("Unit type", ["Apartment", "Villa", "Townhouse", "Studio"])
                cap = c2.number_input("Monthly capacity (EGP)", min_value=0, step=5000)
                pref = c3.text_input("Payment preference")
                objection = c1.text_input("Objection")
                note = c2.text_area("Notes / evidence")
                consent = c3.checkbox("Recorded contact consent")
                if st.form_submit_button("Create lead"):
                    try:
                        if not name:
                            raise ValueError("Name is required")
                        radar.add_lead(dict(name=name, phone=phone, language=lang, budget=budget or None, area=area, bedrooms=beds, unit_type=kind, monthly_capacity=cap or None, payment_preference=pref, objection=objection, notes=note, consent=consent))
                        st.success("Lead created")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        if leads:
            lead = st.selectbox("Lead profile", leads, format_func=lambda x: f"{x['name']} · {x['status']}")
            st.write(f"Budget EGP {lead['budget']:,.0f} · {lead['area']} · {lead['bedrooms']} bedrooms · monthly capacity EGP {lead['monthly_capacity'] or 0:,.0f}")
            st.caption("Notes: " + (lead["notes"] or "None"))
            ev = radar.rows("SELECT * FROM events WHERE lead_id=? ORDER BY created_at DESC", (lead["id"],))
            st.caption("Timeline: " + " · ".join(e["kind"] + ": " + e["details"] for e in ev[:5]))
            n = st.text_input("Add dated note", key=f"n{lead['id']}")
            if st.button("Save note", key=f"sn{lead['id']}") and n:
                radar.add_note(lead["id"], n)
                st.rerun()
            s = st.selectbox("Funnel outcome", radar.STATUSES, index=radar.STATUSES.index(lead["status"]), key=f"s{lead['id']}")
            co = st.checkbox("Contact consent recorded", bool(lead["consent"]), key=f"c{lead['id']}")
            if st.button("Save profile", key=f"sp{lead['id']}"):
                with radar.connect() as db:
                    db.execute("UPDATE leads SET status=?,consent=? WHERE id=?", (s, int(co), lead["id"]))
                st.rerun()

    # 4. Opportunities
    with tabs[3]:
        st.subheader("Explainable opportunity review")
        state_filter = st.selectbox("Show opportunities", ["Pending", "All", "Approved", "Snoozed", "Dismissed"])
        for o in [x for x in opps if state_filter == "All" or x["status"] == state_filter]:
            info = json.loads(o["reasons"])
            with st.container(border=True):
                left, right = st.columns([4, 1])
                left.markdown(f"### {o['lead_name']} · {o['project']} {o['unit_id']}")
                right.metric("Fit score", f"{o['score']}/100")
                st.caption(f"{o['trigger'].replace('_', ' ').title()} · EGP {o['price']:,.0f} · verified {info['unit_verified'][:10]}")
                st.write("**Why match / why now**")
                for v in info["why"]:
                    st.write("• " + v)
                st.write("**Factor breakdown**")
                st.json(info["factors"])
                st.caption("Evidence: " + info["evidence"])
                st.write("Suggested action: " + o["action"])
                st.text_area("Prepared draft · edit before approval", o["draft"], key=f"d{o['id']}")
                if o["status"] == "Pending":
                    p1, p2, p3, p4 = st.columns([1, 1, 2, 1])
                    if p1.button("Approve", key=f"a{o['id']}"):
                        try:
                            radar.decide_opportunity(o["id"], "Approved")
                            st.success("Approval recorded. No external message was sent.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                    if p2.button("Snooze", key=f"z{o['id']}"):
                        radar.decide_opportunity(o["id"], "Snoozed")
                        st.rerun()
                    reason = p3.selectbox("Dismissal reason", ["Not relevant", "Already contacted", "Customer not interested", "Data incorrect"], key=f"r{o['id']}")
                    if p4.button("Dismiss", key=f"x{o['id']}"):
                        radar.decide_opportunity(o["id"], "Dismissed", reason)
                        st.rerun()

    # 5. Inventory
    with tabs[4]:
        st.subheader("Project and unit inventory")
        st.dataframe(pd.DataFrame(units), use_container_width=True, hide_index=True)
        if units:
            u = st.selectbox("Edit unit", units, format_func=lambda x: f"{x['unit_id']} · {x['project']} · EGP {x['price']:,.0f}")
            with st.form("unit_edit_form"):
                price = st.number_input("Price (EGP)", value=float(u["price"]), step=50000.0)
                available = st.selectbox(
                    "Availability", ["Available", "Reserved", "Sold"],
                    index=["Available", "Reserved", "Sold"].index(u["status"]) if u["status"] in ["Available", "Reserved", "Sold"] else 0,
                )
                if st.form_submit_button("Save unit change"):
                    st.success(f"Unit updated · {radar.update_unit(u['unit_id'], price, available)} opportunities created")
                    st.rerun()

    # 6. Research Lab
    with tabs[5]:
        st.subheader("📄 Research Lab")
        st.caption("Upload a PDF for 4-agent LangGraph analysis: Extraction → Research → Analysis → Report.")

        rcol1, rcol2 = st.columns([2, 1])
        with rcol1:
            uploaded = st.file_uploader("Choose a PDF", type=["pdf"], help=f"Max size: {config.MAX_UPLOAD_MB} MB", key="main_res_pdf")
        with rcol2:
            task = st.text_area("What should the research focus on?", placeholder="Summarize key findings...", key="main_res_task")

        if uploaded:
            if uploaded.size > config.MAX_UPLOAD_BYTES:
                st.error(f"The file is too large. Maximum is {config.MAX_UPLOAD_MB} MB.")
            elif st.button("🚀 Run Research Pipeline", type="primary"):
                progress = st.progress(0, text="Extracting PDF…")
                try:
                    document = extract_pdf(uploaded.getvalue(), uploaded.name)
                    st.session_state["main_res_doc"] = document
                    progress.progress(25, text=f"Extracted {document.page_count} pages.")

                    with st.status("Running multi-agent research pipeline…", expanded=True) as status:
                        st.write("✅ PDF extraction complete.")
                        st.write("🔍 Research Agent: analyzing source text…")
                        result = run_research_workflow(document, task or "")
                        st.session_state["main_res_res"] = result
                        if result.get("error"):
                            status.update(label="Pipeline finished with errors", state="error")
                        else:
                            st.write("📊 Analysis Agent: cross-referencing findings…")
                            st.write("📝 Report Agent: generating structured report…")
                            status.update(label="Research pipeline complete", state="complete")

                    progress.progress(100, text="Finished.")
                    if result.get("report") and not result.get("error"):
                        report_path = save_report(uploaded.name, result["report"], result.get("research_notes", ""), result.get("analysis", ""), task or "")
                        st.success(f"Report saved to `{report_path.name}`")
                except PDFExtractionError as exc:
                    progress.empty()
                    st.error(str(exc))
                except Exception as exc:
                    progress.empty()
                    st.error(f"Error: {exc}")

        document = st.session_state.get("main_res_doc")
        result = st.session_state.get("main_res_res")

        if document or result:
            st.divider()
            col_l, col_r = st.columns([1, 1])
            with col_l:
                if document:
                    st.subheader("Extracted Document")
                    st.caption(f"{document.filename} · {document.page_count} pages · {len(document.text):,} characters")
                    with st.expander("View extracted text"):
                        st.text(document.text[:10000])
                if result and result.get("research_notes"):
                    st.subheader("Research Notes")
                    with st.expander("View notes"):
                        st.markdown(result["research_notes"])
                if result and result.get("analysis"):
                    st.subheader("Critical Analysis")
                    with st.expander("View analysis"):
                        st.markdown(result["analysis"])
            with col_r:
                if result and result.get("report"):
                    st.subheader("Generated Report")
                    st.markdown(result["report"])
                elif result and result.get("error"):
                    st.error(result["error"])

        st.divider()
        st.subheader("Saved Reports")
        reports = list_reports()
        if reports:
            for r in reports[:10]:
                with st.expander(f"📄 {r['filename']}"):
                    st.caption(f"Generated: {r['modified']} · {r['size']:,} bytes")
                    try:
                        content = Path(r["path"]).read_text(encoding="utf-8")
                        st.markdown(content)
                    except Exception:
                        pass
        else:
            st.info("No research reports generated yet.")

    # 7. Insights & Imports
    with tabs[6]:
        st.subheader("Imports & manager insights")
        file = st.file_uploader("Upload unit inventory (.csv or .xlsx, max 5 MB)", type=["csv", "xlsx"])
        if file:
            try:
                total, changed = radar.import_units(file.getvalue(), file.name)
                st.success(f"Imported {total} rows · {changed} changed/new · matching complete")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if leads:
            st.markdown("#### Pipeline by status")
            f = pd.DataFrame(leads).groupby("status").size().rename("Leads").reset_index()
            st.bar_chart(f.set_index("status"))
            st.markdown("#### Buyer demand vs available inventory")
            demand = pd.DataFrame(leads).groupby("area", dropna=False).size().rename("Lead demand").to_frame()
            inv = pd.DataFrame(units)
            inventory = (
                inv[inv.status == "Available"].groupby("location").size().rename("Available units")
                if len(inv)
                else pd.Series(dtype=int)
            )
            demand["Available units"] = inventory
            st.dataframe(demand.fillna(0), use_container_width=True)

    # 8. Agent Controls
    with tabs[7]:
        st.subheader("Autonomy and safety controls")
        vals = radar.settings()
        level = st.selectbox("Autonomy level", [0, 1, 2], index=int(vals.get("autonomy", "1")))
        kill = st.toggle("Kill switch · pause agent actions", value=vals.get("kill_switch") == "1")
        if st.button("Save controls"):
            radar.update_setting("autonomy", level)
            radar.update_setting("kill_switch", int(kill))
            st.success("Controls saved")
            st.rerun()
        st.markdown("#### Recent audit events")
        st.dataframe(pd.DataFrame(radar.rows("SELECT created_at,kind,details FROM events ORDER BY created_at DESC LIMIT 25")), use_container_width=True, hide_index=True)

    # 9. Architecture
    with tabs[8]:
        st.subheader("Phase 2 · All-In-One Unified Architecture")
        st.code(
            """All-in-One RADAR Platform (main.py)
│
├─ CRM Multi-Agent Pipeline (LangGraph)
│   Intake → Matching → Opportunity → Recommendation → Persistence → Review
│
├─ Real Estate RAG Assistant
│   Query Rewrite → FAISS Retrieval → Context + Memory → Answer + Citations
│
└─ Research Multi-Agent Pipeline (LangGraph)
    PDF Upload → Extraction → Research → Analysis → Report → Export""",
            language="text",
        )


# =============================================================================
# ENTRYPOINT DISPATCHER
# =============================================================================

if __name__ == "__main__":
    cli_commands = {"build", "chat", "ask"}
    if any(arg in cli_commands for arg in sys.argv[1:]):
        run_cli()
    else:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            is_st = get_script_run_ctx() is not None
        except Exception:
            is_st = False

        if is_st:
            run_streamlit()
        else:
            print("=" * 70)
            # Keep the no-argument launcher usable in Windows terminals that
            # still use a legacy code page (for example cp1252).
            print("RADAR PLATFORM - PHASE 2")
            print("=" * 70)
            print("To launch the interactive dashboard, run:")
            print("    streamlit run main.py")
            print("\nOr run the CLI commands:")
            print("    python main.py build")
            print("    python main.py chat")
            print("    python main.py ask \"What is the down payment for Palm Residence?\"")
            print("=" * 70)
