"""
=============================================================================
RADAR PLATFORM — ENTRYPOINT
=============================================================================
Real estate CRM + RAG chat with live daily prices + price-group automation.

    python main.py serve                 # HTTP API on :8000 (docs at /docs)
    python main.py telegram              # listen to the Telegram price group
    python main.py ingest --text "PR-101 3,350,000"
    python main.py ingest --file prices.xlsx
    python main.py build                 # rebuild the RAG index
    python main.py chat                  # terminal chat with memory
    python main.py ask "your question"   # single query

Add --offline before the command to test with no API key and no downloads.
=============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

from src import config, radar
from src.indexer import build_index, load_index
from src.llm import check_llm, get_llm
from src.loaders import load_all
from src.memory import BufferMemory, SummaryBufferMemory
from src.rag import RealEstateAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# =============================================================================
# CLI HANDLERS
# =============================================================================

def cmd_build(args):
    radar.init_db()
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
    radar.init_db()
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


def cmd_serve(args):
    import uvicorn
    if args.offline:
        config.OFFLINE_MODE = True
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)


def cmd_telegram(args):
    from src.telegram_bot import run_polling
    if args.offline:
        config.OFFLINE_MODE = True
    run_polling()


def cmd_ingest(args):
    from src import price_ingest
    radar.init_db()
    if args.file:
        path = Path(args.file)
        result = price_ingest.ingest_file(path.read_bytes(), path.name, source="cli", rebuild=False)
    else:
        text = args.text if args.text is not None else sys.stdin.read()
        llm = None if args.offline or args.no_llm else get_llm()
        result = price_ingest.ingest_text(text, source="cli", llm=llm, rebuild=False)
    print(result.summary("ar" if args.arabic else "en"))
    if (result.applied or result.added) and not args.no_rebuild:
        print("Rebuilding the RAG index...")
        price_ingest.rebuild_index_now(offline=args.offline)


def run_cli(argv=None):
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
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)
    sub.add_parser("telegram", help="listen to the Telegram price group").set_defaults(fn=cmd_telegram)
    i = sub.add_parser("ingest", help="apply a price message or sheet")
    src = i.add_mutually_exclusive_group()
    src.add_argument("--text", help="message text (reads stdin if neither --text nor --file)")
    src.add_argument("--file", help=".xlsx or .csv price sheet")
    i.add_argument("--no-llm", action="store_true", help="rule-based parser only")
    i.add_argument("--no-rebuild", action="store_true", help="skip the RAG index rebuild")
    i.add_argument("--arabic", action="store_true", help="print the summary in Arabic")
    i.set_defaults(fn=cmd_ingest)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    run_cli()
