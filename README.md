# RADAR Platform — Phase 2

> **Unified Real Estate CRM + RAG Chat + Multi-Agent Research Lab**

This project combines three capabilities behind one HTTP API (FastAPI) and CLI:

1. **RADAR CRM** — lead management, explainable opportunity matching, inventory tracking, bilingual drafts, human approval workflow, consent checks, kill switch, and audit trail.
2. **RAG Chat Assistant** — PDF and spreadsheet knowledge base with conversation memory, follow-up resolution, and source citations.
3. **Research Lab** *(new in Phase 2)* — upload any PDF for multi-agent analysis using a four-stage LangGraph pipeline: extraction → research → analysis → report.

## Quickstart

Run the entire platform from a single clean file:

```bash
pip install -r requirements.txt
python main.py build          # build the RAG index once
python main.py serve          # API on http://localhost:8000, try it at /docs
python main.py telegram       # listen to the daily price group (see below)
```

Or run the CLI commands from the same file:

```bash
python main.py build                                           # Build vector index
python main.py chat                                            # Interactive terminal chat
python main.py ask "What is the down payment for Palm Residence?"  # One-off question
```

The first launch creates `data/radar.db` with sample leads and seeds the unit inventory from `data/units.csv`.

## Architecture

```
Unified RADAR Platform
├─ CRM Workflow (LangGraph: src/crm_workflow.py)
│   ├─ Intake Agent       → loads active leads + available inventory
│   ├─ Matching Agent     → budget, location, bedroom filters + fit scoring
│   ├─ Opportunity Agent  → trigger relevance + deduplication
│   ├─ Recommendation Agent → bilingual draft + evidence
│   └─ Persistence        → save for human review + audit event
│
├─ RAG Chat (src/rag.py)
│   question → rewrite → FAISS top-k → LLM(context + memory) → answer + sources
│
└─ Research Lab (LangGraph: src/research_workflow.py)  ← Phase 2
    ├─ Extraction Agent   → PyMuPDF page-aware text extraction
    ├─ Research Agent     → source-grounded notes with page citations
    ├─ Analysis Agent     → themes, patterns, gaps, significance ranking
    └─ Report Agent       → structured Markdown report
```

## Project Structure

| Path | Purpose |
|---|---|
| `main.py` | **Unified entrypoint** (CLI: serve, telegram, ingest, build, chat, ask) |
| `src/api.py` | FastAPI service: units, leads, opportunities, chat, price ingest, research |
| `src/inventory.py` | Live unit search; feeds current prices into every chat answer |
| `src/price_ingest.py` | Parses daily price messages/sheets, applies them with history and approvals |
| `src/telegram_bot.py` | Telegram price-group listener (long polling) |
| `app.py` | CLI alias redirecting to `main.py` |
| `src/config.py` | Unified configuration (LLM providers, paths, limits) |
| `src/radar.py` | CRM domain services and SQLite data layer |
| `src/crm_workflow.py` | LangGraph CRM opportunity orchestrator |
| `src/research_workflow.py` | LangGraph research pipeline orchestrator |
| `src/report_generator.py` | Report persistence and export |
| `src/pdf_service.py` | Unified PDF extraction (PyMuPDF + PyPDF fallback) |
| `src/rag.py` | RAG pipeline with memory-aware follow-ups |
| `src/memory.py` | Buffer and summary-buffer conversation memory |
| `src/llm.py` | Multi-provider LLM client (OpenAI, Groq, Gemini, Ollama, etc.) |
| `src/loaders.py` | PDF and spreadsheet document loaders |
| `src/indexer.py` | FAISS embedding and vector index |
| `data/` | Sample PDFs, unit inventory CSV, and SQLite database |
| `eval/` | Evaluation framework and test set |
| `tests/` | Pytest wiring and integration tests |
| `generated_reports/` | Saved research reports |

## Phase 3 · Live prices and daily price automation

Unit prices change every day, so they live in the `units` table, not in the vector index.

* **Chat answers use live prices.** Every question is matched against the units table
  (unit code, project, area, bedrooms, budget, in Arabic or English) and the matching rows are
  added to the prompt as `[live inventory]` with their last-updated time. The prompt tells the
  model that live inventory beats any price written in a brochure.
* **Price history.** Every change is stored in `price_history` with its source
  (`GET /units/{unit_id}/history`).
* **Ingestion from the price group.** Messages like these are understood:

  ```
  PR-101 3,350,000
  NV-102 : 3.7 مليون
  PR-102 محجوزة
  Rai Valleys - 88 - 129 - E  18,300,000
  زيادة 5% على Nile View Towers
  ```

  An `.xlsx`/`.csv` sheet posted in the group is applied row by row (new units are added).
  Free-form messages the rules can't read go to the LLM, whose answer is checked so it can't
  invent unit codes.
* **Safety.** A change bigger than `PRICE_CHANGE_HOLD_PCT` (25% by default) is held until someone
  approves it (`GET /ingest/pending`, `POST /ingest/pending/{id}/decision`). Unknown units are
  rejected and reported. The same message is never applied twice.
* After each update the CRM opportunity agents run (price drops create opportunities) and the RAG
  index is rebuilt in the background.

### Telegram setup
1. Create a bot with @BotFather and set `TELEGRAM_BOT_TOKEN`.
2. In @BotFather run `/setprivacy` → Disable, so the bot can read group messages.
3. Add the bot to the price group and send `/chatid`. Put the number in `TELEGRAM_PRICE_CHAT_IDS`.
4. Run `python main.py telegram`. The bot replies in the group with what it changed.

### WhatsApp
WhatsApp's official API can't read group messages. Use any automation that can forward the
group's messages (n8n, Make, a forwarding number, or an unofficial WhatsApp Web bridge, which
risks a ban) to the webhook:

```bash
curl -X POST http://localhost:8000/ingest/webhook \
  -H "X-Ingest-Token: $INGEST_WEBHOOK_TOKEN" -H "Content-Type: application/json" \
  -d '{"source": "whatsapp", "message_id": "abc123", "text": "PR-101 3,350,000"}'
```

### Manual ingestion
```bash
python main.py ingest --text "PR-101 3,350,000"
python main.py ingest --file "today prices.xlsx"
```

## Configuration

Set your LLM provider in `.env`:

```bash
# Groq (free fast inference)
LLM_PROVIDER=groq
LLM_API_KEY=gsk_your_key
LLM_MODEL=llama-3.3-70b-versatile

# Or: openai, bigmodel, deepseek, openrouter, gemini, ollama
# See .env.example for all options
```

The Research Lab uses the same LLM provider — no separate Groq key needed.

RAG ingestion reads up to 5,000 rows per spreadsheet by default to keep index
builds and chat startup responsive. Set `MAX_RAG_SHEET_ROWS` in `.env` to adjust
that limit, then rebuild the index with `python main.py --offline build` or
`python main.py build`.

## Features by Tab

| Tab | Description |
|---|---|
| **Overview** | Opportunity inbox, workflow summary, dormant-lead scan |
| **💬 Chat** | Interactive RAG assistant with memory, sources, and follow-ups |
| **Leads** | Create, search, filter, profile, and manage sales leads |
| **Opportunities** | Explainable match review: fit factors, evidence, bilingual drafts, approve/snooze/dismiss |
| **Inventory** | View and edit unit inventory, trigger matching on price changes |
| **📄 Research Lab** | Upload PDF → four-agent analysis → structured report |
| **Insights & Imports** | CSV/XLSX import, pipeline charts, buyer demand vs inventory |
| **Agent Controls** | Autonomy level, kill switch, audit event log |
| **Architecture** | Full system architecture diagram |

## RAG Chat (CLI)

The original CLI chat assistant is still available:

```bash
python app.py build    # index PDFs + CSV
python app.py chat     # interactive chat with memory
```

## Testing

```bash
pytest -q                          # wiring tests, no API key needed
python -m eval.evaluate            # full evaluation with real models
python -m eval.evaluate --offline  # retrieval check only
```

## Phase 2 vs Phase 1

| What changed | Phase 1 | Phase 2 |
|---|---|---|
| Project Structure | Two separate projects (`final_project` + `radar_multi_agent`) | Single unified codebase in root |
| Entry point | Two separate apps (`radar_app.py` + `app.py`) | Single unified `main.py` (UI + CLI) |
| Research | Upload PDF → single Groq review | Four-agent LangGraph pipeline → structured report |
| Chat | CLI only | Streamlit chat UI with real-time sources + CLI |
| LLM backend | Separate Groq client for research | Unified multi-provider client for everything |
| Reports | None | Saved to `generated_reports/` with audit trail |
| PDF extraction | PyPDF only | PyMuPDF (primary) + PyPDF fallback |

## Known Limitations

- Small local models are weaker than large hosted ones, especially in Arabic.
- RAG list/filter questions can miss rows (top-k retrieval).
- Scanned (image-only) PDFs need OCR (not included).
- Local SQLite; no auth, multi-tenant deployment, or external messaging.
- Level 2 automated sending is not implemented.
- Research pipeline processes documents sequentially (not streaming).

## Tested With

langchain 1.4.9, langchain-core 1.7.6, langchain-community 0.4.2, langgraph 1.x, PyMuPDF 1.24+, FastAPI 0.110+.
