# RADAR Platform — Phase 2

> **Unified Real Estate CRM + RAG Chat + Multi-Agent Research Lab**

This project combines three capabilities into a single Streamlit application:

1. **RADAR CRM** — lead management, explainable opportunity matching, inventory tracking, bilingual drafts, human approval workflow, consent checks, kill switch, and audit trail.
2. **RAG Chat Assistant** — PDF and spreadsheet knowledge base with conversation memory, follow-up resolution, and source citations.
3. **Research Lab** *(new in Phase 2)* — upload any PDF for multi-agent analysis using a four-stage LangGraph pipeline: extraction → research → analysis → report.

## Quickstart

Run the entire platform from a single clean file:

```bash
pip install -r requirements.txt
streamlit run main.py
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
| `main.py` | **Unified application entrypoint** (Streamlit dashboard + CLI) |
| `app.py` | CLI alias redirecting to `main.py` |
| `radar_app.py` | Streamlit alias redirecting to `main.py` |
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

langchain 1.4.9, langchain-core 1.7.6, langchain-community 0.4.2, langgraph 1.x, PyMuPDF 1.24+, Streamlit 1.36+.
