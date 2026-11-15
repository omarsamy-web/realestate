"""LangGraph orchestration for the Phase 2 Research Lab.

The research pipeline is a four-node sequential graph:

    extraction_agent → research_agent → analysis_agent → report_agent

Each node has a single responsibility and passes explicit state forward.
The LLM calls use the unified ``ChatOpenAICompatible`` provider so any
configured backend (Groq, OpenAI, Gemini, Ollama …) works automatically.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config
from .llm import get_llm
from .pdf_service import ExtractedDocument, chunk_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    # Inputs
    document: ExtractedDocument
    task: str
    # Intermediate
    extracted_text: str
    research_notes: str
    analysis: str
    # Output
    report: str
    error: str | None
    created_at: str


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(node: str, **kw: Any) -> None:
    logger.info("Research agent node=%s %s", node, kw)


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------

def extraction_agent(state: ResearchState) -> dict:
    """Validate the extracted document and prepare text for downstream agents."""
    doc = state["document"]
    if not doc.text.strip():
        return {"error": "The document contains no extractable text.", "extracted_text": ""}

    text = doc.text
    _log("extraction", pages=doc.page_count, chars=len(text))
    return {
        "extracted_text": text,
        "created_at": _ts(),
    }


def research_agent(state: ResearchState) -> dict:
    """Read through the extracted text and produce structured research notes.

    For long documents the text is processed in chunks; each chunk gets a
    focused summary that preserves page references.
    """
    if state.get("error"):
        return {}

    text = state["extracted_text"]
    task = state.get("task") or "Summarize the document and identify its key findings."
    llm = get_llm()
    parts = chunk_text(text)
    notes: list[str] = []

    for idx, part in enumerate(parts, 1):
        prompt = (
            "You are a meticulous research assistant.  Read ONLY the supplied source text.\n"
            "Do NOT invent facts, figures, or page references that are not in the text.\n"
            "Include [Page N] citations when the text contains page markers.\n\n"
            f"Research focus: {task}\n\n"
            f"Source text (part {idx} of {len(parts)}):\n{part}\n\n"
            "Produce structured research notes with key findings, data points, "
            "and direct quotes where relevant."
        )
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response)).strip()
        notes.append(content)

    combined = "\n\n---\n\n".join(notes) if len(notes) > 1 else notes[0]
    _log("research", chunks_processed=len(parts), notes_length=len(combined))
    return {"research_notes": combined}


def analysis_agent(state: ResearchState) -> dict:
    """Cross-reference the research notes and produce a critical analysis."""
    if state.get("error"):
        return {}

    llm = get_llm()
    prompt = (
        "You are an analytical research reviewer.\n\n"
        "Below are structured research notes extracted from a document.  "
        "Your job is to:\n"
        "1. Identify the most important themes and patterns.\n"
        "2. Highlight any contradictions, gaps, or limitations in the source.\n"
        "3. Rank the key findings by significance.\n"
        "4. Note questions that remain unanswered by the source material.\n\n"
        f"Research notes:\n{state['research_notes']}\n\n"
        "Produce a concise critical analysis."
    )
    response = llm.invoke(prompt)
    analysis = getattr(response, "content", str(response)).strip()
    _log("analysis", analysis_length=len(analysis))
    return {"analysis": analysis}


def report_agent(state: ResearchState) -> dict:
    """Generate a polished, structured Markdown report from the analysis."""
    if state.get("error"):
        return {"report": f"**Error:** {state['error']}"}

    llm = get_llm()
    doc = state["document"]
    task = state.get("task") or "General document review"

    prompt = (
        "You are a professional report writer.\n\n"
        "Using the analysis and research notes below, produce a well-structured "
        "Markdown report with the following sections:\n\n"
        "1. **Executive Summary** – 2-3 sentence overview.\n"
        "2. **Key Findings** – numbered list of the most important discoveries.\n"
        "3. **Detailed Analysis** – expand on each finding with evidence and page references.\n"
        "4. **Gaps & Limitations** – what the source does not cover.\n"
        "5. **Recommendations** – actionable next steps based on the findings.\n\n"
        f"Document: {doc.filename} ({doc.page_count} pages)\n"
        f"Research focus: {task}\n\n"
        f"Research notes:\n{state['research_notes'][:6000]}\n\n"
        f"Critical analysis:\n{state['analysis'][:4000]}\n\n"
        "Write the final report now."
    )
    response = llm.invoke(prompt)
    report = getattr(response, "content", str(response)).strip()
    _log("report", report_length=len(report))
    return {"report": report}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_research_graph():
    """Build the compiled sequential research workflow."""
    graph = StateGraph(ResearchState)
    graph.add_node("extraction_agent", extraction_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("analysis_agent", analysis_agent)
    graph.add_node("report_agent", report_agent)

    graph.add_edge(START, "extraction_agent")
    graph.add_edge("extraction_agent", "research_agent")
    graph.add_edge("research_agent", "analysis_agent")
    graph.add_edge("analysis_agent", "report_agent")
    graph.add_edge("report_agent", END)

    return graph.compile()


_RESEARCH_GRAPH = build_research_graph()


def run_research_workflow(
    document: ExtractedDocument,
    task: str = "",
) -> dict:
    """Execute the full research pipeline and return the final state.

    Returns a dict with keys: ``report``, ``research_notes``, ``analysis``,
    ``created_at``, and optionally ``error``.
    """
    result = _RESEARCH_GRAPH.invoke({
        "document": document,
        "task": task,
    })
    return {
        "report": result.get("report", ""),
        "research_notes": result.get("research_notes", ""),
        "analysis": result.get("analysis", ""),
        "created_at": result.get("created_at", _ts()),
        "error": result.get("error"),
    }

