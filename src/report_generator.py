"""Report persistence and export helpers (Phase 2).

Saves generated research reports to the ``generated_reports/`` directory
and records them in the RADAR SQLite database for the audit trail.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from . import config


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(name: str, maxlen: int = 40) -> str:
    slug = re.sub(r"[^\w]+", "_", name.lower()).strip("_")
    return slug[:maxlen]


def save_report(
    filename: str,
    report_md: str,
    research_notes: str = "",
    analysis: str = "",
    task: str = "",
) -> Path:
    """Persist a research report as a Markdown file and return its path.

    Also records an audit event in the RADAR database when the DB is available.
    """
    reports_dir = Path(config.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _slug(Path(filename).stem)
    out_name = f"{stamp}_{slug}.md"
    out_path = reports_dir / out_name

    header = (
        f"---\n"
        f"source: {filename}\n"
        f"task: {task or 'General review'}\n"
        f"generated: {_ts()}\n"
        f"---\n\n"
    )
    out_path.write_text(header + report_md, encoding="utf-8")

    # Best-effort audit event
    try:
        from . import radar
        with radar.connect() as db:
            db.execute(
                "INSERT INTO events(kind, details, created_at) VALUES(?,?,?)",
                ("Research report", f"Generated report for {filename}: {out_name}", _ts()),
            )
    except Exception:
        pass  # DB may not exist yet; non-critical

    return out_path


def list_reports() -> list[dict]:
    """Return metadata for all saved reports, newest first."""
    reports_dir = Path(config.REPORTS_DIR)
    if not reports_dir.exists():
        return []
    results = []
    for p in sorted(reports_dir.glob("*.md"), reverse=True):
        results.append({
            "filename": p.name,
            "path": str(p),
            "size": p.stat().st_size,
            "modified": datetime.fromtimestamp(
                p.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds"),
        })
    return results

