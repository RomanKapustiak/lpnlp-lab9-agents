from __future__ import annotations

import json

from google.adk.tools import ToolContext
from pydantic import BaseModel, Field

from .config import CHART_RE
from .render import render_pdf


def _from_json(value):
    """Models sometimes send a nested argument as a JSON string instead of an object - decode it."""
    return json.loads(value) if isinstance(value, str) else value


def _as_list(value) -> list:
    """List argument that may arrive as a JSON string or as a single object."""
    value = _from_json(value) or []
    return [value] if isinstance(value, dict) else value


# Charts
class ChartSeries(BaseModel):
    name: str = Field(description="Line name (legend)")
    values: list[float] = Field(description="Y values; same length as x")


def make_line_chart(
    tool_context: ToolContext,
    title: str,
    x: list[str],
    series: list[ChartSeries],
    x_label: str = "",
    y_label: str = "",
    replace_chart_id: str = "",
) -> dict:
    """Creates a line chart (one or more lines) or updates an existing one. The interactive chart appears
    in the Canvas; to put it into the report, insert the placeholder returned by this tool.

    Args:
        title: Chart title.
        x: X-axis labels (dates, months, categories) in ascending order.
        series: Chart lines; each has a name and values (as many values as x labels).
        x_label: X-axis label.
        y_label: Y-axis label (with units).
        replace_chart_id: Id of an existing chart (e.g. "chart_1") to replace its data (add a line, change
            the period). The report that contains its placeholder is updated automatically.
            Empty = create a new chart.
    """
    try:
        parsed = [ChartSeries.model_validate(_from_json(s)) for s in _as_list(series)]
    except (ValueError, TypeError) as e:
        return {"status": "error", "error": "series має бути списком об'єктів {\"name\": str, \"values\": [числа]}. "
                                            f"Деталі: {str(e)[:300]}"}
    if not x or not parsed:
        return {"status": "error", "error": "Потрібні x і хоча б одна серія."}
    bad = [s.name for s in parsed if len(s.values) != len(x)]
    if bad:
        return {"status": "error", "error": f"Довжина values не збігається з x ({len(x)}) у серіях: {bad}"}

    charts = dict(tool_context.state.get("charts", {}))
    if replace_chart_id and replace_chart_id not in charts:
        return {"status": "error", "error": f"Графіка {replace_chart_id} немає. Доступні: {sorted(charts)}"}
    chart_id = replace_chart_id or f"chart_{len(charts) + 1}"
    charts[chart_id] = {"title": title, "x": list(x), "series": [s.model_dump() for s in parsed],
                        "x_label": x_label, "y_label": y_label}
    tool_context.state["charts"] = charts

    # the PDF contains a picture of the chart, so a report that uses this chart must be re-rendered
    report = tool_context.state.get("report")
    if replace_chart_id and report and f"[[chart:{chart_id}]]" in report["markdown"]:
        result = _save_report(tool_context, report["title"], report["markdown"])
        if result["status"] != "ok":
            return result
        return {"status": "ok", "chart_id": chart_id,
                "message": f"Графік {chart_id} оновлено в документі. "
                           "Онови текст документа під нові дані, якщо він їх стосується."}
    return {"status": "ok", "chart_id": chart_id, "placeholder": f"[[chart:{chart_id}]]",
            "hint": "Щоб додати графік у документ, встав placeholder окремим рядком."}


# Report
def _save_report(tool_context: ToolContext, title: str, body: str) -> dict:
    """Validates chart placeholders, renders the PDF and stores the report in state.
    Version = number of user turns that changed the document (several saves in one turn = one version)."""
    title = title.strip()
    charts = tool_context.state.get("charts", {})
    missing = sorted(set(CHART_RE.findall(body)) - set(charts))
    if missing:
        return {"status": "error", "error": f"Невідомі графіки: {missing}. Доступні: {sorted(charts)}"}
    try:
        pdf = render_pdf(title, body, charts)
    except Exception as e:  # noqa: BLE001 - give the error back to the model so it can fix the markdown
        return {"status": "error", "error": f"Не вдалося згенерувати PDF: {e}"}
    prev = tool_context.state.get("report") or {}
    turn = tool_context.invocation_id  # one ADK invocation = one user message
    version = prev.get("version", 0) + (prev.get("turn") != turn)
    tool_context.state["report"] = {"title": title, "markdown": body, "version": version, "pdf": pdf, "turn": turn}
    return {"status": "ok", "version": version, "message": "Документ відображено в Canvas, PDF доступний для завантаження."}


def create_report(tool_context: ToolContext, title: str, markdown: str) -> dict:
    """Canvas: creates (or fully replaces) the document that the user sees on the right and can download as PDF.

    Args:
        title: Document title (added automatically, do not repeat it in markdown).
        markdown: Document body in Markdown: ## headings, lists, tables, **bold**, links.
            Charts are inserted on their own line with the [[chart:ID]] placeholder from make_line_chart.
    """
    return _save_report(tool_context, title, markdown)


class ReportEdit(BaseModel):
    old_str: str = Field(description="Exact fragment of the document (must occur exactly once)")
    new_str: str = Field(description="Replacement text (empty = delete the fragment)")


def edit_report(tool_context: ToolContext, edits: list[ReportEdit], new_title: str = "") -> dict:
    """Canvas: targeted edits of the current document. All edits are applied together as one new version.

    Args:
        edits: List of replacements {old_str, new_str}; each old_str is an exact fragment that occurs once.
            Empty list if only the title changes.
        new_title: New document title (empty = keep the current one).
    """
    report = tool_context.state.get("report")
    if not report:
        return {"status": "error", "error": "Документа ще немає — спочатку виклич create_report."}
    try:
        parsed = [ReportEdit.model_validate(_from_json(e)) for e in _as_list(edits)]
    except (ValueError, TypeError) as e:
        return {"status": "error", "error": f"edits має бути списком {{old_str, new_str}}: {str(e)[:300]}"}
    if not parsed and not new_title.strip():
        return {"status": "error", "error": "Немає жодної правки."}

    body = report["markdown"]
    for i, edit in enumerate(parsed, 1):
        count = body.count(edit.old_str) if edit.old_str else 0
        if count != 1:  # all-or-nothing: if one edit does not match, nothing is applied
            problem = "не знайдено" if count == 0 else f"зустрічається {count} рази — додай контексту"
            return {"status": "error", "error": f"Правка {i}: old_str {problem}. Жодну правку не застосовано.",
                    "document": body[:6000]}
        body = body.replace(edit.old_str, edit.new_str, 1)
    return _save_report(tool_context, new_title.strip() or report["title"], body)
