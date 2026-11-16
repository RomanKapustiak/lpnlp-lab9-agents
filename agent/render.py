from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

import markdown as md
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fpdf import FPDF
from fpdf.fonts import TextStyle

from .config import CHART_RE, FONT_DIR, OUT_DIR, TZ


# Charts
def chart_figure(spec: dict) -> go.Figure:
    """Interactive chart for the Canvas in the app (hover shows all lines for one x)."""
    fig = go.Figure()
    for s in spec["series"]:
        fig.add_trace(go.Scatter(x=spec["x"], y=s["values"], mode="lines+markers", name=s["name"]))
    fig.update_layout(
        title=spec["title"],
        xaxis_title=spec.get("x_label") or None,
        yaxis_title=spec.get("y_label") or None,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def chart_png(spec: dict, path: Path) -> Path:
    """Static version of the same chart for the PDF."""
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    xs = list(range(len(spec["x"])))
    for s in spec["series"]:
        ax.plot(xs, s["values"], marker="o" if len(xs) <= 40 else None, markersize=3, linewidth=2, label=s["name"])
    step = max(1, len(xs) // 12)  # at most ~12 x-axis labels
    ax.set_xticks(xs[::step], [spec["x"][i] for i in xs[::step]], rotation=35, ha="right", fontsize=8)
    ax.set_title(spec["title"], fontsize=12)
    ax.set_xlabel(spec.get("x_label", ""))
    ax.set_ylabel(spec.get("y_label", ""))
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if len(spec["series"]) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# Markdown -> HTML
LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")


def normalize_markdown(text: str) -> str:
    """Fixes typical LLM markdown for Python-Markdown: adds blank lines around lists and tables
    and makes nested list indents multiples of 4 spaces (nesting level is tracked with a stack)."""
    out: list[str] = []
    prev, in_code, stack = "", False, []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code:
            m = LIST_ITEM_RE.match(line)
            prev_is_item = bool(LIST_ITEM_RE.match(prev))
            if m:
                indent = len(m[1].expandtabs(4))
                while stack and stack[-1] > indent:
                    stack.pop()
                if not stack or stack[-1] < indent:
                    stack.append(indent)
                line = " " * 4 * (len(stack) - 1) + line.lstrip()
                if prev.strip() and not prev_is_item and not prev.startswith((" ", "\t")):
                    out.append("")
            elif line.strip() and not line.startswith((" ", "\t")):
                stack = []
                is_table = line.lstrip().startswith("|")
                if prev_is_item or (is_table and prev.strip() and not prev.lstrip().startswith("|")):
                    out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def _fix_table_cell(m: re.Match) -> str:
    """fpdf2 does not support <br> inside table cells; also align cells to the left."""
    attrs = m[2] or ' align="left"'
    content = re.sub(r"<br\s*/?>", " / ", m[3])
    return f"<{m[1]}{attrs}>{content}</{m[1]}>"


# PDF
class ReportPDF(FPDF):
    def footer(self):  # page number at the bottom of every page
        self.set_y(-12)
        self.set_font("DejaVu", size=8)
        self.set_text_color(130)
        self.cell(0, 8, str(self.page_no()), align="C")


def render_pdf(title: str, body: str, charts: dict) -> str:
    """Report markdown -> PDF file. Chart placeholders are replaced with PNG images. Returns the file path."""
    folder = OUT_DIR / uuid.uuid4().hex[:12]  # unique folder: several users/versions never overwrite each other
    folder.mkdir(parents=True, exist_ok=True)

    def chart_img(m: re.Match) -> str:
        png = chart_png(charts[m.group(1)], folder / f"{m.group(1)}.png")
        return f'\n\n<img src="{png}" width="480">\n\n'

    body_html = md.markdown(normalize_markdown(CHART_RE.sub(chart_img, body)),
                            extensions=["tables", "fenced_code", "sane_lists"])
    body_html = re.sub(r"<(td|th)([^>]*)>(.*?)</\1>", _fix_table_cell, body_html, flags=re.S)

    pdf = ReportPDF(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, margin=18)
    for style, fname in (("", "DejaVuSans"), ("B", "DejaVuSans-Bold"), ("I", "DejaVuSans-Oblique"),
                         ("BI", "DejaVuSans-BoldOblique")):
        pdf.add_font("DejaVu", style, str(FONT_DIR / f"{fname}.ttf"))
    pdf.add_font("DejaVuMono", "", str(FONT_DIR / "DejaVuSansMono.ttf"))
    pdf.set_title(title)
    pdf.add_page()
    # title + generation time
    pdf.set_font("DejaVu", "B", 18)
    pdf.multi_cell(0, 9, title, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(120)
    pdf.cell(0, 8, f"Згенеровано: {datetime.now(TZ):%Y-%m-%d %H:%M}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(2)
    # body
    pdf.set_font("DejaVu", "", 11)
    accent = (31, 58, 104)
    pdf.write_html(
        body_html,
        font_family="DejaVu",
        pre_code_font="DejaVuMono",
        table_line_separators=True,
        warn_on_tags_not_matching=False,
        li_prefix_color=accent,
        tag_styles={
            "h1": TextStyle("DejaVu", "B", 16, color=accent, t_margin=6, b_margin=3),
            "h2": TextStyle("DejaVu", "B", 14, color=accent, t_margin=6, b_margin=2),
            "h3": TextStyle("DejaVu", "B", 12, color=accent, t_margin=4, b_margin=2),
            "h4": TextStyle("DejaVu", "B", 11, color=accent, t_margin=3, b_margin=1),
        },
    )
    slug = re.sub(r"[^\w-]+", "_", title).strip("_")[:60] or "report"
    path = folder / f"{slug}.pdf"
    pdf.output(str(path))
    return str(path)
