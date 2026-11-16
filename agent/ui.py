from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from .config import CHART_RE
from .render import chart_figure

TOOL_LABELS = {
    "get_weather": "Погода",
    "get_exchange_rate": "Курс НБУ",
    "search_web": "Пошук в інтернеті",
    "fetch_url": "Читаю сторінку",
    "make_line_chart": "Графік",
    "create_report": "Створюю документ",
    "edit_report": "Редагую документ",
    "deep_research": "Глибоке дослідження",
}

DESCRIPTION = """### Універсальний ReAct-агент
Агент на **Google ADK + Gemini** міркує й сам обирає інструменти: погода, курс НБУ, пошук в інтернеті,
читання сторінок, **глибоке дослідження** (агент-інструмент, що працює за планом), **графіки** та
**Canvas-документ** з експортом у **PDF**.

Кроки агента (міркування та виклики інструментів) видно в чаті."""

EXAMPLES = [
    "Яка погода буде у Львові найближчі 7 днів? Побудуй графік мін/макс температури.",
    "Як змінювався курс долара та євро за останні 30 днів? Зроби звіт з графіком.",
    "Зроби глибоке дослідження: які електромобілі найпопулярніші в Україні у 2026 році і чому? Оформи звіт у PDF.",
    "Прочитай https://docs.streamlit.io/deploy/tutorials і стисло поясни, де можна задеплоїти Streamlit-застосунок.",
]


# Formatting tool calls

def _short(obj, limit: int = 1500) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    return text if len(text) <= limit else text[:limit] + "\n… (обрізано)"


def format_call(name: str, args: dict) -> str:
    """Tool arguments as markdown. The deep_research plan is shown in a readable form."""
    if name == "deep_research":
        questions = "\n".join(f"{i}. {q}" for i, q in enumerate(args.get("sub_questions", []), 1))
        out = f"**План дослідження**\n\n**Мета:** {args.get('goal', '')}\n\n**Підпитання:**\n{questions}"
        for label, key in (("Контекст", "context"), ("Обмеження", "constraints"), ("Формат", "output_format")):
            if args.get(key):
                out += f"\n\n**{label}:** {args[key]}"
        return out
    if name in ("create_report", "edit_report"):  # long markdown - show only the beginning
        args = {k: (v if len(str(v)) < 300 else str(v)[:300] + "…") for k, v in args.items()}
    return f"```json\n{_short(args)}\n```"


def format_result(response) -> str:
    if isinstance(response, dict) and set(response) == {"result"}:  # AgentTool wraps text as {"result": ...}
        response = response["result"]
    if isinstance(response, str):
        return _short(response, 6000)
    return f"```json\n{_short(response)}\n```"


def is_error(response) -> bool:
    return isinstance(response, dict) and response.get("status") == "error"


# Chat history
# A message is a dict: {"role": "user", "content"} or {"role": "assistant", "steps", "text", "trace_url"}.
# A step is {"kind": "thought" | "tool" | "error", ...}.

def render_step(step: dict) -> None:
    if step["kind"] == "thought":
        with st.expander("Міркування"):
            st.markdown(step["body"])
    elif step["kind"] == "tool":
        with st.status(step["title"], state=step["state"], expanded=False):
            st.markdown(step["body"])
            if step["result"]:
                st.markdown("**Результат:**\n\n" + step["result"])
    elif step["kind"] == "error":
        st.error(step["body"])


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return
        for step in msg["steps"]:
            render_step(step)
        if msg["text"]:
            st.markdown(msg["text"])
        if msg.get("trace_url"):
            st.caption(f"[Трейс у Langfuse]({msg['trace_url']})")


# Canvas
def render_canvas() -> None:
    """Report text with interactive charts in place of [[chart:ID]] placeholders + PDF download."""
    canvas = st.session_state.canvas
    report, charts = canvas.get("report"), canvas.get("charts", {})
    st.markdown("### Canvas")
    used: set[str] = set()
    if report:
        pdf = Path(report["pdf"])
        col1, col2 = st.columns(2)
        col1.download_button("Завантажити PDF", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf",
                             on_click="ignore", key=f"pdf_{report['version']}", width="stretch")
        col2.caption(f"Версія {report['version']}")
        with st.container(height=560, border=True):
            st.markdown(f"## {report['title']}")
            parts = CHART_RE.split(report["markdown"])  # [text, chart_id, text, chart_id, ...]
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    if part.strip():
                        st.markdown(part)
                elif part in charts:
                    used.add(part)
                    st.plotly_chart(chart_figure(charts[part]), key=f"inline_{part}_{report['version']}")
    rest = [cid for cid in charts if cid not in used]
    if rest:  # charts that are not inserted into the report
        chosen = st.selectbox("Графіки поза документом", rest, index=len(rest) - 1,
                              format_func=lambda cid: f"{charts[cid]['title']} ({cid})")
        st.plotly_chart(chart_figure(charts[chosen]), key=f"extra_{chosen}")
