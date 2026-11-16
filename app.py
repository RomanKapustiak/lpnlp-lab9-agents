"""Universal ReAct agent: Google ADK + Gemini + Streamlit.

Run: streamlit run app.py

Code map:
  app.py             - this file: Streamlit page and running one agent turn
  agent/agents.py    - assembling the agents: main ReAct agent + deep_research as a tool
  agent/prompts.py   - system prompts
  agent/tools.py     - weather, NBU rates, web search, reading pages
  agent/canvas.py    - charts and the report document (create / edit)
  agent/render.py    - charts and PDF rendering
  agent/ui.py        - rendering chat messages, agent steps and the Canvas
  agent/tracing.py   - Langfuse
  agent/config.py    - models, API key, constants
"""

from __future__ import annotations

import asyncio

import streamlit as st
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent.agents import build_agent, friendly_error, make_model
from agent.config import APP_NAME, CHART_RE, EMOJI_RE, ENV_API_KEY, MODELS, RUN_CONFIG, USER_ID
from agent.tracing import init_langfuse, langfuse_turn, trace_url
from agent.ui import (DESCRIPTION, EXAMPLES, TOOL_LABELS, format_call, format_result, is_error,
                      render_canvas, render_message, render_step)


async def run_turn(prompt: str, api_key: str, model_name: str) -> dict:
    """Runs the agent on one user message and shows its steps live. Returns the message for the chat history."""
    ss = st.session_state
    # Streamlit re-executes this script on every interaction, so the agent and Runner are rebuilt each turn
    # (cheap, and the model can be switched mid-conversation). The conversation itself - history and
    # state (report, charts) - lives in the ADK session that is kept in st.session_state.
    runner = Runner(app_name=APP_NAME, agent=build_agent(make_model(model_name, api_key)),
                    session_service=ss.adk_service)
    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    turn = {"role": "assistant", "steps": [], "text": ""}
    running = {}  # function_call.id -> (status widget, step) for tools that are still running
    with st.chat_message("assistant"), st.spinner("Агент думає…"):
        steps_area, text_area = st.container(), st.empty()
        try:
            # ADK runs the ReAct loop and yields an event for every model response and every tool result
            async for event in runner.run_async(user_id=USER_ID, session_id=ss.adk_session_id,
                                                new_message=message, run_config=RUN_CONFIG):
                for part in event.content.parts if event.content and event.content.parts else []:
                    if part.thought and part.text:  # Reason
                        step = {"kind": "thought", "body": part.text}
                        turn["steps"].append(step)
                        with steps_area:
                            render_step(step)
                    elif part.function_call:  # Act
                        call = part.function_call
                        step = {"kind": "tool", "title": f"{TOOL_LABELS.get(call.name, call.name)} · `{call.name}`",
                                "state": "running", "body": format_call(call.name, call.args or {}), "result": ""}
                        turn["steps"].append(step)
                        with steps_area:
                            widget = st.status(step["title"], state="running", expanded=call.name == "deep_research")
                            widget.markdown(step["body"])
                        running[call.id] = (widget, step)
                    elif part.function_response:  # Observation
                        response = part.function_response
                        widget, step = running.pop(response.id, (None, None))
                        if step:
                            step["result"] = format_result(response.response)
                            step["state"] = "error" if is_error(response.response) else "complete"
                            widget.markdown("**Результат:**\n\n" + step["result"])
                            widget.update(state=step["state"], expanded=False)
                    elif part.text and event.author == "assistant":  # final answer
                        turn["text"] += CHART_RE.sub("", EMOJI_RE.sub("", part.text))  # charts live in Canvas
                        text_area.markdown(turn["text"])
        except Exception as e:  # noqa: BLE001 - e.g. quota exceeded or invalid key
            step = {"kind": "error", "body": friendly_error(e)}
            turn["steps"].append(step)
            with steps_area:
                render_step(step)
    for _, step in running.values():  # tools interrupted by an error
        step["state"] = "error"

    # tools wrote the report and charts into session state -> copy them for the Canvas panel
    session = await ss.adk_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=ss.adk_session_id)
    ss.canvas = {"report": session.state.get("report"), "charts": session.state.get("charts", {})}
    return turn


async def run_traced_turn(prompt: str, api_key: str, model_name: str) -> dict:
    """run_turn inside a Langfuse trace (if Langfuse is configured)."""
    with langfuse_turn(prompt, model_name) as obs:
        turn = await run_turn(prompt, api_key, model_name)
        if obs is not None:
            errors = [s["body"] for s in turn["steps"] if s["kind"] == "error"]
            if errors:
                obs.update(output=turn["text"] or None, level="ERROR", status_message=errors[0])
            else:
                obs.update(output=turn["text"])
            turn["trace_url"] = trace_url(obs)
    return turn


def reset_conversation() -> None:
    ss = st.session_state
    ss.adk_service = InMemorySessionService()
    ss.adk_session_id = asyncio.run(ss.adk_service.create_session(app_name=APP_NAME, user_id=USER_ID)).id
    ss.messages = []  # chat history for display (the model's own history is in the ADK session)
    ss.canvas = {}


def main() -> None:
    st.set_page_config(page_title="Universal ReAct Agent", layout="wide")
    ss = st.session_state
    if "adk_service" not in ss:
        reset_conversation()

    with st.sidebar:
        st.markdown(DESCRIPTION)
        key_input = st.text_input(
            "Gemini API key", type="password",
            placeholder="використовується ключ з .env" if ENV_API_KEY else "AIza…",
            help="Ключ з Google AI Studio (aistudio.google.com/apikey). Не зберігається.",
        )
        model_name = st.selectbox("Модель агента", MODELS, accept_new_options=True)
        st.caption(f"Langfuse: {init_langfuse()[1]}")
        if st.button("Нова розмова", width="stretch"):
            reset_conversation()
            st.rerun()
        st.markdown("**Приклади запитів**")
        for i, example in enumerate(EXAMPLES):
            if st.button(example, key=f"ex_{i}", width="stretch"):
                ss.pending = example
    api_key = key_input.strip() or ENV_API_KEY

    # the Canvas column appears only after the agent has created a report or a chart
    if ss.canvas.get("report") or ss.canvas.get("charts"):
        chat_col, canvas_col = st.columns([3, 2], gap="large")
        with canvas_col:
            render_canvas()
    else:
        chat_col = st.container()
    with chat_col:
        chat_box = st.container(height=680, border=False)
        with chat_box:
            for msg in ss.messages:
                render_message(msg)
        prompt = st.chat_input("Запитай щось…") or ss.pop("pending", None)

    if prompt:
        if not api_key:
            chat_box.warning("Введи Gemini API key у боковій панелі.")
            return
        user_msg = {"role": "user", "content": prompt}
        ss.messages.append(user_msg)
        with chat_box:
            render_message(user_msg)
            turn = asyncio.run(run_traced_turn(prompt, api_key, model_name))
        ss.messages.append(turn)
        st.rerun()  # redraw the Canvas with the new report/charts


if __name__ == "__main__":
    main()
