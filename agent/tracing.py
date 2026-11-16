from __future__ import annotations

import contextlib
import os

import streamlit as st


@st.cache_resource(show_spinner=False)  # once per process, not on every Streamlit rerun
def init_langfuse():
    """Returns (client or None, status text for the sidebar)."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None, "вимкнено (немає LANGFUSE_* ключів)"
    try:
        from langfuse import Langfuse
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor

        client = Langfuse()
        if not client.auth_check():
            return None, "помилка автентифікації"
        GoogleADKInstrumentor().instrument()
        return client, "підключено"
    except Exception as e:  # noqa: BLE001
        return None, f"помилка: {type(e).__name__}"


@contextlib.contextmanager
def langfuse_turn(prompt: str, model_name: str):
    """Wraps one chat turn into a trace; traces of one conversation are grouped into a Langfuse session."""
    client, _ = init_langfuse()
    if client is None:
        yield None
        return
    from langfuse import propagate_attributes

    with propagate_attributes(
        session_id=st.session_state.adk_session_id,
        user_id="streamlit-user",
        trace_name="chat-turn",
        tags=[model_name],
    ), client.start_as_current_observation(name="chat-turn", as_type="agent", input=prompt) as obs:
        yield obs


def trace_url(obs) -> str:
    client, _ = init_langfuse()
    return client.get_trace_url(trace_id=obs.trace_id)
