from __future__ import annotations

import re
from collections import Counter

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import AgentTool, ToolContext
from google.genai import types
from pydantic import BaseModel, Field

from .canvas import create_report, edit_report, make_line_chart
from .prompts import main_instruction, research_instruction
from .tools import fetch_url, get_exchange_rate, get_weather, search_web

RESEARCH_BUDGET = {"search_web": 4, "fetch_url": 8}  # max tool calls per research: caps LLM calls and quota


def make_model(model_name: str, api_key: str) -> Gemini:
    return Gemini(model=model_name, client_kwargs={"api_key": api_key},
                  retry_options=types.HttpRetryOptions(attempts=3, initial_delay=3.0))


# Errors

def friendly_error(err: Exception) -> str:
    """Turns Gemini API errors into a short message for the user."""
    text = str(err)
    if "PerDay" in text and ("429" in text or "RESOURCE_EXHAUSTED" in text):
        return ("Вичерпано денний ліміт безкоштовного тарифу Gemini для цієї моделі. Ліміт скидається "
                "опівночі за тихоокеанським часом (10:00 за Києвом). Обери іншу модель у боковій панелі.")
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "Вичерпано ліміт запитів Gemini (429). Зачекай хвилину або обери іншу модель у боковій панелі."
    if "API_KEY_INVALID" in text or "API key not valid" in text:
        return "Невірний Gemini API key."
    if "404" in text or "NOT_FOUND" in text:
        return "Модель недоступна для цього API-ключа (404). Обери іншу модель у боковій панелі."
    return f"Помилка: {type(err).__name__}: {re.sub(r'\s+', ' ', text)[:500]}"


def tool_error_to_result(tool, args, tool_context, error: Exception) -> dict:
    """on_tool_error_callback: an exception inside any tool (including deep_research) does not crash the turn -
    the model receives it as a normal tool result and can retry or explain it to the user."""
    return {"status": "error", "error": friendly_error(error)}


# deep_research

class ResearchPlan(BaseModel):
    """Input of deep_research. AgentTool uses this schema as the tool's arguments, so the main agent
    has to pass a structured plan instead of a single query."""

    goal: str = Field(description="Main goal of the research: what exactly to find out and why")
    context: str = Field(default="", description="Context from the conversation: who the user is, what is "
                                                 "already known, how the result will be used")
    sub_questions: list[str] = Field(description="3-6 specific sub-questions to answer")
    search_hints: list[str] = Field(default_factory=list, description="Starting search queries, any language")
    constraints: str = Field(default="", description="Constraints: time period, region, source types, language")
    output_format: str = Field(default="Структурований Markdown-звіт з висновками",
                               description="Desired result format (comparison table, list, key points, ...)")


def build_research_tool(model) -> AgentTool:
    """deep_research is a separate ReAct agent with its own prompt and tools; it decides itself when to stop.
    The main agent only gets its final report, so search results do not fill the main agent's context."""
    used = Counter()  # tool calls in this turn (agents are rebuilt for every turn)

    def check_budget(tool, args, tool_context: ToolContext):
        # before_tool_callback: returning a dict skips the real tool call and gives this dict to the model
        used[tool.name] += 1
        if used[tool.name] > RESEARCH_BUDGET[tool.name]:
            return {"status": "limit", "message": f"Бюджет {tool.name} на дослідження вичерпано. "
                                                  "Більше не викликай інструменти — пиши фінальний звіт."}
        return None

    deep_research = LlmAgent(
        name="deep_research",
        description="Глибоке дослідження в інтернеті за детальним планом: агент сам шукає, читає джерела, "
                    "закриває прогалини і повертає звіт з джерелами. Для складних питань, порівнянь, оглядів ринку.",
        model=model,
        input_schema=ResearchPlan,
        instruction=research_instruction,
        tools=[search_web, fetch_url],
        before_tool_callback=check_budget,
        on_tool_error_callback=tool_error_to_result,
    )
    return AgentTool(deep_research)


# Main agent
def build_agent(model) -> LlmAgent:
    return LlmAgent(
        name="assistant",
        model=model,
        instruction=main_instruction,
        tools=[
            get_weather,
            get_exchange_rate,
            search_web,
            fetch_url,
            make_line_chart,
            create_report,
            edit_report,
            build_research_tool(model),
        ],
        on_tool_error_callback=tool_error_to_result,
        generate_content_config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )
