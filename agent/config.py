from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
from google.adk.agents.run_config import RunConfig

APP_NAME = "universal_agent"
USER_ID = "user"
TZ = ZoneInfo("Europe/Kyiv")
MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-3.1-flash-lite"]

try:  # local development: the key is read from .env next to app.py
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass
ENV_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

RUN_CONFIG = RunConfig(max_llm_calls=60)  # hard cap on LLM calls per user turn
OUT_DIR = Path(tempfile.gettempdir()) / "universal_agent"  # generated PDFs
OUT_DIR.mkdir(parents=True, exist_ok=True)
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # DejaVu fonts (with Cyrillic) ship with matplotlib
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UniversalAgent/1.0)"}
WEEKDAYS = ["понеділок", "вівторок", "середа", "четвер", "п'ятниця", "субота", "неділя"]

CHART_RE = re.compile(r"\[\[chart:([\w-]+)\]\]")  # chart placeholder inside report markdown: [[chart:chart_1]]
