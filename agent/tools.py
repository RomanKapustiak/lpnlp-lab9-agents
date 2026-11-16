"""Data tools: weather, NBU exchange rates, web search and reading web pages.

Every tool is a plain Python function. ADK turns its signature and docstring into the JSON schema
that the model sees, so the docstrings below are part of the prompt.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import time
from datetime import date, datetime
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from .config import HTTP_HEADERS, TZ

# Weather

# WMO weather codes used by Open-Meteo
WMO = {
    0: "ясно", 1: "переважно ясно", 2: "мінлива хмарність", 3: "хмарно", 45: "туман", 48: "паморозь, туман",
    51: "слабка мряка", 53: "мряка", 55: "сильна мряка", 56: "крижана мряка", 57: "сильна крижана мряка",
    61: "невеликий дощ", 63: "дощ", 65: "сильний дощ", 66: "крижаний дощ", 67: "сильний крижаний дощ",
    71: "невеликий сніг", 73: "сніг", 75: "сильний сніг", 77: "снігова крупа",
    80: "короткочасний дощ", 81: "зливи", 82: "сильні зливи", 85: "снігопад", 86: "сильний снігопад",
    95: "гроза", 96: "гроза з градом", 99: "сильна гроза з градом",
}


async def get_weather(city: str, days: int = 3) -> dict:
    """Returns current weather and a daily forecast for a city (source: Open-Meteo).

    Args:
        city: City name in Ukrainian or English, e.g. "Львів" or "Paris".
        days: Number of forecast days, 1 to 16 (including today).
    """
    days = max(1, min(int(days), 16))
    async with httpx.AsyncClient(timeout=20, headers=HTTP_HEADERS) as client:
        # 1) city name -> coordinates
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "uk", "format": "json"},
        )
        results = geo.json().get("results")
        if not results:
            return {"status": "error", "error": f"Місто '{city}' не знайдено. Спробуй іншу назву (напр. англійською)."}
        place = results[0]
        # 2) coordinates -> current weather + daily forecast
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "timezone": "auto",
                "forecast_days": days,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,"
                "precipitation_sum,precipitation_probability_max,wind_speed_10m_max,weather_code",
            },
        )
        data = resp.json()
    cur, daily = data["current"], data["daily"]
    return {
        "status": "ok",
        "location": ", ".join(x for x in (place.get("name"), place.get("admin1"), place.get("country")) if x),
        "current": {
            "time": cur["time"],
            "temperature_c": cur["temperature_2m"],
            "feels_like_c": cur["apparent_temperature"],
            "humidity_pct": cur["relative_humidity_2m"],
            "wind_kmh": cur["wind_speed_10m"],
            "conditions": WMO.get(cur["weather_code"], str(cur["weather_code"])),
        },
        "daily": [
            {
                "date": daily["time"][i],
                "t_min_c": daily["temperature_2m_min"][i],
                "t_max_c": daily["temperature_2m_max"][i],
                "feels_like_min_c": daily["apparent_temperature_min"][i],
                "feels_like_max_c": daily["apparent_temperature_max"][i],
                "precipitation_mm": daily["precipitation_sum"][i],
                "precipitation_probability_pct": daily["precipitation_probability_max"][i],
                "wind_max_kmh": daily["wind_speed_10m_max"][i],
                "conditions": WMO.get(daily["weather_code"][i], str(daily["weather_code"][i])),
            }
            for i in range(len(daily["time"]))
        ],
    }


# Exchange rates
async def get_exchange_rate(currency: str, start_date: str = "", end_date: str = "") -> dict:
    """Official hryvnia exchange rate from the National Bank of Ukraine: for one date or a period (up to 1 year).

    Args:
        currency: ISO 4217 currency code, e.g. "USD", "EUR", "PLN".
        start_date: Start date YYYY-MM-DD. Empty = today.
        end_date: End date YYYY-MM-DD. Empty = same as start_date.
    """
    try:
        start = date.fromisoformat(start_date) if start_date else datetime.now(TZ).date()
        end = date.fromisoformat(end_date) if end_date else start
    except ValueError:
        return {"status": "error", "error": "Дати мають бути у форматі YYYY-MM-DD."}
    start, end = min(start, end), max(start, end)
    if (end - start).days > 366:
        return {"status": "error", "error": "Максимальний період — 1 рік."}
    async with httpx.AsyncClient(timeout=20, headers=HTTP_HEADERS) as client:
        resp = await client.get(
            "https://bank.gov.ua/NBU_Exchange/exchange_site",
            params={"start": f"{start:%Y%m%d}", "end": f"{end:%Y%m%d}", "valcode": currency.lower(),
                    "sort": "exchangedate", "order": "asc", "json": ""},
        )
        rows = resp.json()
    if not rows:
        return {"status": "error", "error": f"Немає даних НБУ для {currency.upper()} за цей період."}
    points = [
        {"date": datetime.strptime(r["exchangedate"], "%d.%m.%Y").date().isoformat(), "rate": r["rate_per_unit"]}
        for r in rows
    ]
    rates = [p["rate"] for p in points]
    if len(points) > 60:  # keep ~60 evenly spaced points so the model can pass them to a chart
        step = len(points) / 60
        points = [points[int(i * step)] for i in range(60)] + [points[-1]]
    return {
        "status": "ok",
        "currency": rows[0]["cc"],
        "name": rows[0]["txt"],
        "unit": "UAH за 1 одиницю валюти",
        "summary": {
            "first": rates[0],
            "last": rates[-1],
            "min": min(rates),
            "max": max(rates),
            "change_pct": round((rates[-1] / rates[0] - 1) * 100, 2),
        },
        "points": points,
    }

# Web search
def _ddg_search(query: str, max_results: int, timelimit: str | None) -> list[dict]:
    """One DuckDuckGo query. DDG sometimes randomly answers "No results", so we retry a couple of times."""
    region = "ua-uk" if re.search("[а-яіїєґ]", query, re.IGNORECASE) else "wt-wt"
    for attempt in range(3):
        try:
            return DDGS(timeout=15).text(query, region=region, max_results=max_results, timelimit=timelimit)
        except DDGSException as e:
            if "no results" not in str(e).lower():
                raise
            time.sleep(0.8 * (attempt + 1))
    return []


async def search_web(queries: list[str], max_results: int = 5, time_range: str = "") -> dict:
    """Web search (DuckDuckGo). Takes a LIST of queries and runs them in parallel, so several phrasings
    should be sent in one call. Returns titles, links and short snippets; open the most relevant pages
    with fetch_url for exact numbers and details.

    Args:
        queries: 1-5 specific search queries (different phrasings; English is fine for international topics).
        max_results: Results per query, 1 to 8.
        time_range: Freshness filter: "d" day, "w" week, "m" month, "y" year; empty = no filter.
    """
    if isinstance(queries, str):
        queries = [queries]
    queries = [q.strip() for q in queries if q and q.strip()][:5]
    if not queries:
        return {"status": "error", "error": "Передай хоча б один запит у queries."}
    max_results = max(1, min(int(max_results), 8))
    timelimit = time_range if time_range in ("d", "w", "m", "y") else None
    # DDGS is synchronous -> run each query in a thread, all in parallel
    outcomes = await asyncio.gather(*(asyncio.to_thread(_ddg_search, q, max_results, timelimit) for q in queries),
                                    return_exceptions=True)
    seen: set[str] = set()  # the same URL is often found by several queries
    by_query, errors = [], []
    for query, rows in zip(queries, outcomes):
        if isinstance(rows, Exception):
            errors.append(f"«{query}»: {rows}")
            continue
        results = []
        for r in rows:
            url = r.get("href", "")
            if url and url not in seen:
                seen.add(url)
                results.append({"title": r.get("title", ""), "url": url, "snippet": r.get("body", "")})
        by_query.append({"query": query, "results": results})
    if not any(b["results"] for b in by_query):
        return {"status": "empty", "errors": errors,
                "hint": "Нічого не знайдено — переформулюй запити або прибери time_range."}
    return {"status": "ok", "results_by_query": by_query, "errors": errors}


# Reading a page
class _TextExtractor(HTMLParser):
    """Minimal HTML -> text: skips scripts/menus/footers, keeps paragraph breaks."""

    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form", "iframe"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0  # depth inside SKIP tags
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        lines = (re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in "".join(self.parts).splitlines())
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _is_public_host(host: str) -> bool:
    """SSRF protection: the agent must not read localhost or private-network addresses."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(info[4][0]).is_global for info in infos)


async def fetch_url(url: str, max_chars: int = 8000) -> dict:
    """Downloads a web page and returns its title and main text.

    Args:
        url: Full page address (http or https).
        max_chars: Maximum number of text characters in the result (up to 20000).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return {"status": "error", "error": "Підтримуються лише http/https URL."}
    if not await asyncio.to_thread(_is_public_host, parsed.hostname):
        return {"status": "error", "error": "Хост недоступний або не є публічним."}
    try:
        async with httpx.AsyncClient(timeout=25, headers=HTTP_HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
    except httpx.HTTPError as e:
        return {"status": "error", "error": f"Не вдалося завантажити сторінку: {e}"}
    if resp.status_code >= 400:
        return {"status": "error", "error": f"HTTP {resp.status_code}"}
    ctype = resp.headers.get("content-type", "")
    if "html" in ctype:
        parser = _TextExtractor()
        parser.feed(resp.text)
        title, text = parser.title.strip(), parser.text()
    elif ctype.startswith("text/") or "json" in ctype:
        title, text = "", resp.text
    else:
        return {"status": "error", "error": f"Непідтримуваний тип вмісту: {ctype}"}
    max_chars = max(500, min(int(max_chars), 20000))
    return {"status": "ok", "url": str(resp.url), "title": title, "text": text[:max_chars],
            "truncated": len(text) > max_chars}
