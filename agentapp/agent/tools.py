"""Tool registry exposed to the language model."""

from __future__ import annotations

import ast
import asyncio
import html
import ipaddress
import json
import math
import operator
import re
from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from ..config import settings
from ..rag.service import rag

MAX_TOOL_OUTPUT = 6000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------
_BIN_OPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[float], float]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "exp": math.exp,
    "floor": math.floor,
    "log": math.log,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ValueError("only numbers are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _FUNCTIONS.get(node.func.id)
        if func is None or node.keywords:
            raise ValueError(f"unsupported function: {node.func.id}")
        return func(*[_eval_node(arg) for arg in node.args])
    raise ValueError("unsupported expression")


async def _calculator(args: dict[str, Any]) -> str:
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return "Error: no expression provided."
    if len(expression) > 200:
        return "Error: expression too long."
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


# ---------------------------------------------------------------------------
# current_time
# ---------------------------------------------------------------------------
async def _current_time(args: dict[str, Any]) -> str:
    name = str(args.get("timezone") or "UTC").strip()
    if name.upper() == "UTC":
        tz = dt_timezone.utc
    else:
        try:
            tz = ZoneInfo(name)
        except ZoneInfoNotFoundError:
            return f"Error: unknown timezone '{name}'. Use an IANA name like 'Asia/Kolkata' or 'UTC'."
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)") + f" | ISO: {now.isoformat(timespec='seconds')}"


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------
async def _tavily_search(query: str, max_results: int) -> list[dict[str, str]]:
    async with httpx.AsyncClient(timeout=20) as http:
        response = await http.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            headers={"Authorization": f"Bearer {settings.tavily_key}"},
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    return [
        {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": (item.get("content") or "")[:400]}
        for item in results
    ]


def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore[import-not-found]
    with DDGS() as search:
        rows = list(search.text(query, max_results=max_results))
    return [
        {
            "title": row.get("title", ""),
            "url": row.get("href", "") or row.get("url", ""),
            "snippet": (row.get("body", "") or "")[:400],
        }
        for row in rows
    ]


async def _web_search(args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "Error: no query provided."
    limit = max(1, min(int(args.get("max_results") or 5), 8))
    try:
        if settings.tavily_key:
            results = await _tavily_search(query, limit)
            engine = "tavily"
        else:
            results = await asyncio.to_thread(_ddg_search, query, limit)
            engine = "duckduckgo"
    except Exception as exc:  # noqa: BLE001
        return (
            f"Web search failed ({type(exc).__name__}). "
            "Set TAVILY_API_KEY for a reliable provider, or answer from other context."
        )
    if not results:
        return "No web results found."
    lines = [f"Web results ({engine}) for: {query}"]
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item['title']}\n   {item['url']}\n   {item['snippet']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------
def _is_public_url(url: str) -> bool:
    """Cheap SSRF guard: refuse obvious private hosts. Production should also
    resolve DNS and pin the IP."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True  # a hostname; DNS-level checks belong in a proxy
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
    )


_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_RE = re.compile(r"</(p|div|li|h[1-6]|tr|br)>", re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]+>")


def _html_to_text(raw: str) -> str:
    raw = _TAG_RE.sub(" ", raw)
    raw = _BLOCK_RE.sub("\n", raw)
    raw = _ANY_TAG_RE.sub(" ", raw)
    text = html.unescape(raw)
    text = re.sub(r"[ \t\x0b\f]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


async def _fetch_url(args: dict[str, Any]) -> str:
    url = str(args.get("url", "")).strip()
    if not _is_public_url(url):
        return "Error: only public http(s) URLs can be fetched."
    try:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": "ai-agent/1.0 (+https://vercel.com)"},
        ) as http:
            response = await http.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            body = response.text
    except Exception as exc:  # noqa: BLE001
        return f"Fetch failed: {type(exc).__name__}: {exc}"
    text = _html_to_text(body) if "html" in content_type.lower() else body
    text = text.strip()
    if not text:
        return "The page had no readable text."
    return f"Content from {url}:\n\n{text[:MAX_TOOL_OUTPUT]}"


# ---------------------------------------------------------------------------
# search_documents (RAG)
# ---------------------------------------------------------------------------
async def _search_documents(args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "Error: no query provided."
    k = max(1, min(int(args.get("k") or 4), 8))
    hits = await rag.search(query, k=k)
    if not hits:
        return (
            "The knowledge base is empty or has no match. "
            "Tell the user to upload documents, or answer from general knowledge."
        )
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{index}] source: {hit['source']} (score {hit['score']})\n{hit['text'][:1200]}"
        )
    return "Knowledge base excerpts:\n\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            name="search_documents",
            description=(
                "Search the user's uploaded documents and the bundled knowledge base. "
                "Use this for any question that might be answered by internal or "
                "project documentation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "k": {"type": "integer", "description": "Number of excerpts (1-8).", "default": 4},
                },
                "required": ["query"],
            },
            handler=_search_documents,
        ),
        Tool(
            name="web_search",
            description="Search the public web for recent or factual information.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "description": "1-8 results.", "default": 5},
                },
                "required": ["query"],
            },
            handler=_web_search,
        ),
        Tool(
            name="fetch_url",
            description="Download a public web page and return its readable text.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."}},
                "required": ["url"],
            },
            handler=_fetch_url,
        ),
        Tool(
            name="calculator",
            description="Evaluate a math expression, e.g. '(1200 * 1.18) / 3'.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression."}},
                "required": ["expression"],
            },
            handler=_calculator,
        ),
        Tool(
            name="current_time",
            description="Current date and time in a timezone (IANA name, default UTC).",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "e.g. Asia/Kolkata", "default": "UTC"}
                },
            },
            handler=_current_time,
        ),
    ]
}


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI function-calling schema for every registered tool."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in TOOLS.values()
    ]


async def run_tool(name: str, raw_arguments: str) -> tuple[str, str]:
    """Execute a tool by name. Returns ``(display_input, output)``."""
    tool = TOOLS.get(name)
    if tool is None:
        return (name, f"Error: unknown tool '{name}'.")
    try:
        args = json.loads(raw_arguments) if raw_arguments.strip() else {}
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except Exception as exc:  # noqa: BLE001
        return (raw_arguments, f"Error: could not parse tool arguments ({exc}).")

    display = ", ".join(f"{key}={value}" for key, value in args.items() if key != "k") or name
    try:
        output = await asyncio.wait_for(tool.handler(args), timeout=30)
    except asyncio.TimeoutError:
        return (display, "Error: tool timed out after 30s.")
    except Exception as exc:  # noqa: BLE001
        return (display, f"Error: {type(exc).__name__}: {exc}")
    return (display, str(output)[:MAX_TOOL_OUTPUT])
