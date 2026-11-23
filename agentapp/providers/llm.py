"""Chat-model access through one OpenAI-compatible client.

Groq, OpenRouter, Gemini, OpenAI and Ollama all speak the OpenAI wire format,
so a single client + ``base_url`` covers every provider in ``config.PROVIDERS``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from weakref import WeakKeyDictionary

from openai import AsyncOpenAI

from ..config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the upstream provider fails."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None


# One client per event loop: httpx connections are bound to the loop that
# created them, so a cached client breaks if the loop is replaced (tests,
# reloads, some serverless runtimes).
_clients: WeakKeyDictionary[Any, AsyncOpenAI] = WeakKeyDictionary()


def client() -> AsyncOpenAI:
    loop = asyncio.get_running_loop()
    existing = _clients.get(loop)
    if existing is None:
        existing = AsyncOpenAI(
            api_key=settings.api_key or "not-needed",
            base_url=settings.base_url,
            timeout=settings.request_timeout,
            max_retries=1,  # free tiers return 429 quickly; let the UI retry
        )
        _clients[loop] = existing
    return existing


def _usage_of(resp: Any) -> dict[str, int] | None:
    usage = getattr(resp, "usage", None)
    if not usage:
        return None
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _base_kwargs(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": settings.model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    # Some providers reject a temperature of exactly 0, so it is opt-in only.
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    return kwargs


async def complete(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    on_text: Callable[[str], Awaitable[None]] | None = None,
) -> LLMReply:
    """Run one completion.

    When ``on_text`` is provided the response is streamed and every text delta
    is forwarded to the callback while tool-call fragments are accumulated.
    """
    kwargs = _base_kwargs(messages, tools)
    try:
        if on_text is None:
            resp = await client().chat.completions.create(**kwargs)
            choice = resp.choices[0]
            message = choice.message
            calls = [
                ToolCall(
                    id=call.id or f"call_{i}",
                    name=call.function.name if call.function else "",
                    arguments=call.function.arguments if call.function and call.function.arguments else "{}",
                )
                for i, call in enumerate(message.tool_calls or [])
            ]
            return LLMReply(
                content=message.content or "",
                tool_calls=calls,
                finish_reason=choice.finish_reason or "stop",
                usage=_usage_of(resp),
            )

        stream = await client().chat.completions.create(stream=True, **kwargs)
        text_parts: list[str] = []
        pending: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if delta is not None and delta.content:
                text_parts.append(delta.content)
                await on_text(delta.content)
            for frag in (getattr(delta, "tool_calls", None) or []) if delta else []:
                slot = pending.setdefault(
                    frag.index or 0, {"id": "", "name": "", "arguments": ""}
                )
                if frag.id:
                    slot["id"] = frag.id
                if frag.function is not None:
                    if frag.function.name:
                        slot["name"] = frag.function.name
                    if frag.function.arguments:
                        slot["arguments"] += frag.function.arguments

        calls = [
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=slot["arguments"] or "{}",
            )
            for index, slot in sorted(pending.items())
            if slot["name"]
        ]
        return LLMReply(
            content="".join(text_parts),
            tool_calls=calls,
            finish_reason=finish_reason,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the UI
        log.warning("LLM call failed (%s/%s): %s", settings.provider, settings.model, exc)
        raise LLMError(_friendly_error(exc)) from exc


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "429" in text or "rate limit" in text.lower():
        return (
            "The model provider rate-limited us (429). Free tiers allow roughly "
            "30 requests/minute - wait a few seconds and try again."
        )
    if "401" in text or "invalid api key" in text.lower() or "unauthorized" in text.lower():
        return "The provider rejected the API key. Check the key in your environment variables."
    if "404" in text or "model_not_found" in text or "does not exist" in text:
        return (
            f"Model '{settings.model}' is not available for {settings.provider_label}. "
            "Set LLM_MODEL to a model your account can access."
        )
    if "timeout" in text.lower():
        return "The model provider timed out. Try a shorter question or retry."
    return f"Model provider error: {text[:300]}"
