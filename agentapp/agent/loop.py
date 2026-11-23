"""The agent loop: plan -> call tools -> answer.

Two entry points:

* :func:`run` - one-shot, returns the final answer plus a tool trace.
* :func:`stream` - same loop, yielding events (``step`` / ``token`` / ``done``)
  so the browser can render progress and streamed text.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from ..config import settings
from ..providers import llm
from .prompts import DEMO_NOTICE, SYSTEM_PROMPT
from .tools import run_tool, tool_schemas

log = logging.getLogger(__name__)

Event = dict[str, Any]
EventHandler = Callable[[Event], Awaitable[None]]


@dataclass
class AgentResult:
    answer: str
    steps: list[dict[str, str]] = field(default_factory=list)
    usage: dict[str, int] | None = None
    mode: str = "llm"

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": self.steps,
            "provider": settings.provider,
            "model": settings.model,
            "mode": self.mode,
            "usage": self.usage,
        }


async def _emit(handler: EventHandler | None, event: Event) -> None:
    if handler is not None:
        await handler(event)


def _prepare(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed = messages[-settings.max_history_messages :]
    return [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": message["role"], "content": message["content"]} for message in trimmed
    ]


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


async def _demo_answer(messages: list[dict[str, Any]], on_event: EventHandler | None) -> AgentResult:
    """No API key: still demonstrate retrieval instead of failing."""
    from ..rag.service import rag

    question = _last_user_message(messages)
    steps: list[dict[str, str]] = []
    hits = await rag.search(question, k=3) if question else []
    if question:
        await _emit(
            on_event,
            {"type": "step", "tool": "search_documents", "input": question, "output": f"{len(hits)} excerpts"},
        )
        steps.append({"tool": "search_documents", "input": question, "output": f"{len(hits)} excerpts"})

    notice = DEMO_NOTICE.format(keys=settings.keys_url or "https://console.groq.com/keys")
    if hits:
        body = "\n\n".join(
            f"**[{index}] {hit['source']}** (score {hit['score']})\n\n> {hit['text'][:600]}"
            for index, hit in enumerate(hits, start=1)
        )
        answer = f"{notice}\n\nHere is the closest material I found for *{question}*:\n\n{body}"
    elif question:
        answer = f"{notice}\n\nYour knowledge base has no match for *{question}* yet. Upload a document in the sidebar and try again."
    else:
        answer = notice

    return AgentResult(answer=answer, steps=steps, mode="demo")


async def run(
    messages: list[dict[str, Any]],
    *,
    max_steps: int | None = None,
    on_event: EventHandler | None = None,
    stream_tokens: bool = False,
) -> AgentResult:
    """Execute the agent loop and return the final answer."""
    if not settings.configured:
        return await _demo_answer(messages, on_event)

    step_limit = min(max_steps or settings.max_steps, 10)
    convo = _prepare(messages)
    steps: list[dict[str, str]] = []
    usage: dict[str, int] | None = None

    try:
        for _ in range(step_limit):
            streamed = False
            on_text: Callable[[str], Awaitable[None]] | None = None
            if stream_tokens:

                async def on_text(delta: str) -> None:  # noqa: F811 - closure per iteration
                    nonlocal streamed
                    streamed = True
                    await _emit(on_event, {"type": "token", "text": delta})

            reply = await llm.complete(convo, tools=tool_schemas(), on_text=on_text)
            usage = reply.usage or usage

            if not reply.tool_calls:
                return AgentResult(
                    answer=reply.content.strip() or "(the model returned an empty message)",
                    steps=steps,
                    usage=usage,
                )

            if streamed:  # partial prose that is not the real answer
                await _emit(on_event, {"type": "reset"})

            convo.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                display, output = await run_tool(call.name, call.arguments)
                convo.append({"role": "tool", "tool_call_id": call.id, "content": output})
                steps.append({"tool": call.name, "input": display, "output": output[:800]})
                await _emit(
                    on_event,
                    {"type": "step", "tool": call.name, "input": display, "output": output[:400]},
                )

        # Step budget exhausted - force a final answer with no more tools.
        final_conv = convo + [
            {
                "role": "user",
                "content": "Stop using tools and answer now with what you already have.",
            }
        ]
        on_text_final = None
        if stream_tokens:

            async def on_text_final(delta: str) -> None:
                await _emit(on_event, {"type": "token", "text": delta})

        reply = await llm.complete(final_conv, tools=None, on_text=on_text_final)
        usage = reply.usage or usage
        return AgentResult(
            answer=reply.content.strip() or "(the model returned an empty message)",
            steps=steps,
            usage=usage,
        )
    except llm.LLMError as exc:
        log.warning("Agent run failed: %s", exc)
        return AgentResult(answer=str(exc), steps=steps, usage=usage, mode="error")


async def stream(messages: list[dict[str, Any]], *, max_steps: int | None = None) -> AsyncIterator[Event]:
    """Yield events from the loop, ending with ``done`` or ``error``."""
    queue: asyncio.Queue[Event | None] = asyncio.Queue()

    async def on_event(event: Event) -> None:
        await queue.put(event)

    async def worker() -> None:
        try:
            result = await run(
                messages, max_steps=max_steps, on_event=on_event, stream_tokens=True
            )
            await queue.put({"type": "done", **result.as_dict()})
        except Exception as exc:  # noqa: BLE001 - always close the stream cleanly
            log.exception("Streaming agent crashed")
            await queue.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            await queue.put(None)

    task = asyncio.create_task(worker())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        if not task.done():
            task.cancel()
