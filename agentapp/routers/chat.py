"""Chat endpoints: JSON one-shot and SSE streaming."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..agent import loop
from ..schemas import ChatRequest, ChatResponse
from ..security import guard

router = APIRouter(prefix="/api", tags=["chat"], dependencies=[Depends(guard)])


def _messages(request: ChatRequest) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in request.messages]


@router.post("/chat", response_model=ChatResponse, summary="Ask the agent (one shot)")
async def chat(request: ChatRequest) -> ChatResponse:
    result = await loop.run(_messages(request), max_steps=request.max_steps)
    return ChatResponse(**result.as_dict())


@router.post("/chat/stream", summary="Ask the agent with server-sent events")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    messages = _messages(request)

    async def events() -> AsyncIterator[str]:
        try:
            async for event in loop.stream(messages, max_steps=request.max_steps):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - the browser needs a clean close
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
