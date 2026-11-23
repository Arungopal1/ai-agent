"""End-to-end check against a fake OpenAI-compatible provider.

Runs the *real* agent loop (tool calling + streaming) without spending API
quota and without any key:

    python scripts/fake_provider_check.py

It starts a tiny fake provider on 127.0.0.1:8099, points the app at it, then
exercises /api/chat and /api/chat/stream in-process.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


FAKE_PORT = int(os.environ.get("FAKE_PROVIDER_PORT") or _free_port())
os.environ["LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "fake-key"
os.environ["LLM_BASE_URL"] = f"http://127.0.0.1:{FAKE_PORT}/v1"
os.environ["LLM_MODEL"] = "fake-model"
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["RATE_LIMIT_PER_MIN"] = "0"

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

fake = FastAPI()
FINAL_TEXT = "12 times 12 is 144."


def _tool_call() -> dict:
    return {
        "id": "call_fake_1",
        "type": "function",
        "function": {"name": "calculator", "arguments": json.dumps({"expression": "12*12"})},
    }


def _envelope(message: dict, finish: str) -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "fake-model",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _chunk(delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "fake-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@fake.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    used_tool = any(message.get("role") == "tool" for message in messages)
    wants_stream = bool(body.get("stream"))

    if not used_tool:
        call = _tool_call()
        if not wants_stream:
            return JSONResponse(_envelope({"content": None, "tool_calls": [call]}, "tool_calls"))

        def tool_stream():
            yield _chunk({"role": "assistant", "content": None})
            yield _chunk({"tool_calls": [{"index": 0, "id": call["id"], "type": "function",
                                          "function": {"name": "calculator", "arguments": ""}}]})
            yield _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"expression": "12*12"}'}}]})
            yield _chunk({}, "tool_calls")
            yield "data: [DONE]\n\n"

        return StreamingResponse(tool_stream(), media_type="text/event-stream")

    if not wants_stream:
        return JSONResponse(_envelope({"content": FINAL_TEXT}, "stop"))

    def text_stream():
        for word in FINAL_TEXT.split(" "):
            yield _chunk({"content": word + " "})
        yield _chunk({}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(text_stream(), media_type="text/event-stream")


def start_fake_provider() -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(fake, host="127.0.0.1", port=FAKE_PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise SystemExit("fake provider did not start")
    return server


def main() -> int:
    server = start_fake_provider()
    failures: list[str] = []
    try:
        from fastapi.testclient import TestClient

        from agentapp.main import app

        client = TestClient(app)
        question = {"messages": [{"role": "user", "content": "What is 12 times 12?"}]}

        # 1. non-streaming JSON path
        response = client.post("/api/chat", json=question)
        body = response.json()
        print("POST /api/chat ->", response.status_code, json.dumps(body, indent=2)[:600])
        if response.status_code != 200:
            failures.append(f"/api/chat status {response.status_code}")
        if body.get("mode") != "llm":
            failures.append(f"/api/chat mode was {body.get('mode')!r}, expected 'llm'")
        if body.get("answer") != FINAL_TEXT:
            failures.append(f"/api/chat answer was {body.get('answer')!r}")
        if [step["tool"] for step in body.get("steps", [])] != ["calculator"]:
            failures.append("calculator tool step was not recorded")
        if body.get("usage", {}).get("total_tokens") != 15:
            failures.append("usage was not passed through")

        # 2. streaming SSE path
        tokens: list[str] = []
        events: list[dict] = []
        with client.stream("POST", "/api/chat/stream", json=question) as stream:
            for line in stream.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                events.append(event)
                if event["type"] == "token":
                    tokens.append(event["text"])
        print("stream event types ->", [event["type"] for event in events])
        print("streamed text       ->", repr("".join(tokens)))
        if "".join(tokens).strip() != FINAL_TEXT:
            failures.append("streamed tokens did not match the final answer")
        if not any(event["type"] == "step" and event["tool"] == "calculator" for event in events):
            failures.append("stream did not report the calculator tool step")
        done = next((event for event in events if event["type"] == "done"), None)
        if not done or done.get("answer") != FINAL_TEXT:
            failures.append("stream is missing a correct 'done' event")

        # 3. tool execution itself
        from agentapp.agent.tools import run_tool

        async def check_tools() -> None:
            _, output = await run_tool("calculator", '{"expression": "(1200*1.18)/3"}')
            print("calculator tool     ->", output)
            if "472" not in output:
                failures.append(f"calculator returned {output!r}")
            _, blocked = await run_tool("fetch_url", '{"url": "http://127.0.0.1:8099/"}')
            print("ssrf guard          ->", blocked)
            if "only public" not in blocked:
                failures.append("fetch_url did not block a loopback address")

        asyncio.run(check_tools())
    finally:
        server.should_exit = True

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nAll integration checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
