"""Smoke tests - run with ``pytest -q``.

The environment is pinned to demo mode before the app is imported, so these
tests pass without any API keys and without calling a paid provider.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["LLM_PROVIDER"] = "mock"
for _key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(_key, None)
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["RATE_LIMIT_PER_MIN"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from agentapp.main import app  # noqa: E402

client = TestClient(app)


def test_health_seeds_knowledge_base():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["configured"] is False
    assert body["documents"] >= 1  # bundled data/docs files
    assert body["chunks"] >= 1


def test_config_exposes_tools():
    response = client.get("/api/config")
    assert response.status_code == 200
    body = response.json()
    names = {tool["name"] for tool in body["tools"]}
    assert {"search_documents", "web_search", "calculator", "current_time"} <= names
    assert body["configured"] is False


def test_add_text_search_and_delete():
    added = client.post(
        "/api/documents/text",
        json={
            "source": "smoke-test.md",
            "text": "The launch code for the Orion project is BLUE-HARBOR-42. " * 12,
        },
    )
    assert added.status_code == 201, added.text
    assert added.json()["chunks"] >= 1

    found = client.post("/api/documents/search", json={"query": "Orion launch code", "k": 3})
    assert found.status_code == 200
    hits = found.json()["hits"]
    assert hits and hits[0]["source"] == "smoke-test.md"

    listed = client.get("/api/documents").json()
    assert any(doc["source"] == "smoke-test.md" for doc in listed["documents"])

    deleted = client.delete("/api/documents", params={"source": "smoke-test.md"})
    assert deleted.status_code == 200
    assert deleted.json()["chunks"] >= 1


def test_upload_rejects_unsupported_type():
    response = client.post(
        "/api/documents",
        files={"file": ("virus.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_chat_demo_mode_answers_from_documents():
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "How do I deploy this app?"}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "demo"
    assert "Demo mode" in body["answer"]
    assert body["provider"] == "mock"


def test_chat_stream_emits_sse_events():
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "What is this app?"}]},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payload = "".join(response.iter_text())
    assert "data:" in payload
    assert '"type": "step"' in payload or '"type": "token"' in payload
    assert "[DONE]" in payload


def test_validation_rejects_empty_message():
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": ""}]})
    assert response.status_code == 422
