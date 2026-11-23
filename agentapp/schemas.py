"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant", "system"]


class ChatMessage(BaseModel):
    role: Role = "user"
    content: str = Field(min_length=1, max_length=32000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=60)
    max_steps: int | None = Field(default=None, ge=1, le=10)


class AgentStep(BaseModel):
    tool: str
    input: str
    output: str


class ChatResponse(BaseModel):
    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    provider: str
    model: str
    mode: str = "llm"
    usage: dict[str, int] | None = None


class TextDocumentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=400_000)
    source: str = Field(default="pasted-text", max_length=200)


class DocumentInfo(BaseModel):
    source: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int
    embedding_backend: str


class IngestResponse(BaseModel):
    source: str
    chunks: int
    total_chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=4, ge=1, le=20)


class SearchHit(BaseModel):
    source: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    configured: bool
    documents: int
    chunks: int
    version: str
