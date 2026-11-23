"""Embeddings for document search.

Two backends:

* ``gemini`` - Google ``gemini-embedding-001`` via REST (free tier).
* ``hashing`` - a deterministic local bag-of-words vectoriser used when no
  embedding key is configured. Good enough to demo retrieval, not for
  production semantic search.

Both return L2-normalised float lists, so cosine similarity is a plain dot
product.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Literal

import httpx

from ..config import settings

log = logging.getLogger(__name__)

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]

_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
_HASH_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Without an embedding model, common words would dominate the score.
_STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into
    is it its itself just me more most my no nor not of off on once only or other our
    ours out over own same she should so some such than that the their theirs them then
    there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours""".split()
)


def _stem(token: str) -> str:
    """Crude plural stripping so 'deploy' matches 'deploys'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _hash_embed(text: str) -> list[float]:
    """Signed feature hashing - deterministic, dependency free."""
    vector = [0.0] * _HASH_DIM
    tokens = [
        _stem(token)
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % _HASH_DIM
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign
    return _normalise(vector)


async def _gemini_embed(texts: list[str], task: TaskType) -> list[list[float]]:
    vectors: list[list[float]] = []
    headers = {"x-goog-api-key": settings.embedding_key, "content-type": "application/json"}
    url = f"{_GEMINI_ENDPOINT}/{settings.embedding_model}:batchEmbedContents"
    async with httpx.AsyncClient(timeout=settings.request_timeout) as http:
        for start in range(0, len(texts), 100):
            batch = texts[start : start + 100]
            payload = {
                "requests": [
                    {
                        "model": f"models/{settings.embedding_model}",
                        "content": {"parts": [{"text": item}]},
                        "taskType": task,
                    }
                    for item in batch
                ]
            }
            response = await http.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            for item in data.get("embeddings", []):
                vectors.append(_normalise([float(v) for v in item.get("values", [])]))
    if len(vectors) != len(texts):
        raise RuntimeError("embedding provider returned an unexpected number of vectors")
    return vectors


async def embed(texts: list[str], task: TaskType = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a list of texts, falling back to local hashing on any failure."""
    if not texts:
        return []
    if settings.embedding_backend == "gemini":
        try:
            return await _gemini_embed(texts, task)
        except Exception as exc:  # noqa: BLE001 - never break ingestion
            log.warning("Gemini embeddings failed, using local hashing instead: %s", exc)
    return [_hash_embed(text) for text in texts]


async def embed_one(text: str, task: TaskType = "RETRIEVAL_QUERY") -> list[float]:
    vectors = await embed([text], task=task)
    return vectors[0] if vectors else []
