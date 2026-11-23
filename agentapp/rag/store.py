"""In-memory vector store.

A few thousand chunks of pure-Python cosine similarity is fast enough for a
demo-scale knowledge base and keeps cold starts tiny (no numpy, no external
vector database). Swap ``VectorStore`` for pgvector / Upstash / Qdrant when you
need persistence across cold starts or a much larger corpus.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._dim = 0

    # -- writes ---------------------------------------------------------
    def add(self, source: str, texts: list[str], vectors: list[list[float]], metadata: dict[str, Any] | None = None) -> int:
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must have the same length")
        if not texts:
            return 0
        dim = len(vectors[0])
        with self._lock:
            if self._dim and dim != self._dim:
                # Different embedding model: the old vectors are not comparable.
                self._chunks.clear()
                self._vectors.clear()
            self._dim = dim
            start = sum(1 for chunk in self._chunks if chunk.source == source)
            for offset, (text, vector) in enumerate(zip(texts, vectors)):
                self._chunks.append(
                    Chunk(
                        id=uuid.uuid4().hex,
                        source=source,
                        text=text,
                        index=start + offset,
                        metadata=dict(metadata or {}),
                    )
                )
                self._vectors.append(vector)
            return len(texts)

    def delete_source(self, source: str) -> int:
        with self._lock:
            keep = [i for i, chunk in enumerate(self._chunks) if chunk.source != source]
            removed = len(self._chunks) - len(keep)
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = [self._vectors[i] for i in keep]
            return removed

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._vectors.clear()
            self._dim = 0

    # -- reads ----------------------------------------------------------
    def search(self, vector: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        with self._lock:
            if not self._chunks or len(vector) != self._dim:
                return []
            scored = [
                (chunk, sum(a * b for a, b in zip(vector, stored)))
                for chunk, stored in zip(self._chunks, self._vectors)
            ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def sources(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for chunk in self._chunks:
                counts[chunk.source] = counts.get(chunk.source, 0) + 1
            return counts

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._chunks)
