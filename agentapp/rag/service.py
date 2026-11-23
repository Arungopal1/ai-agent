"""High-level RAG service: seeding, ingestion, search, deletion."""

from __future__ import annotations

import asyncio
import io
import logging
import re
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import settings
from ..providers.embeddings import embed, embed_one
from .chunker import chunk_text, clean_text
from .store import VectorStore

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = PROJECT_ROOT / "data" / "docs"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "ai_agent_uploads"

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".csv", ".json", ".py", ".html"}
PDF_SUFFIXES = {".pdf"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    candidate = _SAFE_NAME.sub("_", Path(name).name).strip("._") or "document"
    return candidate[:120]


def extract_text(filename: str, data: bytes) -> str:
    """Turn raw bytes into plain text (PDF via pypdf, everything else as text)."""
    suffix = Path(filename).suffix.lower()
    if suffix in PDF_SUFFIXES:
        from pypdf import PdfReader  # imported lazily to keep cold starts cheap

        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages)
    return data.decode("utf-8", errors="ignore")


class RagService:
    def __init__(self) -> None:
        self.store = VectorStore()
        self._seed_lock = Lock()
        self._seeded = False

    # -- seeding --------------------------------------------------------
    async def ensure_seeded(self) -> None:
        """Load bundled docs (and surviving /tmp uploads) exactly once."""
        if self._seeded:
            return
        with self._seed_lock:
            if self._seeded:
                return
            self._seeded = True  # set first: a failed seed must not retry forever

        seeded_files = 0
        for directory in (SEED_DIR, UPLOAD_DIR):
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix not in TEXT_SUFFIXES | PDF_SUFFIXES:
                    continue
                try:
                    data = path.read_bytes()
                    await self._ingest(safe_filename(path.name), data, metadata={"origin": "seed"})
                    seeded_files += 1
                except Exception as exc:  # noqa: BLE001 - one bad file must not break startup
                    log.warning("Could not seed %s: %s", path.name, exc)
        if seeded_files:
            log.info("Seeded %s document(s) into the knowledge base", seeded_files)

    # -- ingestion ------------------------------------------------------
    async def _ingest(self, source: str, data: bytes, metadata: dict[str, Any] | None = None) -> int:
        text = clean_text(extract_text(source, data))
        if not text:
            raise ValueError("No readable text found in that file.")
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No readable text found in that file.")
        vectors = await embed(chunks, task="RETRIEVAL_DOCUMENT")
        self.store.delete_source(source)  # re-upload replaces the old version
        return self.store.add(source, chunks, vectors, metadata=metadata)

    async def add_text(self, text: str, source: str) -> int:
        await self.ensure_seeded()
        source = safe_filename(source)
        return await self._ingest(source, text.encode("utf-8"), metadata={"origin": "text"})

    async def add_file(self, filename: str, data: bytes) -> int:
        await self.ensure_seeded()
        source = safe_filename(filename)
        chunks = await self._ingest(source, data, metadata={"origin": "upload", "bytes": len(data)})
        try:  # best effort: survive warm instance restarts, not cold starts
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            (UPLOAD_DIR / source).write_bytes(data)
        except Exception as exc:  # noqa: BLE001
            log.info("Could not cache upload on disk: %s", exc)
        return chunks

    # -- queries --------------------------------------------------------
    async def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        await self.ensure_seeded()
        vector = await embed_one(query, task="RETRIEVAL_QUERY")
        hits = []
        for chunk, score in self.store.search(vector, k=k):
            hits.append(
                {
                    "source": chunk.source,
                    "text": chunk.text,
                    "score": round(float(score), 4),
                    "metadata": {**chunk.metadata, "chunk": chunk.index},
                }
            )
        return hits

    def sources(self) -> dict[str, int]:
        return self.store.sources()

    def remove(self, source: str) -> int:
        removed = self.store.delete_source(safe_filename(source))
        try:
            path = UPLOAD_DIR / safe_filename(source)
            if path.is_file():
                path.unlink()
        except Exception:  # noqa: BLE001
            pass
        return removed

    @property
    def total_chunks(self) -> int:
        return self.store.size


rag = RagService()
