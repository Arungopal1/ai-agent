"""Paragraph-aware text chunking."""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"[ \t]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Collapse noisy whitespace but keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries, then hard-wrap."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            for start in range(0, len(sentence), max_chars):
                pieces.append(sentence[start : start + max_chars])
            continue
        if len(current) + len(sentence) + 1 > max_chars and current:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Greedy paragraph packing with a small character overlap for context."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    units: list[str] = []
    for paragraph in cleaned.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units.extend(_split_long(paragraph, max_chars) if len(paragraph) > max_chars else [paragraph])

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + len(unit) + 2 <= max_chars:
            current = f"{current}\n\n{unit}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{unit}".strip() if tail else unit
    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk.strip()]
