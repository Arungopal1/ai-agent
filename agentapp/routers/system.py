"""Health, config and tool introspection. No auth required."""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..agent.tools import TOOLS
from ..config import settings
from ..rag.service import rag
from ..schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness + capability probe")
async def health() -> HealthResponse:
    await rag.ensure_seeded()
    return HealthResponse(
        status="ok",
        provider=settings.provider,
        model=settings.model,
        configured=settings.configured,
        documents=len(rag.sources()),
        chunks=rag.total_chunks,
        version=__version__,
    )


@router.get("/config", summary="Public client configuration")
async def config() -> dict[str, object]:
    return {
        **settings.public_dict(),
        "tools": [
            {"name": tool.name, "description": tool.description} for tool in TOOLS.values()
        ],
    }
