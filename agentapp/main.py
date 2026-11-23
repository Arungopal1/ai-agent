"""FastAPI application factory.

Local:   uvicorn agentapp.main:app --reload
Vercel:  main.py re-exports ``app`` for the Python runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path

# Load .env before anything reads the environment (local development only).
try:  # pragma: no cover - optional dependency path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # pragma: no cover
    pass

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from . import __version__  # noqa: E402
from .config import settings  # noqa: E402
from .routers import chat, documents, system  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("agentapp")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = PROJECT_ROOT / "public"

app = FastAPI(
    title="AI Agent API",
    version=__version__,
    description=(
        "A small agent service with tool calling, document search (RAG) and "
        "streaming answers. Deployable to Vercel as a single Python function."
    ),
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)

app.include_router(system.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error. Check the server logs.", "error": type(exc).__name__},
    )


@app.get("/api", include_in_schema=False)
async def api_root() -> dict[str, object]:
    return {
        "name": "AI Agent API",
        "version": __version__,
        "provider": settings.provider,
        "model": settings.model,
        "docs": "/api/docs",
        "health": "/api/health",
    }


# Serve the bundled UI locally. On Vercel, public/ is served straight from the CDN.
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
else:

    @app.get("/", include_in_schema=False)
    async def index() -> dict[str, str]:
        return {"detail": "UI not found. See /api/docs for the API."}
