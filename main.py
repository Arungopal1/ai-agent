"""Vercel entrypoint.

Vercel's Python runtime looks for a FastAPI instance named ``app`` in
``main.py`` / ``index.py`` / ``server.py`` / ``app.py`` at the project root
(or inside ``src/`` or ``app/``). We keep the real application inside the
``agentapp`` package so ``uvicorn agentapp.main:app`` works locally too.
"""

from agentapp.main import app

__all__ = ["app"]
