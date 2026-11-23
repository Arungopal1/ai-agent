"""Request guards: optional API key + best-effort per-IP rate limiting.

Both are deliberately simple. On Vercel every warm instance keeps its own
counters, so the rate limit is a speed bump, not a hard quota. Put a real
gateway (or Vercel Firewall) in front for public production traffic.
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Header, HTTPException, Request, status

from .config import settings

_WINDOW_SECONDS = 60.0
_hits: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for") or request.headers.get("x-real-ip")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> None:
    limit = settings.rate_limit_per_min
    if limit <= 0:
        return
    now = time.monotonic()
    with _lock:
        bucket = _hits[ip]
        while bucket and now - bucket[0] > _WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(_WINDOW_SECONDS - (now - bucket[0])) + 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit reached ({limit}/min). Retry in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)
        if len(_hits) > 2048:  # keep memory bounded on long-lived instances
            for stale in [k for k, v in _hits.items() if not v][:1024]:
                _hits.pop(stale, None)


async def guard(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Dependency applied to every /api route (except health)."""
    if settings.access_key:
        if not x_api_key or not secrets.compare_digest(x_api_key, settings.access_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header.",
            )
    _check_rate_limit(_client_ip(request))
