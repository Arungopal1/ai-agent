"""Configuration for the agent service.

Everything is read from environment variables so the same code runs locally
(with a ``.env`` file) and on Vercel (with project env vars).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

TRUTHY = {"1", "true", "yes", "on"}

PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
        "keys_url": "https://console.groq.com/keys",
        "needs_key": True,
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "keys_url": "https://openrouter.ai/keys",
        "needs_key": True,
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "keys_url": "https://aistudio.google.com/apikey",
        "needs_key": True,
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "keys_url": "https://platform.openai.com/api-keys",
        "needs_key": True,
    },
    "ollama": {
        "label": "Ollama (local)",
        "base_url": "http://localhost:11434/v1",
        "key_env": "OLLAMA_API_KEY",
        "default_model": "qwen2.5:7b",
        "keys_url": "https://ollama.com/download",
        "needs_key": False,
    },
}

# Order used when LLM_PROVIDER is not set: first provider with a key wins.
AUTO_ORDER = ("groq", "openrouter", "gemini", "openai", "ollama")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in TRUTHY


def _pick_provider() -> str:
    explicit = _env("LLM_PROVIDER").lower()
    if explicit in PROVIDERS:
        return explicit
    for name in AUTO_ORDER:
        cfg = PROVIDERS[name]
        if _env(str(cfg["key_env"])):
            return name
    return "mock"  # no key configured -> readable demo mode instead of a 500


@dataclass(frozen=True)
class Settings:
    # --- LLM -------------------------------------------------------------
    provider: str
    provider_label: str
    base_url: str
    api_key: str
    model: str
    keys_url: str
    temperature: float | None

    # --- Embeddings / RAG -------------------------------------------------
    embedding_backend: str  # "gemini" | "hashing"
    embedding_model: str
    embedding_key: str

    # --- Optional integrations -------------------------------------------
    tavily_key: str

    # --- Behaviour --------------------------------------------------------
    max_steps: int
    request_timeout: float
    max_history_messages: int
    max_message_chars: int
    max_upload_mb: int

    # --- HTTP surface -----------------------------------------------------
    access_key: str
    allowed_origins: str
    rate_limit_per_min: int

    @property
    def configured(self) -> bool:
        """True when a real LLM backend is usable."""
        return self.provider != "mock"

    @property
    def origin_list(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    def public_dict(self) -> dict[str, Any]:
        """Safe-to-expose subset for the frontend."""
        return {
            "provider": self.provider,
            "provider_label": self.provider_label,
            "model": self.model,
            "configured": self.configured,
            "keys_url": self.keys_url,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model
            if self.embedding_backend != "hashing"
            else "local-hashing (demo)",
            "max_upload_mb": self.max_upload_mb,
            "max_steps": self.max_steps,
            "auth_required": bool(self.access_key),
        }


def _build_settings() -> Settings:
    provider = _pick_provider()
    cfg = PROVIDERS.get(provider, {})

    api_key = _env(str(cfg.get("key_env", "")), "")
    base_url = _env("LLM_BASE_URL", str(cfg.get("base_url", "http://localhost:11434/v1")))
    model = _env("LLM_MODEL", str(cfg.get("default_model", "gpt-4o-mini")))

    # Ollama on a custom port, or any OpenAI-compatible gateway.
    if provider == "ollama":
        base_url = _env("OLLAMA_BASE_URL", base_url)

    embedding_backend = _env("EMBEDDING_BACKEND").lower()
    gemini_key = _env("GEMINI_API_KEY")
    if not embedding_backend:
        embedding_backend = "gemini" if gemini_key else "hashing"
    if embedding_backend == "gemini" and not gemini_key:
        embedding_backend = "hashing"

    temperature_raw = _env("LLM_TEMPERATURE")

    return Settings(
        provider=provider,
        provider_label=str(cfg.get("label", "Demo mode")) if cfg else "Demo mode",
        base_url=base_url,
        api_key=api_key,
        model=model if cfg else "demo",
        keys_url=str(cfg.get("keys_url", "")) if cfg else "",
        temperature=_env_float("LLM_TEMPERATURE", 0.3) if temperature_raw else None,
        embedding_backend=embedding_backend,
        embedding_model=_env("EMBEDDING_MODEL", "gemini-embedding-001"),
        embedding_key=gemini_key,
        tavily_key=_env("TAVILY_API_KEY"),
        max_steps=max(1, _env_int("AGENT_MAX_STEPS", 5)),
        request_timeout=_env_float("REQUEST_TIMEOUT", 45.0),
        max_history_messages=max(2, _env_int("MAX_HISTORY_MESSAGES", 20)),
        max_message_chars=max(500, _env_int("MAX_MESSAGE_CHARS", 8000)),
        max_upload_mb=max(1, _env_int("MAX_UPLOAD_MB", 4)),
        access_key=_env("AGENT_API_KEY"),
        allowed_origins=_env("ALLOWED_ORIGINS", "*"),
        rate_limit_per_min=_env_int("RATE_LIMIT_PER_MIN", 30),
    )


settings = _build_settings()
