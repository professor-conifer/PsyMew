"""Lazy singleton DeepSeek client (OpenAI-compatible API).

DeepSeek uses API key auth only — set DEEPSEEK_API_KEY env var
or pass --deepseek-api-key.  Keys are created at:
  https://platform.deepseek.com/api_keys
"""

import logging
import os
import threading
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None
_lock = threading.Lock()


def _sanitize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


class DeepSeekAuthError(Exception):
    """Raised when no valid DeepSeek API key was found."""


def _resolve_client(api_key_override: Optional[str] = None) -> AsyncOpenAI:
    api_key = (
        _sanitize(api_key_override)
        or _sanitize(os.environ.get("DEEPSEEK_API_KEY"))
    )
    if not api_key:
        raise DeepSeekAuthError(
            "No DeepSeek API key found.\n\n"
            "  Set DEEPSEEK_API_KEY in your .env file or pass --deepseek-api-key.\n"
            "  Keys are created at: https://platform.deepseek.com/api_keys\n"
        )

    logger.info("DeepSeek auth: using API key")
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def get_client(api_key_override: Optional[str] = None) -> AsyncOpenAI:
    """Return the lazily-initialized DeepSeek async client singleton.

    Thread-safe.  Uses DEEPSEEK_API_KEY from env or .env file, or an
    explicit override from the --deepseek-api-key CLI flag.
    """
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        _client = _resolve_client(api_key_override)
        return _client


def get_model_name() -> str:
    """Return the configured DeepSeek model name from config."""
    from config import FoulPlayConfig

    return getattr(FoulPlayConfig, "deepseek_model", "deepseek-v4-pro")
