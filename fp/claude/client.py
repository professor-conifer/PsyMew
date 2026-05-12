"""Lazy singleton Anthropic client.

Authentication is API-key only.  Set ANTHROPIC_API_KEY in your `.env` file,
pass --claude-api-key on the command line, or paste it into the GUI's AI tab.
Keys are issued at: https://console.anthropic.com/settings/keys
"""

import logging
import os
import threading
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

_client: Optional[anthropic.Anthropic] = None
_async_client: Optional[anthropic.AsyncAnthropic] = None
_lock = threading.Lock()


def _sanitize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


class ClaudeAuthError(Exception):
    """Raised when no usable Claude API key was found."""


def _resolve_clients(
    api_key_override: Optional[str] = None,
) -> tuple[anthropic.Anthropic, anthropic.AsyncAnthropic]:
    api_key = (
        _sanitize(api_key_override)
        or _sanitize(os.environ.get("ANTHROPIC_API_KEY"))
    )
    if not api_key:
        raise ClaudeAuthError(
            "No Anthropic API key found.\n\n"
            "  Set ANTHROPIC_API_KEY in your .env file or pass --claude-api-key.\n"
            "  Keys are created at: https://console.anthropic.com/settings/keys\n"
            "  Or open the GUI: `python psymew_gui.py`\n"
        )

    logger.info("Claude auth: using API key")
    return anthropic.Anthropic(api_key=api_key), anthropic.AsyncAnthropic(api_key=api_key)


def get_client(
    auth_mode: str = "api_key",  # kept for backwards-compatible call sites
    api_key_override: Optional[str] = None,
) -> anthropic.Anthropic:
    """Return the lazily-initialized synchronous Anthropic client singleton."""
    global _client, _async_client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        _client, _async_client = _resolve_clients(api_key_override)
        return _client


def get_async_client(
    auth_mode: str = "api_key",
    api_key_override: Optional[str] = None,
) -> anthropic.AsyncAnthropic:
    """Return the lazily-initialized async Anthropic client singleton."""
    global _client, _async_client
    if _async_client is not None:
        return _async_client
    with _lock:
        if _async_client is not None:
            return _async_client
        _client, _async_client = _resolve_clients(api_key_override)
        return _async_client


def get_model_name() -> str:
    """Return the configured model name from config, with fallback."""
    from config import FoulPlayConfig

    return getattr(FoulPlayConfig, "claude_model", "claude-sonnet-4-6")
