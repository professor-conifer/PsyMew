"""Lazy singleton Gemini client.

Authentication is API-key only.  Set GEMINI_API_KEY (or GOOGLE_API_KEY) in
your `.env` file, pass --gemini-api-key on the command line, or paste it
into the GUI's AI tab.
Keys are issued at: https://aistudio.google.com/apikey
"""

import logging
import os
import threading
from typing import Optional

from google import genai

from fp.gemini.errors import GeminiAuthError

logger = logging.getLogger(__name__)

_client: Optional[genai.Client] = None
_lock = threading.Lock()


def _sanitize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _resolve_client(api_key_override: Optional[str] = None) -> genai.Client:
    api_key = (
        _sanitize(api_key_override)
        or _sanitize(os.environ.get("GEMINI_API_KEY"))
        or _sanitize(os.environ.get("GOOGLE_API_KEY"))
    )
    if not api_key:
        raise GeminiAuthError(
            "No Gemini API key found.\n\n"
            "  Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file,\n"
            "  pass --gemini-api-key, or open the GUI: `python psymew_gui.py`.\n"
            "  Keys are issued at: https://aistudio.google.com/apikey\n"
        )

    logger.info("Gemini auth: using API key")
    return genai.Client(api_key=api_key)


def get_client(
    auth_mode: str = "api_key",  # kept for backwards-compatible call sites
    api_key_override: Optional[str] = None,
) -> genai.Client:
    """Return the lazily-initialized Gemini client singleton."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        _client = _resolve_client(api_key_override)
        return _client


def get_model_name() -> str:
    """Return the configured model name from config, with fallback."""
    from config import FoulPlayConfig

    return getattr(FoulPlayConfig, "gemini_model", "gemini-2.5-pro")
