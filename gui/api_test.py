"""Per-engine API key connection tests.

Each function is synchronous and short — callers should run them on a
worker thread to avoid blocking the GUI. Returns a `(ok, message)`
tuple; `ok=False` carries a short error string for tooltip display.
"""

from __future__ import annotations

import socket
from typing import Tuple

_DEFAULT_TIMEOUT = 8.0


def test_claude(api_key: str) -> Tuple[bool, str]:
    if not api_key.strip():
        return False, "API key is empty."
    try:
        import anthropic  # type: ignore
    except ImportError:
        return False, "anthropic SDK not installed."
    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=_DEFAULT_TIMEOUT)
        models = list(client.models.list(limit=1))
        if not models:
            return False, "Authentication succeeded but no models were returned."
        return True, "Connection OK."
    except Exception as exc:  # noqa: BLE001
        return False, _first_line(str(exc)) or exc.__class__.__name__


def test_gemini(api_key: str) -> Tuple[bool, str]:
    if not api_key.strip():
        return False, "API key is empty."
    try:
        from google import genai  # type: ignore
    except ImportError:
        return False, "google-genai SDK not installed."
    try:
        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
        if not models:
            return False, "Authentication succeeded but no models were returned."
        return True, "Connection OK."
    except Exception as exc:  # noqa: BLE001
        return False, _first_line(str(exc)) or exc.__class__.__name__


def test_deepseek(api_key: str) -> Tuple[bool, str]:
    if not api_key.strip():
        return False, "API key is empty."
    try:
        import requests  # type: ignore
    except ImportError:
        return False, "requests not installed."
    socket.setdefaulttimeout(_DEFAULT_TIMEOUT)
    try:
        r = requests.get(
            "https://api.deepseek.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, _first_line(str(exc)) or "Network error."
    if r.status_code == 401:
        return False, "Invalid API key (401)."
    if r.status_code >= 400:
        return False, f"HTTP {r.status_code}: {_first_line(r.text)}"
    try:
        body = r.json()
    except ValueError:
        return False, "Response was not JSON."
    if isinstance(body, dict) and body.get("data"):
        return True, "Connection OK."
    return True, "Reachable (response shape unfamiliar)."


def _first_line(text: str) -> str:
    if not text:
        return ""
    line = text.splitlines()[0]
    return line[:200]
