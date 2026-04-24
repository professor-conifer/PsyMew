"""Lazy singleton Anthropic client with multi-mode auth resolution.

Auth modes (tried in order for 'auto'):
  1. oauth   – Reuse Claude Code's stored OAuth credentials
               (~/.claude/.credentials.json on Linux/Windows, macOS Keychain fallback)
  2. api_key – ANTHROPIC_API_KEY env var or --claude-api-key flag
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

_client: Optional[anthropic.Anthropic] = None
_async_client: Optional[anthropic.AsyncAnthropic] = None
_lock = threading.Lock()

# Claude Code OAuth token refresh endpoint
_OAUTH_TOKEN_URL = "https://claude.ai/v1/oauth/token"

# Beta headers required when using Claude Code OAuth tokens
_OAUTH_BETA_HEADERS = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,prompt-caching-scope-2026-01-05"


def _sanitize(value: Optional[str]) -> Optional[str]:
    """Return stripped value or None if blank."""
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


# ---------------------------------------------------------------------------
# Claude Code credential paths
# ---------------------------------------------------------------------------


def _get_credential_paths() -> list[Path]:
    """Return possible Claude Code credential file paths, in priority order."""
    paths: list[Path] = []

    # Explicit override via env var
    explicit = _sanitize(os.environ.get("CLAUDE_CONFIG_DIR"))
    if explicit:
        paths.append(Path(explicit) / ".credentials.json")

    # Standard location: ~/.claude/.credentials.json
    home = Path.home()
    paths.append(home / ".claude" / ".credentials.json")

    # Windows: check %USERPROFILE%\.claude\ as well
    userprofile = _sanitize(os.environ.get("USERPROFILE"))
    if userprofile:
        p = Path(userprofile) / ".claude" / ".credentials.json"
        if p not in paths:
            paths.append(p)

    # Windows: %APPDATA%\.claude\
    appdata = _sanitize(os.environ.get("APPDATA"))
    if appdata:
        p = Path(appdata) / ".claude" / ".credentials.json"
        if p not in paths:
            paths.append(p)

    return paths


def _find_credential_file() -> Optional[Path]:
    """Return the first existing Claude Code credentials file, or None."""
    for p in _get_credential_paths():
        if p.is_file():
            return p
    return None


def _load_oauth_credentials(creds_path: Path) -> Optional[dict]:
    """Load and parse Claude Code OAuth credentials from a JSON file.

    Returns the credential dict with keys: accessToken, refreshToken, expiresAt, etc.
    Returns None on failure.
    """
    try:
        raw = json.loads(creds_path.read_text(encoding="utf-8"))
        # Claude Code stores creds under "claudeAiOauth" key
        creds = raw.get("claudeAiOauth")
        if not creds:
            # Fallback: try top-level (in case format changes)
            if "accessToken" in raw:
                creds = raw
            else:
                return None

        if not creds.get("accessToken"):
            return None

        return creds
    except Exception as exc:
        logger.warning("Could not read Claude Code credentials: %s", exc)
        return None


def _is_token_expired(creds: dict) -> bool:
    """Check if the access token is expired or about to expire (60s buffer)."""
    expires_at = creds.get("expiresAt")
    if not expires_at:
        return True
    # expiresAt is epoch milliseconds
    return (time.time() * 1000) >= (expires_at - 60_000)


def _refresh_oauth_token(creds: dict, creds_path: Path) -> Optional[str]:
    """Refresh the OAuth access token using the stored refresh token.

    Returns the new access token, or None on failure.
    Writes refreshed tokens back to the credentials file (refresh tokens rotate).
    """
    refresh_token = creds.get("refreshToken")
    if not refresh_token:
        logger.warning("OAuth: no refresh token available")
        return None

    try:
        import requests as http_requests

        resp = http_requests.post(
            _OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={"Content-Type": "application/json"},
            timeout=15,
        )

        if resp.status_code != 200:
            logger.warning(
                "OAuth: token refresh failed (HTTP %d): %s",
                resp.status_code,
                resp.text[:200],
            )
            return None

        data = resp.json()
        new_access = data.get("access_token")
        new_refresh = data.get("refresh_token")
        expires_in = data.get("expires_in", 3600)

        if not new_access:
            logger.warning("OAuth: refresh response missing access_token")
            return None

        # Update the creds dict
        creds["accessToken"] = new_access
        if new_refresh:
            creds["refreshToken"] = new_refresh
        creds["expiresAt"] = int((time.time() + expires_in) * 1000)

        # Write back to file (refresh tokens rotate on each use)
        try:
            raw = json.loads(creds_path.read_text(encoding="utf-8"))
            raw["claudeAiOauth"] = creds
            creds_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            logger.debug("OAuth: saved refreshed token to %s", creds_path)
        except Exception as exc:
            logger.warning("OAuth: could not save refreshed token: %s", exc)

        logger.info("OAuth: token refreshed successfully")
        return new_access

    except ImportError:
        logger.warning("OAuth: 'requests' package required for token refresh")
        return None
    except Exception as exc:
        logger.warning("OAuth: token refresh failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Auth resolvers
# ---------------------------------------------------------------------------


def _try_oauth() -> Optional[tuple[anthropic.Anthropic, anthropic.AsyncAnthropic]]:
    """Attempt to create clients using Claude Code's stored OAuth credentials."""
    creds_path = _find_credential_file()
    if creds_path is None:
        return None

    creds = _load_oauth_credentials(creds_path)
    if creds is None:
        return None

    access_token = creds.get("accessToken")

    # Refresh if expired
    if _is_token_expired(creds):
        logger.info("OAuth: token expired, refreshing...")
        access_token = _refresh_oauth_token(creds, creds_path)
        if not access_token:
            return None

    sub_type = creds.get("subscriptionType", "unknown")
    logger.info(
        "Claude auth: using Claude Code OAuth from %s (subscription: %s)",
        creds_path,
        sub_type,
    )

    # Create clients with OAuth bearer token auth
    # We use the auth_token parameter which sets Authorization: Bearer header
    sync_client = anthropic.Anthropic(
        auth_token=access_token,
        default_headers={"anthropic-beta": _OAUTH_BETA_HEADERS},
    )
    async_client = anthropic.AsyncAnthropic(
        auth_token=access_token,
        default_headers={"anthropic-beta": _OAUTH_BETA_HEADERS},
    )

    return sync_client, async_client


def _try_api_key(api_key_override: Optional[str] = None) -> Optional[tuple[anthropic.Anthropic, anthropic.AsyncAnthropic]]:
    """Attempt to create clients using an API key."""
    api_key = api_key_override or _sanitize(os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key:
        return None

    logger.info("Claude auth: using API key")
    sync_client = anthropic.Anthropic(api_key=api_key)
    async_client = anthropic.AsyncAnthropic(api_key=api_key)
    return sync_client, async_client


# ---------------------------------------------------------------------------
# Client resolution
# ---------------------------------------------------------------------------


class ClaudeAuthError(Exception):
    """Raised when Claude authentication fails (no valid credentials found)."""
    pass


def _resolve_clients(
    auth_mode: str,
    api_key_override: Optional[str] = None,
) -> tuple[anthropic.Anthropic, anthropic.AsyncAnthropic]:
    """Resolve Anthropic clients based on the requested auth mode.

    Parameters
    ----------
    auth_mode : str
        One of 'auto', 'api_key', 'oauth'.
    api_key_override : str | None
        Explicit API key from --claude-api-key flag.
    """
    resolvers = {
        "oauth": [lambda: _try_oauth()],
        "api_key": [lambda: _try_api_key(api_key_override)],
        "auto": [lambda: _try_oauth(), lambda: _try_api_key(api_key_override)],
    }

    attempts = resolvers.get(auth_mode, resolvers["auto"])
    for resolver in attempts:
        result = resolver()
        if result is not None:
            return result

    raise ClaudeAuthError(
        f"No valid Claude credentials found for auth_mode='{auth_mode}'.\n\n"
        "  Option 1: Install Claude Code and run `claude` to log in\n"
        "            (your Pro/Max subscription will be used)\n\n"
        "  Option 2: Set ANTHROPIC_API_KEY=your-key-from-console.anthropic.com\n"
    )


def get_client(
    auth_mode: str = "auto",
    api_key_override: Optional[str] = None,
) -> anthropic.Anthropic:
    """Return the lazily-initialized synchronous Anthropic client singleton.

    Thread-safe. First call resolves credentials; subsequent calls return
    the cached client.
    """
    global _client, _async_client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        _client, _async_client = _resolve_clients(auth_mode, api_key_override)
        return _client


def get_async_client(
    auth_mode: str = "auto",
    api_key_override: Optional[str] = None,
) -> anthropic.AsyncAnthropic:
    """Return the lazily-initialized async Anthropic client singleton."""
    global _client, _async_client
    if _async_client is not None:
        return _async_client
    with _lock:
        if _async_client is not None:
            return _async_client
        _client, _async_client = _resolve_clients(auth_mode, api_key_override)
        return _async_client


def get_model_name() -> str:
    """Return the configured model name from config, with fallback."""
    from config import FoulPlayConfig

    return getattr(FoulPlayConfig, "claude_model", "claude-sonnet-4-6")
