"""Lazy singleton Gemini client with multi-mode auth resolution.

Auth modes (tried in order for 'auto'):
  1. oauth       – PsyMew OAuth (~/.psymew/oauth_creds.json) or Gemini CLI fallback
  2. api_key     – GEMINI_API_KEY / GOOGLE_API_KEY env var
  3. access_token – GEMINI_ACCESS_TOKEN env var
  4. adc         – gcloud Application Default Credentials
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from google import genai

from fp.gemini.errors import GeminiAuthError

logger = logging.getLogger(__name__)

_client: Optional[genai.Client] = None
_lock = threading.Lock()

# Scopes required for ADC auth against Gemini / Vertex AI
_ADC_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Scopes for Google Generative Language API (used by PsyMew's own OAuth flow).
# Including generative-language scope lets us use the simpler API endpoint
# without needing Vertex AI / GCP project discovery.
_PSYMEW_OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/generative-language.retriever",
    "https://www.googleapis.com/auth/cloud-platform",
    "openid",
]

# Gemini CLI's public OAuth client credentials (from @google/gemini-cli npm package).
# These are public/PKCE credentials — security is maintained by per-user refresh tokens.
_GEMINI_CLI_CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
_GEMINI_CLI_CLIENT_SECRET = "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"


def _sanitize(value: Optional[str]) -> Optional[str]:
    """Return stripped value or None if blank."""
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


# ---------------------------------------------------------------------------
# OAuth credential paths
# ---------------------------------------------------------------------------


def _get_oauth_credential_paths() -> list[Path]:
    """Return possible OAuth credential file paths, in priority order.

    PsyMew's own creds are checked first, then Gemini CLI fallback paths.
    """
    paths: list[Path] = []

    # Explicit override via env var
    explicit = _sanitize(os.environ.get("GEMINI_OAUTH_CREDS_FILE"))
    if explicit:
        paths.append(Path(explicit))

    # PsyMew's own OAuth credentials (created by `python gemini_login.py`)
    from fp.gemini.cache import get_psymew_oauth_path
    paths.append(get_psymew_oauth_path())

    # Gemini CLI fallback: ~/.gemini/oauth_creds.json
    home = Path.home()
    paths.append(home / ".gemini" / "oauth_creds.json")

    # Windows: check %USERPROFILE%\.gemini\ as well (usually same as home)
    userprofile = _sanitize(os.environ.get("USERPROFILE"))
    if userprofile:
        p = Path(userprofile) / ".gemini" / "oauth_creds.json"
        if p not in paths:
            paths.append(p)

    # Also check %APPDATA%\.gemini\ (some CLI versions use this)
    appdata = _sanitize(os.environ.get("APPDATA"))
    if appdata:
        p = Path(appdata) / ".gemini" / "oauth_creds.json"
        if p not in paths:
            paths.append(p)

    return paths


def _find_oauth_creds_file() -> Optional[Path]:
    """Return the first existing OAuth credentials file, or None."""
    for p in _get_oauth_credential_paths():
        if p.is_file():
            return p
    return None


def _load_oauth_credentials(creds_path: Path):
    """Load OAuth credentials from a JSON file, handling both PsyMew and Gemini CLI formats.

    Returns (Credentials, creds_info_dict) or (None, None) on failure.
    """
    import datetime
    from google.oauth2.credentials import Credentials

    creds_info = json.loads(creds_path.read_text(encoding="utf-8"))

    # Map field names: Gemini CLI uses 'access_token', PsyMew/standard uses 'token'
    token = creds_info.get("access_token") or creds_info.get("token")
    refresh_token = creds_info.get("refresh_token")
    client_id = creds_info.get("client_id", _GEMINI_CLI_CLIENT_ID)
    client_secret = creds_info.get("client_secret", _GEMINI_CLI_CLIENT_SECRET)

    if not token and not refresh_token:
        return None, None

    # Parse expiry — Gemini CLI uses epoch ms, PsyMew/standard uses ISO string.
    # google-auth compares expiry with utcnow() (naive UTC), so we produce naive datetimes.
    expiry = None
    expiry_raw = creds_info.get("expiry_date") or creds_info.get("expiry")
    if expiry_raw:
        if isinstance(expiry_raw, (int, float)):
            expiry = datetime.datetime.utcfromtimestamp(expiry_raw / 1000)
        elif isinstance(expiry_raw, str):
            try:
                dt = datetime.datetime.fromisoformat(expiry_raw)
                expiry = dt.replace(tzinfo=None)
            except ValueError:
                pass

    creds = Credentials(
        token=token,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        expiry=expiry,
    )
    return creds, creds_info


def _has_generative_language_scope(creds_info: dict) -> bool:
    """Check if credentials were issued with generative-language scope."""
    scope = creds_info.get("scope", "")
    return "generative-language" in scope


def _try_oauth() -> Optional[genai.Client]:
    """Attempt to create a client using OAuth credentials.

    Checks two credential sources in order:
    1. PsyMew's own OAuth creds (~/.psymew/oauth_creds.json)
       - These have the generative-language scope → can use the simple API directly
    2. Gemini CLI creds (~/.gemini/oauth_creds.json)
       - These only have cloud-platform scope → need Vertex AI + GCP project

    Auto-refreshes expired tokens and saves them back.
    """
    creds_path = _find_oauth_creds_file()
    if creds_path is None:
        return None

    try:
        from google.auth.transport.requests import Request as AuthRequest

        creds, creds_info = _load_oauth_credentials(creds_path)
        if creds is None:
            return None

        # Refresh if expired
        if (not creds.valid or creds.expired) and creds.refresh_token:
            logger.info("OAuth: refreshing token...")
            creds.refresh(AuthRequest())
            _save_oauth_creds(creds_path, creds, creds_info)

        if not creds.valid:
            logger.warning("OAuth: credentials invalid after refresh")
            return None

        # Check which API path to use based on scopes
        if _has_generative_language_scope(creds_info):
            # PsyMew OAuth flow — has the right scopes, use the simple API
            logger.info(
                "Gemini auth: using PsyMew OAuth from %s (direct API, Pro limits apply)",
                creds_path,
            )
            return genai.Client(credentials=creds)
        else:
            # Gemini CLI fallback — cloud-platform scope only, needs Vertex AI
            return _create_vertexai_client(creds, creds_path)

    except ImportError as exc:
        logger.warning(
            "OAuth: missing dependency: %s. Run: pip install google-auth requests",
            exc,
        )
        return None
    except Exception as exc:
        logger.warning("OAuth: failed to load credentials: %s", exc)
        return None


def _create_vertexai_client(creds, creds_path: Path) -> Optional[genai.Client]:
    """Create a Vertex AI client for credentials that lack generative-language scope.

    Discovers a GCP project (from env, cache, or auto-discovery) and uses the
    Vertex AI global endpoint.
    """
    from fp.gemini.cache import get_cached_project, save_project

    # Try env var → cache → auto-discover
    project = (
        _sanitize(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        or _sanitize(os.environ.get("GCLOUD_PROJECT"))
        or get_cached_project()
    )

    if project:
        # Validate cached/env project still works
        try:
            client = genai.Client(
                vertexai=True, credentials=creds, project=project, location="global"
            )
            client.models.generate_content(model="gemini-2.5-flash", contents="OK")
            logger.info(
                "Gemini auth: using Gemini CLI OAuth from %s "
                "(vertexai, project=%s, Pro limits apply)",
                creds_path,
                project,
            )
            return client
        except Exception:
            logger.debug("Cached project '%s' failed, re-discovering...", project)

    # Auto-discover
    project = _discover_gcp_project(creds)
    if not project:
        logger.warning(
            "OAuth: could not find a working GCP project. "
            "Set GOOGLE_CLOUD_PROJECT env var or run `python gemini_login.py` "
            "to set up PsyMew's own OAuth (recommended)."
        )
        return None

    save_project(project)

    logger.info(
        "Gemini auth: using Gemini CLI OAuth from %s "
        "(vertexai, project=%s, Pro limits apply)",
        creds_path,
        project,
    )
    return genai.Client(
        vertexai=True, credentials=creds, project=project, location="global"
    )


def _discover_gcp_project(creds) -> Optional[str]:
    """Auto-discover a usable GCP project for Gemini via the user's OAuth creds.

    Lists the user's GCP projects and tries each one until Gemini responds.
    Prefers projects named like 'gen-lang-client-*' (auto-created by AI Studio).
    """
    try:
        import requests as http_requests

        r = http_requests.get(
            "https://cloudresourcemanager.googleapis.com/v1/projects",
            headers={"Authorization": "Bearer " + creds.token},
            timeout=10,
        )
        if r.status_code != 200:
            logger.debug("OAuth: could not list projects: %s", r.status_code)
            return None

        projects = r.json().get("projects", [])
        if not projects:
            return None

        # Sort: prefer gen-lang-client projects (auto-created by AI Studio)
        projects.sort(
            key=lambda p: (
                0 if p.get("projectId", "").startswith("gen-lang-client") else 1,
                p.get("projectId", ""),
            )
        )

        # Try each project with a real Gemini call
        for proj in projects:
            pid = proj.get("projectId")
            if not pid:
                continue
            try:
                client = genai.Client(
                    vertexai=True,
                    credentials=creds,
                    project=pid,
                    location="global",
                )
                resp = client.models.generate_content(
                    model="gemini-2.5-flash", contents="Say OK"
                )
                if resp and resp.text:
                    logger.info("OAuth: auto-discovered project '%s'", pid)
                    return pid
            except Exception:
                logger.debug("OAuth: project '%s' did not work", pid)
                continue

        return None
    except Exception as exc:
        logger.debug("OAuth: project discovery failed: %s", exc)
        return None


def _save_oauth_creds(path: Path, creds, original_info: dict) -> None:
    """Save refreshed OAuth credentials back to the JSON file.

    Preserves the source format (Gemini CLI or PsyMew standard).
    """
    try:
        updated = {**original_info}
        if "access_token" in original_info:
            updated["access_token"] = creds.token
        else:
            updated["token"] = creds.token
        if creds.expiry:
            import datetime
            if "expiry_date" in original_info:
                # creds.expiry is a naive datetime representing UTC
                utc_expiry = creds.expiry.replace(tzinfo=datetime.timezone.utc)
                updated["expiry_date"] = int(utc_expiry.timestamp() * 1000)
            else:
                updated["expiry"] = creds.expiry.isoformat()
        path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        logger.debug("OAuth: saved refreshed token to %s", path)
    except Exception as exc:
        logger.warning("OAuth: could not save refreshed token: %s", exc)


# ---------------------------------------------------------------------------
# ADC credential paths
# ---------------------------------------------------------------------------


def _get_adc_credential_paths() -> list[str]:
    """Return possible ADC credential file paths (cross-platform)."""
    paths: list[str] = []

    explicit = _sanitize(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    if explicit:
        paths.append(explicit)

    # Unix-like default
    home = Path.home()
    paths.append(str(home / ".config" / "gcloud" / "application_default_credentials.json"))

    # Windows %APPDATA% default
    appdata = _sanitize(os.environ.get("APPDATA"))
    if appdata:
        paths.append(str(Path(appdata) / "gcloud" / "application_default_credentials.json"))

    return paths


def _has_adc_credentials() -> bool:
    """Check if any ADC credential file exists on disk."""
    return any(Path(p).is_file() for p in _get_adc_credential_paths())


# ---------------------------------------------------------------------------
# Auth resolvers
# ---------------------------------------------------------------------------


def _try_api_key() -> Optional[genai.Client]:
    """Attempt to create a client using an API key."""
    api_key = _sanitize(os.environ.get("GEMINI_API_KEY")) or _sanitize(
        os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        return None
    logger.info("Gemini auth: using API key")
    return genai.Client(api_key=api_key)


def _try_access_token() -> Optional[genai.Client]:
    """Attempt to create a client using an access token."""
    token = _sanitize(os.environ.get("GEMINI_ACCESS_TOKEN"))
    if not token:
        return None
    logger.info("Gemini auth: using access token")
    # Access tokens are passed as API keys; the SDK handles the rest
    return genai.Client(api_key=token)


def _try_adc() -> Optional[genai.Client]:
    """Attempt to create a client using Application Default Credentials."""
    if not _has_adc_credentials():
        return None
    try:
        import google.auth  # type: ignore[import-untyped]

        credentials, project = google.auth.default(scopes=_ADC_SCOPES)
        logger.info(
            "Gemini auth: using ADC (project=%s)",
            project or "unset",
        )
        return genai.Client(credentials=credentials)
    except Exception as exc:
        logger.warning("ADC auth failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Client resolution
# ---------------------------------------------------------------------------


def _resolve_client(auth_mode: str, api_key_override: Optional[str] = None) -> genai.Client:
    """Resolve a genai.Client based on the requested auth mode.

    Parameters
    ----------
    auth_mode : str
        One of 'auto', 'api_key', 'access_token', 'adc', 'oauth'.
    api_key_override : str | None
        Explicit API key from --gemini-api-key flag (takes priority over env).
    """
    # If an explicit key was passed on the command line, inject it
    if api_key_override:
        os.environ.setdefault("GEMINI_API_KEY", api_key_override)

    resolvers = {
        "oauth": [_try_oauth],
        "api_key": [_try_api_key],
        "access_token": [_try_access_token],
        "adc": [_try_adc],
        # Auto: try OAuth first (for Pro subscription limits), then API key, etc.
        "auto": [_try_oauth, _try_api_key, _try_access_token, _try_adc],
    }

    attempts = resolvers.get(auth_mode, resolvers["auto"])
    for resolver in attempts:
        client = resolver()
        if client is not None:
            return client

    raise GeminiAuthError(
        f"No valid Gemini credentials found for auth_mode='{auth_mode}'.\n\n"
        "  Quick setup:  python gemini_login.py\n"
        "  Or set:       GEMINI_API_KEY=your-key-from-aistudio.google.com\n\n"
        "  For Google AI Pro users (higher rate limits):\n"
        "    python gemini_login.py  →  choose 'Google OAuth'\n"
    )


def get_client(
    auth_mode: str = "auto",
    api_key_override: Optional[str] = None,
) -> genai.Client:
    """Return the lazily-initialized Gemini client singleton.

    Thread-safe. First call resolves credentials; subsequent calls return
    the cached client.
    """
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        _client = _resolve_client(auth_mode, api_key_override)
        return _client


def get_model_name() -> str:
    """Return the configured model name from config, with fallback."""
    from config import FoulPlayConfig

    return getattr(FoulPlayConfig, "gemini_model", "gemini-2.5-pro")
