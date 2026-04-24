"""Persistent config cache for PsyMew Gemini auth.

Stores discovered settings (GCP project, auth mode) in ~/.psymew/config.json
so that auto-discovery only needs to happen once.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".psymew"
_CACHE_FILE = _CACHE_DIR / "config.json"
_OAUTH_FILE = _CACHE_DIR / "oauth_creds.json"


def _ensure_dir() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cache() -> dict:
    """Load cached config, or return empty dict."""
    if _CACHE_FILE.is_file():
        try:
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(data: dict) -> None:
    """Merge data into persistent cache."""
    _ensure_dir()
    existing = load_cache()
    existing.update(data)
    _CACHE_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.debug("Saved config cache to %s", _CACHE_FILE)


def get_cached_project() -> Optional[str]:
    """Return cached GCP project ID, or None."""
    return load_cache().get("gcp_project")


def save_project(project_id: str) -> None:
    """Cache a discovered GCP project ID."""
    import datetime
    save_cache({
        "gcp_project": project_id,
        "discovered_at": datetime.datetime.now().isoformat(),
    })
    logger.info("Cached GCP project '%s' to %s", project_id, _CACHE_FILE)


def get_psymew_oauth_path() -> Path:
    """Return the path to PsyMew's own OAuth credential file."""
    return _OAUTH_FILE


def save_oauth_creds(creds_data: dict) -> None:
    """Save OAuth credentials to PsyMew's own credential file."""
    _ensure_dir()
    _OAUTH_FILE.write_text(json.dumps(creds_data, indent=2), encoding="utf-8")
    logger.info("Saved OAuth credentials to %s", _OAUTH_FILE)
