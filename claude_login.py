#!/usr/bin/env python3
"""Interactive helper to set up Anthropic (Claude) API key authentication for PsyMew.

If you'd rather not use the terminal, run `python psymew_gui.py` instead —
the GUI's AI tab has the same flow with a "Test connection" button.

This script:
  - Detects an existing ANTHROPIC_API_KEY from your environment or .env file.
  - Tests it against the Anthropic API.
  - Optionally writes a new key into your .env file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ENV_FILE = REPO_ROOT / ".env"


def _load_env_file() -> dict[str, str]:
    if not ENV_FILE.is_file():
        return {}
    out: dict[str, str] = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            out[key.strip()] = val.strip().strip("'\"")
    return out


def _save_key_to_env(key: str) -> None:
    """Insert or update ANTHROPIC_API_KEY in .env, preserving other lines."""
    existing = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.is_file() else ""
    lines = existing.splitlines()
    replaced = False
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("ANTHROPIC_API_KEY=") or stripped.startswith(
            "# ANTHROPIC_API_KEY="
        ):
            lines[i] = f"ANTHROPIC_API_KEY={key}"
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"ANTHROPIC_API_KEY={key}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _test_api_key(key: str) -> tuple[bool, str]:
    try:
        import anthropic  # type: ignore
    except ImportError:
        return False, "anthropic package is not installed. Run: pip install -r requirements.txt"
    try:
        client = anthropic.Anthropic(api_key=key, timeout=8.0)
        models = list(client.models.list(limit=1))
        if not models:
            return False, "Authentication succeeded but the response was empty."
        return True, f"OK — {models[0].id} is reachable."
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc.__class__.__name__}: {exc}"


def _detect_existing_key() -> tuple[str, str]:
    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env_key:
        return env_key, "environment variable ANTHROPIC_API_KEY"
    file_key = _load_env_file().get("ANTHROPIC_API_KEY", "").strip()
    if file_key:
        return file_key, f"{ENV_FILE}"
    return "", ""


def main() -> int:
    print("=" * 60)
    print("  PsyMew — Anthropic (Claude) API key setup")
    print("=" * 60)
    print()

    existing_key, source = _detect_existing_key()
    if existing_key:
        masked = existing_key[:6] + "…" + existing_key[-4:] if len(existing_key) > 12 else "(short)"
        print(f"Found existing key ({masked}) in {source}.")
        print("Testing it against the Anthropic API…")
        ok, msg = _test_api_key(existing_key)
        if ok:
            print(f"  ✓ {msg}")
            print("\nYou're all set. Run `python start.py` or `python psymew_gui.py`.\n")
            return 0
        print(f"  ✗ {msg}")
        print()

    print("Get a key at: https://console.anthropic.com/settings/keys")
    print("(Tip: the GUI version is at `python psymew_gui.py`.)")
    print()
    try:
        new_key = input("Paste your Anthropic API key (or blank to abort): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 1
    if not new_key:
        print("Aborted.")
        return 1

    print("Testing the new key…")
    ok, msg = _test_api_key(new_key)
    if not ok:
        print(f"  ✗ {msg}")
        return 1
    print(f"  ✓ {msg}")

    try:
        save = input(f"Save this key to {ENV_FILE}? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        save = "n"
    if save in ("", "y", "yes"):
        _save_key_to_env(new_key)
        print(f"Saved to {ENV_FILE}.")
    else:
        print("Key not saved. Set ANTHROPIC_API_KEY in your shell to use it.")
    print("\nYou're all set. Run `python start.py` or `python psymew_gui.py`.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
