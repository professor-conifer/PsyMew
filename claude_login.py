#!/usr/bin/env python3
"""Interactive Claude credential setup for PsyMew.

Supports:
  - Claude Code OAuth  (auto-detected — recommended for Pro/Max subscribers)
  - API key            (ANTHROPIC_API_KEY)

Usage:
    python claude_login.py
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(val: str | None) -> str | None:
    if val is None:
        return None
    v = val.strip()
    return v if v else None


def _find_claude_code() -> str | None:
    """Find the `claude` CLI binary on PATH."""
    return shutil.which("claude")


def _credential_paths() -> list[Path]:
    """Return possible Claude Code credential file paths."""
    paths: list[Path] = []

    explicit = _sanitize(os.environ.get("CLAUDE_CONFIG_DIR"))
    if explicit:
        paths.append(Path(explicit) / ".credentials.json")

    home = Path.home()
    paths.append(home / ".claude" / ".credentials.json")

    userprofile = _sanitize(os.environ.get("USERPROFILE"))
    if userprofile:
        p = Path(userprofile) / ".claude" / ".credentials.json"
        if p not in paths:
            paths.append(p)

    appdata = _sanitize(os.environ.get("APPDATA"))
    if appdata:
        p = Path(appdata) / ".claude" / ".credentials.json"
        if p not in paths:
            paths.append(p)

    return paths


def _find_credential_file() -> Path | None:
    for p in _credential_paths():
        if p.is_file():
            return p
    return None


def _load_creds(creds_path: Path) -> dict | None:
    """Load Claude Code OAuth creds from disk."""
    try:
        raw = json.loads(creds_path.read_text(encoding="utf-8"))
        creds = raw.get("claudeAiOauth")
        if not creds:
            if "accessToken" in raw:
                creds = raw
            else:
                return None
        if not creds.get("accessToken"):
            return None
        return creds
    except Exception:
        return None


def _test_oauth_client() -> bool:
    """Test that Claude Code OAuth credentials actually work."""
    try:
        from fp.claude.client import _try_oauth
        result = _try_oauth()
        if result is None:
            return False
        sync_client, _ = result

        # Quick model list check
        models = sync_client.models.list(limit=5)
        model_ids = [m.id for m in models.data]
        print(f"  ✓ OAuth credentials valid — {len(model_ids)} models listed")
        return True
    except Exception as exc:
        print(f"  ✗ OAuth test failed: {exc}")
        return False


def _test_api_key(key: str) -> bool:
    """Test that an API key works."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        models = client.models.list(limit=5)
        model_ids = [m.id for m in models.data]
        print(f"  ✓ API key valid — {len(model_ids)} models listed")
        return True
    except Exception as exc:
        print(f"  ✗ API key test failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Detect existing credentials
# ---------------------------------------------------------------------------


def detect_existing():
    """Check for existing Claude credentials. Returns (kind, detail) or None."""
    # Check Claude Code OAuth creds
    creds_path = _find_credential_file()
    if creds_path:
        creds = _load_creds(creds_path)
        if creds:
            sub = creds.get("subscriptionType", "unknown")
            return "oauth", f"{creds_path} (subscription: {sub})"

    # Check API key in env
    api_key = _sanitize(os.environ.get("ANTHROPIC_API_KEY"))
    if api_key:
        return "api_key", api_key

    # Check .env file in repo root
    env_file = Path(__file__).parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
                if key:
                    return "api_key_dotenv", key

    return None


# ---------------------------------------------------------------------------
# Setup flows
# ---------------------------------------------------------------------------


def setup_claude_code():
    """Guide user to install Claude Code and log in for OAuth credentials."""
    print("\n--- Claude Code OAuth Setup (Pro/Max) ---")
    print("This links PsyMew to your Claude Pro or Max subscription.")
    print("Your credentials stay local — never sent to any server.\n")

    # Check if creds already exist and work
    creds_path = _find_credential_file()
    if creds_path:
        creds = _load_creds(creds_path)
        if creds:
            sub = creds.get("subscriptionType", "unknown")
            print(f"Found existing Claude Code credentials: {creds_path}")
            print(f"  Subscription: {sub}")
            print("Testing...")
            if _test_oauth_client():
                return True
            else:
                print("  Credentials exist but didn't work. Re-authenticating...\n")

    # Check if `claude` CLI is installed
    claude_bin = _find_claude_code()
    if claude_bin:
        print(f"Found Claude Code at: {claude_bin}")
        print("Running `claude` to log in...\n")
        print("=" * 50)
        print("  A browser window should open for Anthropic login.")
        print("  Sign in with your Claude Pro/Max account.")
        print("=" * 50)
        print()

        result = subprocess.run([claude_bin], timeout=120)
        if result.returncode != 0:
            print(f"\n  ✗ Claude Code exited with code {result.returncode}")
            # Still check if creds were saved
            creds_path = _find_credential_file()
            if not creds_path:
                print("  No credentials were saved.")
                return False
    else:
        print("Claude Code CLI not found on PATH.\n")
        print("To install it:")
        print("  npm install -g @anthropic-ai/claude-code")
        print("  (requires Node.js — https://nodejs.org/)\n")
        print("After installing, run `claude` once to log in, then re-run this script.\n")

        answer = input("Already have Claude Code installed elsewhere? (y/n): ").strip().lower()
        if answer != "y":
            return False

    # Verify creds were created
    creds_path = _find_credential_file()
    if not creds_path:
        print("\n  ✗ Could not find Claude Code credentials after login.")
        print(f"  Searched: {[str(p) for p in _credential_paths()]}")
        return False

    creds = _load_creds(creds_path)
    if not creds:
        print(f"\n  ✗ Credentials file exists but couldn't be parsed: {creds_path}")
        return False

    sub = creds.get("subscriptionType", "unknown")
    print(f"\n  ✓ Found credentials at {creds_path} (subscription: {sub})")
    print("Testing connection...")
    return _test_oauth_client()


def setup_api_key():
    """Guide user to set up an API key."""
    print("\n--- API Key Setup ---")
    print("Get a key at: https://console.anthropic.com/settings/keys")
    print("(API usage is billed separately from your Pro/Max subscription)\n")
    key = input("Paste your Anthropic API key: ").strip()
    if not key:
        print("No key entered, aborting.")
        return False

    print("Testing key...")
    if not _test_api_key(key):
        return False

    # Offer to save to .env
    save = input("\nSave to .env file? (y/n): ").strip().lower()
    if save == "y":
        env_file = Path(__file__).parent / ".env"
        lines = []
        if env_file.is_file():
            lines = [
                l for l in env_file.read_text().splitlines()
                if not l.startswith("ANTHROPIC_API_KEY=")
            ]
        lines.append(f"ANTHROPIC_API_KEY={key}")
        env_file.write_text("\n".join(lines) + "\n")
        print(f"  ✓ Saved to {env_file}")
        print("  Note: .env is in .gitignore — won't be committed")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 50)
    print("  PsyMew — Claude Credential Setup")
    print("=" * 50)

    # Check existing
    existing = detect_existing()
    if existing:
        kind, detail = existing
        if kind in ("api_key", "api_key_dotenv"):
            display = detail[:8] + "..." + detail[-4:] if len(detail) > 16 else detail
        else:
            display = detail
        print(f"\n✓ Found existing credentials: {kind}")
        print(f"  {display}")
        print("Testing...")

        if kind == "oauth":
            if _test_oauth_client():
                _print_usage("oauth")
                return
        elif kind in ("api_key", "api_key_dotenv"):
            raw_key = detail
            if kind == "api_key_dotenv":
                os.environ["ANTHROPIC_API_KEY"] = raw_key
            if _test_api_key(raw_key):
                _print_usage("api_key")
                return

        print("\nExisting credentials didn't work. Let's set up new ones.\n")

    # Interactive menu
    print("\nChoose authentication method:\n")
    print("  [1] Claude Code OAuth  — RECOMMENDED (uses your Pro/Max subscription)")
    print("  [2] API Key            — billed separately at console.anthropic.com\n")

    choice = input("Enter choice (1/2): ").strip()
    success = False
    auth_kind = None

    if choice == "1":
        success = setup_claude_code()
        auth_kind = "oauth"
    elif choice == "2":
        success = setup_api_key()
        auth_kind = "api_key"
    else:
        print("Invalid choice.")
        sys.exit(1)

    if success:
        _print_usage(auth_kind)
    else:
        print("\n✗ Setup failed. Please try again or check your credentials.")
        sys.exit(1)


def _print_usage(auth_kind: str | None = None):
    print("\n" + "=" * 50)
    print("  Setup complete! You're ready to battle with Claude.")
    print("=" * 50)

    if auth_kind == "oauth":
        print("""
Using Claude Pro/Max OAuth — subscription rate limits apply.

To run PsyMew with Claude, just set your .env:

  DECISION_ENGINE=claude
  TUTOR_MODE=1

Then:  python start.py

Or via CLI:

  python start.py --decision-engine claude
""")
    else:
        print("""
To run PsyMew with Claude, set your .env:

  DECISION_ENGINE=claude
  TUTOR_MODE=1

Then:  python start.py

Or via CLI:

  python start.py --decision-engine claude --claude-api-key YOUR_KEY
""")


if __name__ == "__main__":
    main()
