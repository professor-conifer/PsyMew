#!/usr/bin/env python3
"""Interactive Gemini credential setup for PsyMew.

Supports:
  - Google OAuth  (browser login — recommended for Google AI Pro subscribers)
  - API key       (GEMINI_API_KEY / GOOGLE_API_KEY)

Usage:
    python gemini_login.py
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Gemini CLI's public OAuth client credentials (from @google/gemini-cli npm package).
# These are public PKCE credentials — security is maintained by per-user refresh tokens.
# Standard practice for open-source CLI tools.
_CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
_CLIENT_SECRET = "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"

_SCOPES = [
    "https://www.googleapis.com/auth/generative-language.retriever",
    "https://www.googleapis.com/auth/cloud-platform",
    "openid",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(val: str | None) -> str | None:
    if val is None:
        return None
    v = val.strip()
    return v if v else None


def _has_any_oauth() -> Path | None:
    """Check for existing OAuth credentials (PsyMew or Gemini CLI)."""
    from fp.gemini.cache import get_psymew_oauth_path

    psymew_path = get_psymew_oauth_path()
    if psymew_path.is_file():
        return psymew_path

    gemini_path = Path.home() / ".gemini" / "oauth_creds.json"
    if gemini_path.is_file():
        return gemini_path

    return None


def _test_oauth_client() -> bool:
    """Test that Google OAuth credentials actually work."""
    try:
        from fp.gemini.client import _try_oauth
        client = _try_oauth()
        if not client:
            return False
        models = list(client.models.list())
        print(f"  ✓ OAuth credentials valid — {len(models)} models available")
        return True
    except Exception as exc:
        print(f"  ✗ OAuth test failed: {exc}")
        return False


def _test_api_key(key: str) -> bool:
    """Test that an API key works."""
    try:
        from google import genai
        client = genai.Client(api_key=key)
        models = list(client.models.list())
        print(f"  ✓ API key valid — {len(models)} models available")
        return True
    except Exception as exc:
        print(f"  ✗ API key test failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Detect existing credentials
# ---------------------------------------------------------------------------


def detect_existing():
    """Check for existing Gemini credentials. Returns (kind, detail) or None."""
    # Check OAuth creds (PsyMew's own or Gemini CLI)
    oauth_path = _has_any_oauth()
    if oauth_path:
        return "oauth", str(oauth_path)

    # Check API key in env
    api_key = _sanitize(os.environ.get("GEMINI_API_KEY")) or _sanitize(
        os.environ.get("GOOGLE_API_KEY")
    )
    if api_key:
        return "api_key", api_key

    # Check .env file in repo root
    env_file = Path(__file__).parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
                if key:
                    return "api_key_dotenv", key

    return None


# ---------------------------------------------------------------------------
# Setup flows
# ---------------------------------------------------------------------------


def setup_oauth():
    """Run a browser-based OAuth flow and save credentials to ~/.psymew/.

    Opens the user's browser to Google's consent screen, catches the callback
    on a local HTTP server, and saves the credentials.
    """
    print("\n--- Google OAuth Setup (Google AI Pro) ---")
    print("This links PsyMew to your Google AI Pro subscription.")
    print("Your credentials stay local — never sent to any server.\n")

    # Check if creds already exist and work
    existing = _has_any_oauth()
    if existing:
        print(f"Found existing OAuth credentials: {existing}")
        print("Testing...")
        if _test_oauth_client():
            return True
        else:
            print("  Credentials exist but didn't work. Re-authenticating...\n")

    # Run the OAuth flow
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("ERROR: google-auth-oauthlib is required.")
        print("Run: pip install google-auth-oauthlib")
        return False

    try:
        client_config = {
            "installed": {
                "client_id": _CLIENT_ID,
                "client_secret": _CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }
        flow = InstalledAppFlow.from_client_config(client_config, scopes=_SCOPES)

        print("Opening browser for Google sign-in...")
        print("(If the browser doesn't open, check the URL in your terminal)\n")

        creds = flow.run_local_server(
            port=0,
            prompt="consent",
            success_message="Authentication successful! You can close this tab.",
        )

        if not creds or not creds.valid:
            print("  ✗ OAuth flow did not return valid credentials")
            return False

        # Save to PsyMew's credential file
        from fp.gemini.cache import save_oauth_creds, get_psymew_oauth_path

        creds_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scope": " ".join(_SCOPES),
        }
        if creds.expiry:
            creds_data["expiry"] = creds.expiry.isoformat()

        save_oauth_creds(creds_data)
        print(f"\n  ✓ Credentials saved to {get_psymew_oauth_path()}")

        # Quick test
        print("Testing connection...")
        if _test_oauth_client():
            return True
        else:
            print("  ✗ Could not verify saved credentials")
            return False

    except Exception as exc:
        print(f"\n  ✗ OAuth flow failed: {exc}")
        print("\nAlternative: use an API key instead (option 2)")
        return False


def setup_api_key():
    """Guide user to set up an API key."""
    print("\n--- API Key Setup ---")
    print("Get a free key at: https://aistudio.google.com/apikey")
    print("(Free tier has rate limits — for heavy use, consider Google OAuth)\n")
    key = input("Paste your Gemini API key: ").strip()
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
                if not l.startswith("GEMINI_API_KEY=")
            ]
        lines.append(f"GEMINI_API_KEY={key}")
        env_file.write_text("\n".join(lines) + "\n")
        print(f"  ✓ Saved to {env_file}")
        print("  Note: .env is in .gitignore — won't be committed")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 50)
    print("  PsyMew — Gemini Credential Setup")
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
                os.environ["GEMINI_API_KEY"] = raw_key
            if _test_api_key(raw_key):
                _print_usage("api_key")
                return

        print("\nExisting credentials didn't work. Let's set up new ones.\n")

    # Interactive menu
    print("\nChoose authentication method:\n")
    print("  [1] Google OAuth  — RECOMMENDED (uses your Google AI Pro subscription)")
    print("  [2] API Key       — free key from aistudio.google.com\n")

    choice = input("Enter choice (1/2): ").strip()
    success = False
    auth_kind = None

    if choice == "1":
        success = setup_oauth()
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
    print("  Setup complete! You're ready to battle with Gemini.")
    print("=" * 50)

    if auth_kind == "oauth":
        print("""
Using Google AI Pro OAuth — higher rate limits!

To run PsyMew with Gemini, just set your .env:

  DECISION_ENGINE=gemini
  TUTOR_MODE=1

Then:  python start.py

Or via CLI:

  python start.py --decision-engine gemini
""")
    else:
        print("""
To run PsyMew with Gemini, set your .env:

  DECISION_ENGINE=gemini
  TUTOR_MODE=1

Then:  python start.py

Or via CLI:

  python start.py --decision-engine gemini --gemini-api-key YOUR_KEY
""")


if __name__ == "__main__":
    main()
