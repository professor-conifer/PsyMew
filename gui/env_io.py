"""Round-trip reader/writer for the project's `.env` file.

Preserves blank lines, comments, and original key order when re-writing
so that hand-edited `.env` files survive a Save from the GUI.

Parsing is intentionally tolerant — it mirrors the loader inside
`config.py.configure()`: KEY=VAL, `#` is a comment, surrounding single or
double quotes on the value are stripped.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _Record:
    kind: str  # "kv" | "comment" | "blank"
    raw: str
    key: str = ""
    value: str = ""


def _parse(text: str) -> list[_Record]:
    records: list[_Record] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            records.append(_Record("blank", ""))
            continue
        if stripped.startswith("#") or "=" not in stripped:
            records.append(_Record("comment", line))
            continue
        key, _, raw_val = stripped.partition("=")
        key = key.strip()
        value = raw_val.strip().strip("'\"")
        records.append(_Record("kv", line, key=key, value=value))
    return records


def read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    result: dict[str, str] = {}
    for r in _parse(text):
        if r.kind == "kv" and r.key:
            result[r.key] = r.value
    return result


def write_env(path: Path, mapping: dict[str, str]) -> None:
    """Write `mapping` back to `path`, preserving existing layout.

    - Existing key/value lines have their value rewritten in place.
    - Lines for keys that were removed from `mapping` are stripped.
    - Brand-new keys are appended at the bottom under a "GUI additions" header.
    - Write is atomic (write to .env.tmp, then os.replace).
    """
    existing_text = path.read_text(encoding="utf-8") if path.is_file() else ""
    records = _parse(existing_text)

    seen: set[str] = set()
    out_lines: list[str] = []
    for r in records:
        if r.kind == "blank":
            out_lines.append("")
        elif r.kind == "comment":
            out_lines.append(r.raw)
        else:
            if r.key in mapping:
                out_lines.append(f"{r.key}={_quote_if_needed(mapping[r.key])}")
                seen.add(r.key)
            # else: key was removed; drop the line

    new_keys = [k for k in mapping if k not in seen]
    if new_keys:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        out_lines.append("# Added by PsyMew GUI")
        for k in new_keys:
            out_lines.append(f"{k}={_quote_if_needed(mapping[k])}")

    body = "\n".join(out_lines)
    if not body.endswith("\n"):
        body += "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".env.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _quote_if_needed(value: str) -> str:
    if value == "":
        return ""
    if any(c in value for c in (" ", "\t", "#")) or value != value.strip():
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


SENSITIVE_KEYS = {
    "PS_PASSWORD",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY",
}


def mask_value(key: str, value: str) -> str:
    if not value:
        return value
    if key in SENSITIVE_KEYS:
        if len(value) <= 6:
            return "•" * len(value)
        return f"{value[:3]}{'•' * (len(value) - 6)}{value[-3:]}"
    return value
