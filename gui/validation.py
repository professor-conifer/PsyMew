"""Configuration validation for the PsyMew GUI.

Inspects a flat `key -> value` mapping (the `.env`-shaped config) and
returns a list of human-readable issues. Used to gate the Start button
and to surface tooltips telling the user exactly what's missing.
"""

from __future__ import annotations

from dataclasses import dataclass

from gui.formats import is_teamless_format

VALID_ENGINES = ("mcts", "claude", "gemini", "deepseek")
VALID_BOT_MODES = ("challenge_user", "accept_challenge", "search_ladder")


@dataclass(frozen=True)
class ValidationIssue:
    key: str
    severity: str  # "error" | "warning"
    message: str


_ENGINE_KEY_MAP = {
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def validate(values: dict[str, str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not values.get("PS_USERNAME", "").strip():
        issues.append(
            ValidationIssue(
                "PS_USERNAME", "error", "Showdown username is required."
            )
        )

    engine = (values.get("DECISION_ENGINE") or "gemini").strip().lower()
    if engine not in VALID_ENGINES:
        issues.append(
            ValidationIssue(
                "DECISION_ENGINE",
                "error",
                f"Engine must be one of {', '.join(VALID_ENGINES)}.",
            )
        )
    else:
        required_key = _ENGINE_KEY_MAP.get(engine)
        if required_key and not values.get(required_key, "").strip():
            issues.append(
                ValidationIssue(
                    required_key,
                    "error",
                    f"{required_key} is required when DECISION_ENGINE={engine}.",
                )
            )

    bot_mode = (values.get("PS_BOT_MODE") or "accept_challenge").strip().lower()
    if bot_mode not in VALID_BOT_MODES:
        issues.append(
            ValidationIssue(
                "PS_BOT_MODE",
                "error",
                f"Bot mode must be one of {', '.join(VALID_BOT_MODES)}.",
            )
        )
    elif bot_mode == "challenge_user":
        if not values.get("PS_USER_TO_CHALLENGE", "").strip():
            issues.append(
                ValidationIssue(
                    "PS_USER_TO_CHALLENGE",
                    "error",
                    "User to challenge is required when bot mode is challenge_user.",
                )
            )

    fmt = (values.get("PS_FORMAT") or "").strip().lower()
    if not fmt:
        issues.append(
            ValidationIssue(
                "PS_FORMAT", "warning", "Format is empty; defaulting to gen9randombattle."
            )
        )
    elif not is_teamless_format(fmt) and not values.get("PS_TEAM_NAME", "").strip():
        issues.append(
            ValidationIssue(
                "PS_TEAM_NAME",
                "warning",
                "Non-random format selected — pick a team or the bot will fail at battle start.",
            )
        )

    return issues


def errors_only(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    return [i for i in issues if i.severity == "error"]


def format_issues_summary(issues: list[ValidationIssue]) -> str:
    if not issues:
        return "Config is ready."
    lines = []
    for issue in issues:
        marker = "✗" if issue.severity == "error" else "⚠"
        lines.append(f"{marker} {issue.message}")
    return "\n".join(lines)
