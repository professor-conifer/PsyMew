"""Cross-turn strategic memory for the AI battle engine.

StrategicContext persists across turns so the AI can execute multi-turn plans,
track win conditions, and learn opponent tendencies rather than starting cold
each decision.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategicContext:
    """Rolling strategic narrative updated after each turn."""

    our_win_condition: str = ""
    their_win_condition: str = ""
    active_strategy: str = ""
    key_threats_unresolved: list[str] = field(default_factory=list)
    opponent_tendencies: str = ""
    turn_updated: int = 0

    def to_prompt_block(self) -> str:
        """Render as a prompt section injected at the top of each turn prompt."""
        if not self.our_win_condition:
            return ""
        lines = ["STRATEGIC MEMORY (your analysis from previous turns):"]
        if self.our_win_condition:
            lines.append(f"  Our win condition: {self.our_win_condition}")
        if self.their_win_condition:
            lines.append(f"  Their win condition: {self.their_win_condition}")
        if self.active_strategy:
            lines.append(f"  Current plan: {self.active_strategy}")
        if self.key_threats_unresolved:
            lines.append(f"  Unresolved threats: {', '.join(self.key_threats_unresolved)}")
        if self.opponent_tendencies:
            lines.append(f"  Opponent tendencies: {self.opponent_tendencies}")
        return "\n".join(lines)

    def update_from_json(self, data: dict, turn: int) -> None:
        """Merge a parsed JSON response into this context."""
        self.our_win_condition = data.get("our_win_condition", self.our_win_condition)
        self.their_win_condition = data.get("their_win_condition", self.their_win_condition)
        self.active_strategy = data.get("active_strategy", self.active_strategy)
        threats = data.get("key_threats_unresolved")
        if isinstance(threats, list):
            self.key_threats_unresolved = [str(t) for t in threats[:5]]
        self.opponent_tendencies = data.get("opponent_tendencies", self.opponent_tendencies)
        self.turn_updated = turn


_CONTEXT_UPDATE_PROMPT = """\
You just chose: {decision}

Current battle state: {summary}

Update the strategic context in JSON (all fields required):
{{
  "our_win_condition": "which Pokemon is our win condition and what it needs to succeed",
  "their_win_condition": "their clearest path to winning and what we must deny",
  "active_strategy": "our 2-3 turn plan going forward",
  "key_threats_unresolved": ["threat1", "threat2"],
  "opponent_tendencies": "observed switching/protect/play patterns"
}}

Respond with ONLY the JSON object, nothing else."""


async def update_strategic_context_async(
    battle,
    view,
    decision_made: str,
    client,
    model: str,
) -> None:
    """Fire-and-forget: ask the model to update the strategic narrative.

    Never blocks the main turn decision. On any failure, logs a warning and
    leaves the context unchanged.
    """
    try:
        from google.genai import types as gtypes

        summary = view.brief_summary() if hasattr(view, "brief_summary") else "unknown"
        prompt = _CONTEXT_UPDATE_PROMPT.format(decision=decision_made, summary=summary)

        config = gtypes.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=512,
        )

        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            ),
            timeout=8.0,
        )

        raw = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    raw += part.text

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        turn = view.turn or 0
        battle.strategic_context.update_from_json(data, turn)
        logger.debug("Strategic context updated on turn %d", turn)

    except asyncio.TimeoutError:
        logger.warning("Strategic context update timed out — keeping previous context")
    except Exception as exc:
        logger.warning("Strategic context update failed: %s", exc)
