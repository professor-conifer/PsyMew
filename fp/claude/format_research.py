"""Live web search verification of format rules at battle start (Claude).

At battle start, verify_format_rules() sends the hardcoded rule card plus the
format name to Claude with web_search enabled. The model compares the card
against current online documentation and returns a corrected summary.

This mirrors fp/gemini/format_research.py but uses Claude's built-in web_search
server tool instead of Google Search grounding.
"""

import asyncio
import logging
from typing import Optional

from fp.gemini.format_detection import FormatInfo
from fp.gemini.format_rules import get_rule_card

logger = logging.getLogger(__name__)

# Timeout for the verification search call at battle start
_VERIFY_TIMEOUT_SECONDS = 12.0


async def verify_format_rules(
    client,
    model: str,
    format_info: FormatInfo,
    rule_card_text: Optional[str] = None,
) -> str:
    """Verify rule card against live web search and return corrected rules.

    Parameters
    ----------
    client : anthropic.AsyncAnthropic
        The authenticated async Anthropic client.
    model : str
        Model name (e.g. "claude-sonnet-4-6").
    format_info : FormatInfo
        Detected format metadata.
    rule_card_text : str | None
        The hardcoded rule card. If None, looked up from format_rules.

    Returns
    -------
    str
        Verified/corrected rules text to embed in the system prompt.
    """
    if rule_card_text is None:
        rule_card_text = get_rule_card(format_info.gen, format_info.format_name)

    prompt = (
        f"You are verifying rules for a Pokemon Showdown battle format.\n\n"
        f"Format: {format_info.format_name}\n"
        f"Generation: {format_info.gen}\n"
        f"Gametype: {format_info.gametype}\n\n"
        f"Here is our stored rule card:\n```\n{rule_card_text}\n```\n\n"
        f"Search for the current rules of this Pokemon Showdown format online. "
        f"Compare against our stored card. If anything has changed (bans, clauses, "
        f"allowed gimmicks, regulations, restricted Pokemon, move targeting rules), "
        f"output a corrected and complete rule summary. If the card is accurate, "
        f"output it as-is with a note that it was verified.\n\n"
        f"Be concise — this will be injected into a battle bot's system prompt. "
        f"Focus on: allowed gimmicks, banned Pokemon/moves/abilities, clauses, "
        f"bring/pick counts, move targeting semantics for {format_info.gametype}, "
        f"and any generation-specific mechanics."
    )

    try:
        response = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 3,
                }],
                temperature=0.1,
            ),
            timeout=_VERIFY_TIMEOUT_SECONDS,
        )

        # Extract text from response blocks
        result_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                result_text += block.text

        if result_text.strip():
            logger.info("Format rules verified via Claude web search")
            return result_text.strip()
        else:
            logger.warning("Empty response from format verification, using stored card")
            return rule_card_text

    except asyncio.TimeoutError:
        logger.warning(
            "Format verification timed out (%.1fs), using stored rule card",
            _VERIFY_TIMEOUT_SECONDS,
        )
        return rule_card_text + "\n\n[Note: Live verification timed out — rules may be outdated]"

    except Exception as exc:
        logger.warning("Format verification failed: %s — using stored rule card", exc)
        return rule_card_text + "\n\n[Note: Live verification failed — rules may be outdated]"
