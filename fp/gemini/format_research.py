"""Live Google Search verification of format rules at battle start.

At battle start, verify_format_rules() sends the hardcoded rule card plus the
format name to Gemini with Google Search grounding enabled. The model compares
the card against current online documentation and returns a corrected summary.

Per-turn grounding is NOT handled here — instead, google_search is included
as a tool in the decision call (see tools.py / decision.py) so Gemini can
self-serve rule lookups mid-decision when uncertain.
"""

import asyncio
import logging
from typing import Optional

from google.genai import types

from fp.gemini.format_detection import FormatInfo
from fp.gemini.format_rules import get_rule_card

logger = logging.getLogger(__name__)

# Timeout for the verification search call at battle start
_VERIFY_TIMEOUT_SECONDS = 8.0
_META_TIMEOUT_SECONDS = 10.0


async def verify_format_rules(
    client,
    model: str,
    format_info: FormatInfo,
    rule_card_text: Optional[str] = None,
) -> str:
    """Verify rule card against live Google Search and return corrected rules.

    Parameters
    ----------
    client : genai.Client
        The authenticated Gemini client.
    model : str
        Model name (e.g. "gemini-2.5-pro").
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
            client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            ),
            timeout=_VERIFY_TIMEOUT_SECONDS,
        )

        if response.text:
            logger.info("Format rules verified via Google Search grounding")
            return response.text.strip()
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


async def fetch_format_meta_context(
    client,
    model: str,
    format_info: FormatInfo,
) -> str:
    """Fetch current competitive meta knowledge for this format via Google Search.

    Returns a concise threat/archetype summary (~150 words) for injection into
    the system prompt under CURRENT META CONTEXT.
    """
    format_name = format_info.format_name
    gametype = format_info.gametype

    prompt = (
        f"For Pokemon Showdown format '{format_name}' as of April 2026:\n"
        f"1. What are the top 5-8 threats every team must prepare for? "
        f"List them by name with a one-line reason each.\n"
        f"2. What are the 2-3 dominant team archetypes "
        f"(e.g., hyper offense, balance, stall, trick room, weather, tailwind)?\n"
        f"3. What key strategies or cores define the current meta?\n"
        f"4. What counter-meta plays are seeing success?\n\n"
        f"Gametype: {gametype}. "
        f"Keep the entire response under 200 words. "
        f"Be specific with Pokemon names. "
        f"Focus on actionable threat intelligence for a competitive battle AI."
    )

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            ),
            timeout=_META_TIMEOUT_SECONDS,
        )

        if response.text:
            logger.info("Format meta context fetched for %s", format_name)
            return response.text.strip()
        else:
            logger.warning("Empty meta context response for %s", format_name)
            return ""

    except asyncio.TimeoutError:
        logger.warning("Meta context fetch timed out for %s", format_name)
        return ""

    except Exception as exc:
        logger.warning("Meta context fetch failed for %s: %s", format_name, exc)
        return ""
