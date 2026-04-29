"""Format rule verification for DeepSeek (training-knowledge based).

DeepSeek lacks web search tools, so we ask the model to validate the
rule card from training knowledge.  On any failure — timeout, server
error, empty response — the stored card is returned with a note so
the bot can proceed.
"""

import asyncio
import logging
from typing import Optional

from fp.gemini.format_detection import FormatInfo
from fp.gemini.format_rules import get_rule_card

logger = logging.getLogger(__name__)

_VERIFY_TIMEOUT_SECONDS = 12.0


async def verify_format_rules(
    client,
    model: str,
    format_info: FormatInfo,
    rule_card_text: Optional[str] = None,
) -> str:
    """Verify rule card against DeepSeek training knowledge.

    Since DeepSeek lacks web search / grounding, this is a best-effort
    review of the stored rule card.  Returns the verified (or original)
    text for injection into the battle system prompt.

    Parameters
    ----------
    client : openai.AsyncOpenAI
        Authenticated DeepSeek client.
    model : str
        Model name (e.g. "deepseek-v4-pro").
    format_info : FormatInfo
        Detected format metadata.
    rule_card_text : str | None
        The hardcoded rule card.  If None, looked up from format_rules.

    Returns
    -------
    str
        Verified or annotated rules text.
    """
    if rule_card_text is None:
        rule_card_text = get_rule_card(format_info.gen, format_info.format_name)

    prompt = (
        f"You are verifying rules for a Pokemon Showdown battle format.\n\n"
        f"Format: {format_info.format_name}\n"
        f"Generation: {format_info.gen}\n"
        f"Gametype: {format_info.gametype}\n\n"
        f"Here is our stored rule card:\n```\n{rule_card_text}\n```\n\n"
        f"Based on your knowledge of Pokemon Showdown's rules as of early 2026, "
        f"review this card. If anything needs correction — bans, clauses, "
        f"allowed gimmicks, regulations, restricted Pokemon, move targeting "
        f"semantics — output a corrected rule summary. "
        f"If the card looks accurate or you are uncertain, output the card "
        f"as-is.\n\n"
        f"Be concise — this will be injected into a battle bot's system prompt. "
        f"Focus on: allowed gimmicks, banned Pokemon/moves/abilities, clauses, "
        f"bring/pick counts, move targeting for {format_info.gametype}, "
        f"and generation-specific mechanics."
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.1,
            ),
            timeout=_VERIFY_TIMEOUT_SECONDS,
        )

        result_text = response.choices[0].message.content
        if result_text and result_text.strip():
            logger.info("Format rules verified via DeepSeek knowledge")
            return result_text.strip()

        logger.warning("Empty response from format verification, using stored card")
        return rule_card_text

    except asyncio.TimeoutError:
        logger.warning(
            "Format verification timed out (%.1fs), using stored rule card",
            _VERIFY_TIMEOUT_SECONDS,
        )
        return (
            rule_card_text
            + "\n\n[Note: Live verification timed out — rules may be outdated]"
        )

    except Exception as exc:
        logger.warning(
            "Format verification failed: %s — using stored rule card", exc
        )
        return (
            rule_card_text
            + "\n\n[Note: Live verification failed — rules may be outdated]"
        )
