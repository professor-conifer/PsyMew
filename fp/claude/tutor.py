"""TutorSession — educational coaching mode powered by Claude.

Provides post-turn commentary, responds to opponent chat, and gives
a post-game review. Uses persistent message history via the Anthropic SDK.

Guardrails:
  - Strip leading '/' from replies (prevents Showdown commands)
  - Hard limit replies to 250 chars (single Showdown message)
  - Rate limit: 1 post-turn comment + 2 chat replies per turn
  - Ignore messages from self
  - Trim chat history every 10 turns
"""

import asyncio
import logging
from typing import Optional

from fp.claude.client import get_async_client, get_model_name

logger = logging.getLogger(__name__)

_MAX_REPLY_LENGTH = 250
_MAX_CHAT_REPLIES_PER_TURN = 2
_MAX_POST_TURN_REPLIES = 1
_HISTORY_TRIM_INTERVAL = 10
_TUTOR_TIMEOUT_SECONDS = 20.0


def _sanitize_reply(text: str) -> Optional[str]:
    """Sanitize a tutor reply for Showdown chat.

    Returns a SINGLE string (not a list) capped at _MAX_REPLY_LENGTH chars.
    We intentionally return only one message per tutor event to avoid
    multi-message spam and truncation issues.
    """
    if not text:
        return None
    # Strip any leading slashes to prevent commands
    text = text.replace("\n/", " /").lstrip("/")
    # Collapse to single line
    text = " ".join(text.split()).strip()

    if not text:
        return None

    # Hard cap at max length, break at last space to avoid mid-word cut
    if len(text) > _MAX_REPLY_LENGTH:
        text = text[:_MAX_REPLY_LENGTH]
        last_space = text.rfind(" ")
        if last_space > _MAX_REPLY_LENGTH // 2:
            text = text[:last_space]

    return text


class TutorSession:
    """Persistent Claude chat session for tutoring/coaching during battle."""

    def __init__(self, bot_username: str):
        self.bot_username = bot_username
        self._history: list[dict] = []
        self._chat_replies_this_turn = 0
        self._post_turn_replies_this_turn = 0
        self._turn_count = 0
        self._client = None
        self._model = None

    def _get_client(self):
        if self._client is None:
            from config import FoulPlayConfig
            self._client = get_async_client(
                auth_mode=FoulPlayConfig.claude_auth_mode,
                api_key_override=FoulPlayConfig.claude_api_key,
            )
            self._model = getattr(FoulPlayConfig, "claude_tutor_model", "claude-sonnet-4-6")
        return self._client

    def _build_system_prompt(self, extra: str = "") -> str:
        system = (
            "You are PsyMew Coach, a Pokemon mentor inside a battle bot on Pokemon Showdown.\n\n"

            "WHO IS WHO:\n"
            f"  - The BOT is '{self.bot_username}'. Actions labeled \"Bot's [Pokemon]\" are the bot's.\n"
            "  - The STUDENT is the human battling AGAINST the bot. Actions labeled \"Student's [Pokemon]\" are theirs.\n"
            "  - ONLY analyze the STUDENT's plays. Never critique the bot.\n"
            "  - Say 'you' for the student, 'I' or 'my' for the bot.\n\n"

            "RULES:\n"
            "  - Reply in 1-2 SHORT sentences, under 200 characters total.\n"
            "  - Name the specific Pokemon and move.\n"
            "  - Only mention Pokemon that actually appear in the summary. Do NOT invent or guess Pokemon names.\n"
            "  - If the student played well, say so briefly.\n"
            "  - If they misplayed, say what and why in one sentence.\n"
            "  - NEVER say 'I can't see', 'I need', 'the summary', or comment on missing data.\n"
            "  - NEVER give contradictory feedback.\n"
            "  - If the summary is unclear, give a general tip instead.\n"
        )
        if extra:
            system += f"\n{extra}"
        return system

    async def _ask(self, prompt: str, system_context: str = "") -> Optional[str]:
        """Send a prompt to Claude and return a single sanitized reply string."""
        client = self._get_client()

        system = self._build_system_prompt(system_context)

        # Build messages with recent history for context
        messages = []
        for entry in self._history[-12:]:
            messages.append({"role": entry["role"], "content": entry["text"]})
        messages.append({"role": "user", "content": prompt})

        # Retry with backoff for transient errors
        last_exc = None
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=self._model,
                        max_tokens=200,
                        system=system,
                        messages=messages,
                        temperature=0.7,
                    ),
                    timeout=_TUTOR_TIMEOUT_SECONDS,
                )

                reply_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        reply_text += block.text

                reply = _sanitize_reply(reply_text)
                if reply:
                    self._history.append({"role": "user", "text": prompt})
                    self._history.append({"role": "assistant", "text": reply})
                return reply

            except asyncio.TimeoutError:
                logger.debug("Tutor response timed out (attempt %d/2)", attempt + 1)
                return None
            except Exception as exc:
                err_str = str(exc)
                if ("429" in err_str or "503" in err_str or "529" in err_str
                        or "overloaded" in err_str.lower()):
                    wait = 3 * (attempt + 1)
                    logger.warning("Tutor %s (attempt %d/2), retrying in %ds...",
                                   err_str[:80], attempt + 1, wait)
                    last_exc = exc
                    await asyncio.sleep(wait)
                else:
                    logger.debug("Tutor error: %s", exc)
                    return None

        logger.warning("Tutor retries exhausted: %s", last_exc)
        return None

    def _new_turn(self):
        """Reset per-turn rate limits."""
        self._chat_replies_this_turn = 0
        self._post_turn_replies_this_turn = 0
        self._turn_count += 1

        # Trim history periodically
        if self._turn_count % _HISTORY_TRIM_INTERVAL == 0 and len(self._history) > 20:
            self._history = self._history[-20:]

    async def on_battle_start(self, format_name: str, format_rules_text: str) -> Optional[str]:
        """Called at battle start — introduce the format."""
        prompt = (
            f"A {format_name} battle is starting. "
            f"Give a brief, friendly intro (1 sentence, under 200 chars). "
            f"Let the student know you'll offer tips after each turn."
        )
        return await self._ask(prompt, system_context=f"Format: {format_rules_text[:150]}")

    async def on_incoming_chat(self, sender: str, text: str) -> Optional[str]:
        """Called when opponent sends a chat message."""
        from fp.helpers import normalize_name

        if normalize_name(sender) == normalize_name(self.bot_username):
            return None

        if self._chat_replies_this_turn >= _MAX_CHAT_REPLIES_PER_TURN:
            return None

        prompt = (
            f'The student said: "{text}". '
            f'Reply in one short, friendly sentence. If they ask about the battle, be helpful.'
        )
        reply = await self._ask(prompt)
        if reply:
            self._chat_replies_this_turn += 1
        return reply

    async def on_turn_complete(self, turn_summary: str) -> Optional[str]:
        """Called after each turn — provide coaching commentary."""
        self._new_turn()

        if self._post_turn_replies_this_turn >= _MAX_POST_TURN_REPLIES:
            return None

        prompt = (
            f"Turn summary:\n{turn_summary}\n\n"
            f"Comment on the STUDENT's play (lines starting with \"Student's\") in 1-2 sentences under 200 chars. "
            f"Only reference Pokemon and moves that appear in the summary above."
        )
        reply = await self._ask(prompt)
        if reply:
            self._post_turn_replies_this_turn += 1
        return reply

    async def on_battle_end(self, winner: Optional[str], bot_username: str) -> Optional[str]:
        """Called when the battle ends — give a post-game review."""
        from fp.helpers import normalize_name

        bot_won = winner and normalize_name(winner) == normalize_name(bot_username)

        if bot_won:
            prompt = (
                "The student lost. Give a brief encouraging post-game note (1-2 sentences, under 240 chars). "
                "Mention one key thing they can improve."
            )
        elif winner:
            prompt = (
                "The student won! Briefly congratulate them and mention one area to improve (1-2 sentences, under 240 chars)."
            )
        else:
            prompt = (
                "The battle was a tie. Note what the student could improve (1-2 sentences, under 240 chars)."
            )

        return await self._ask(prompt, system_context="Post-game message.")
