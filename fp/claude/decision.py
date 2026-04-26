"""Claude decision engine — main entry point for move selection.

find_best_move_claude() builds a GeminiBattleView, constructs tools + prompt,
calls Claude with tool use, and parses the result into Showdown decision strings.

Handles both team preview and regular turns.
Supports retry on |error|[Invalid choice].
"""

import asyncio
import logging
from typing import Optional

from fp.claude.client import get_async_client, get_model_name
from fp.gemini.errors import GeminiInvalidChoice, GeminiTimeout
from fp.gemini.prompt import build_system_prompt, build_turn_prompt, build_team_preview_prompt
from fp.claude.tools import build_tools
from fp.gemini.view import GeminiBattleView

logger = logging.getLogger(__name__)

# Max time for a single Claude call
_DECISION_TIMEOUT_SECONDS = 25.0


def _parse_action_part(args: dict, view: GeminiBattleView, slot_idx: int = 0) -> str:
    """Convert a single slot's tool call arguments to a Showdown decision string.

    Returns something like:
      "move earthquake -1 terastallize"
      "switch garchomp"
      "move protect"
    """
    action_type = args.get("action_type", "move")

    if action_type == "switch":
        target = args.get("switch_target", "")
        return f"switch {target}"

    # It's a move
    move_id = args.get("move_id", "struggle")

    parts = [f"move {move_id}"]

    # Target (doubles/triples)
    target = args.get("target")
    if target is not None:
        parts.append(str(target))

    # Gimmick
    gimmick = args.get("gimmick", "none")
    if gimmick and gimmick != "none":
        if gimmick == "terastallize":
            parts.append("terastallize")
        elif gimmick == "mega":
            parts.append("mega")
        elif gimmick == "dynamax":
            parts.append("dynamax")
        elif gimmick == "zmove":
            parts.append("zmove")

    return " ".join(parts)


def _parse_team_preview(args: dict, view: GeminiBattleView) -> str:
    """Convert choose_leads tool call to a Showdown team order string.

    Returns something like "3142" for a 4-pick from 4 team members.
    """
    lead_order = args.get("lead_order", [])

    if not lead_order:
        return "".join(str(i) for i in range(1, len(view.own_team) + 1))

    all_indices = set(range(1, len(view.own_team) + 1))
    selected = []
    for idx in lead_order:
        if isinstance(idx, (int, float)) and int(idx) in all_indices:
            selected.append(int(idx))
            all_indices.discard(int(idx))

    selected.extend(sorted(all_indices))

    return "".join(str(i) for i in selected)


def _compute_temperature(view) -> float:
    """Compute dynamic temperature based on game state."""
    if view is None:
        return 0.65

    own_alive = sum(1 for p in view.own_team if p.hp > 0)

    opp_alive = 6
    if view.snapshot and view.snapshot.opponent_active_slots:
        opp_fainted = sum(1 for o in view.snapshot.opponent_active_slots.values() if o.fainted)
        opp_alive = max(1, 6 - opp_fainted)

    # Endgame — some variance even here, avoid being read
    if own_alive <= 2 and opp_alive <= 2:
        return 0.3

    # Losing badly — high-variance gambles are correct
    if opp_alive > own_alive + 1:
        return 0.85

    # Standard — genuinely creative, not just highest-score
    return 0.65


async def _call_claude(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
    error_context: Optional[str] = None,
    temperature: float = 0.35,
) -> dict:
    """Make a single Claude call and extract the tool use result.

    Returns the parsed arguments dict from the first tool_use block.
    """
    if error_context:
        user_prompt += (
            f"\n\n⚠️ PREVIOUS ERROR: The server rejected our last choice with: "
            f'"{error_context}". Pick a different valid action.'
        )

    from config import FoulPlayConfig as _cfg
    thinking_budget = _cfg.claude_thinking_budget if hasattr(_cfg, "claude_thinking_budget") else 0

    create_kwargs: dict = dict(
        model=model,
        max_tokens=8192,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
        tools=tools,
        tool_choice={"type": "any"},
        temperature=temperature,
    )

    if thinking_budget > 0:
        create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    # Retry with backoff for transient 429/503/529 errors
    last_exc = None
    for attempt in range(3):
        try:
            response = await asyncio.wait_for(
                client.messages.create(**create_kwargs),
                timeout=_DECISION_TIMEOUT_SECONDS,
            )
            break  # success
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "503" in err_str or "529" in err_str or "overloaded" in err_str.lower():
                wait = 2 ** attempt * 3  # 3s, 6s, 12s
                logger.warning("Claude %s (attempt %d/3), retrying in %ds...", err_str[:80], attempt + 1, wait)
                last_exc = e
                await asyncio.sleep(wait)
            else:
                raise
    else:
        raise last_exc  # all retries exhausted

    # Extract tool_use block from response
    if response.content:
        for block in response.content:
            if block.type == "tool_use":
                logger.info("Claude called %s with args: %s", block.name, block.input)
                return {"name": block.name, "args": block.input}

    # No tool use found
    resp_text = ""
    try:
        for block in response.content:
            if hasattr(block, "text"):
                resp_text += block.text
    except Exception:
        resp_text = str(response)
    raise GeminiInvalidChoice(f"Claude did not return a tool call. Response: {resp_text[:200]}")


async def find_best_move_claude(battle) -> list[dict]:
    """Main entry point: pick the best move(s) using Claude.

    Uses a hybrid approach:
    - The move scorer computes scores for all options
    - If one option clearly dominates (high score, big gap to #2), use it directly
    - Only consult Claude when there's a genuine strategic dilemma

    Parameters
    ----------
    battle : Battle
        The current battle object.

    Returns
    -------
    list[dict]
        List of action dicts.
    """
    from config import FoulPlayConfig
    from fp.gemini.move_scorer import score_all_actions, get_best_action

    # Build the view
    view = GeminiBattleView.from_battle(battle)

    # Handle force-switch: active slots are empty because our Pokemon fainted
    if not view.active_slots and not view.is_team_preview:
        from fp.gemini.move_scorer import score_switch, ThreatInfo
        targets = view.legal_switch_targets
        if not targets:
            return [{"decision": "move struggle", "slot": 0}]
        scored = [score_switch(t, view, ThreatInfo()) for t in targets]
        scored.sort(key=lambda s: s.score, reverse=True)
        best_name = scored[0].pokemon_name
        logger.info("Force-switch: scored targets, picked %s (score=%.0f)", best_name, scored[0].score)
        return [{"decision": f"switch {best_name}", "slot": 0}]

    # --- Auto-play: use scorer for obvious decisions (singles only) ---
    if not view.is_team_preview and len(view.active_slots) == 1:
        scored_moves, scored_switches, threat = score_all_actions(view, 0)

        if scored_moves or scored_switches:
            best_move = scored_moves[0] if scored_moves else None
            best_switch = scored_switches[0] if scored_switches else None
            best_move_score = best_move.score if best_move else -1
            best_switch_score = best_switch.score if best_switch else -1

            if best_switch_score > best_move_score + 10:
                best_score = best_switch_score
                best_decision = f"switch {best_switch.pokemon_name}"
                runner_up = best_move_score
            else:
                best_score = best_move_score
                best_decision = f"move {best_move.move_id}" if best_move else "move struggle"
                runner_up = max(
                    best_switch_score,
                    scored_moves[1].score if len(scored_moves) > 1 else -1
                )

            gap = best_score - runner_up

            auto_play = False
            reason = ""

            if best_move and best_move.is_immune and best_switch:
                best_decision = f"switch {best_switch.pokemon_name}"
                auto_play = True
                reason = f"best move immune, switching to {best_switch.pokemon_name}"
            elif best_score >= 70:
                auto_play = True
                reason = f"strong play (score={best_score:.0f})"
            elif best_score >= 50 and gap >= 15:
                auto_play = True
                reason = f"clear winner (score={best_score:.0f}, gap={gap:.0f})"
            elif len(scored_moves) <= 1 and not scored_switches:
                auto_play = True
                reason = "only one option"

            if auto_play:
                logger.info("AUTO-PLAY [%s]: %s", reason, best_decision)
                return [{"decision": best_decision, "slot": 0}]

            logger.info(
                "CLAUDE CONSULT: top=%s(%.0f) vs runner=%.0f, gap=%.0f — asking Claude",
                best_decision, best_score, runner_up, gap,
            )

    # --- Claude decision for complex situations + team preview ---
    client = get_async_client(
        auth_mode=FoulPlayConfig.claude_auth_mode,
        api_key_override=FoulPlayConfig.claude_api_key,
    )
    model = get_model_name()

    # Build prompts (reuse Gemini prompts — they're model-agnostic)
    format_rules_text = getattr(battle, "format_rules_text", "")
    system_prompt = build_system_prompt(view.format_info, format_rules_text)

    if view.is_team_preview:
        user_prompt = build_team_preview_prompt(view)
    else:
        user_prompt = build_turn_prompt(view)

    tools = build_tools(view)
    error_context = getattr(battle, "last_server_error", None)

    temperature = _compute_temperature(view)

    try:
        result = await _call_claude(client, model, system_prompt, user_prompt, tools, error_context, temperature=temperature)
    except asyncio.TimeoutError:
        logger.error("Claude decision timed out after %.1fs", _DECISION_TIMEOUT_SECONDS)
        raise GeminiTimeout(f"Claude call timed out after {_DECISION_TIMEOUT_SECONDS}s")
    except Exception as exc:
        logger.error("Claude decision failed: %s", exc)
        raise

    if hasattr(battle, "last_server_error"):
        battle.last_server_error = None

    # Parse result
    func_name = result.get("name", "")
    args = result.get("args", {})

    if func_name == "choose_leads":
        team_order = _parse_team_preview(args, view)
        return [{"team_order": team_order}]

    elif func_name == "choose_action":
        decision_str = _parse_action_part(args, view, slot_idx=0)
        return [{"decision": decision_str, "slot": 0}]

    elif func_name == "choose_actions":
        actions = []
        for i in range(len(view.active_slots)):
            slot_key = f"slot_{i + 1}"
            slot_args = args.get(slot_key, {})
            if isinstance(slot_args, dict):
                decision_str = _parse_action_part(slot_args, view, slot_idx=i)
                actions.append({"decision": decision_str, "slot": i})
            else:
                logger.warning("Missing args for %s, defaulting to struggle", slot_key)
                actions.append({"decision": "move struggle", "slot": i})
        return actions

    else:
        logger.warning("Unknown function call: %s", func_name)
        return [{"decision": "move struggle", "slot": 0}]
