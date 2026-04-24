"""Gemini decision engine — main entry point for move selection.

find_best_move_gemini() builds a GeminiBattleView, constructs tools + prompt,
calls Gemini with function-calling, and parses the result into Showdown
decision strings.

Handles both team preview and regular turns.
Supports retry on |error|[Invalid choice].
"""

import asyncio
import logging
from typing import Optional

from google.genai import types

from fp.gemini.client import get_client, get_model_name
from fp.gemini.errors import GeminiInvalidChoice, GeminiTimeout
from fp.gemini.prompt import build_system_prompt, build_turn_prompt, build_team_preview_prompt
from fp.gemini.tools import build_tools
from fp.gemini.view import GeminiBattleView

logger = logging.getLogger(__name__)

# Max time for a single Gemini call
_DECISION_TIMEOUT_SECONDS = 15.0


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
        # Fallback: default order
        return "".join(str(i) for i in range(1, len(view.own_team) + 1))

    # Ensure all team indices are covered
    all_indices = set(range(1, len(view.own_team) + 1))
    selected = []
    for idx in lead_order:
        if isinstance(idx, (int, float)) and int(idx) in all_indices:
            selected.append(int(idx))
            all_indices.discard(int(idx))

    # Append any remaining indices
    selected.extend(sorted(all_indices))

    return "".join(str(i) for i in selected)


def _compute_temperature(view) -> float:
    """Compute dynamic temperature based on game state.

    - Endgame (<=2 alive per side): 0.2 (maximize precision)
    - Standard play: 0.35
    - Losing badly (down 2+ Pokemon): 0.5 (encourage creative plays)
    """
    if view is None:
        return 0.35

    own_alive = sum(1 for p in view.own_team if p.hp > 0)

    # Estimate opponent alive
    opp_alive = 6
    if view.snapshot and view.snapshot.opponent_active_slots:
        opp_fainted = sum(1 for o in view.snapshot.opponent_active_slots.values() if o.fainted)
        opp_alive = max(1, 6 - opp_fainted)

    # Endgame: both sides low
    if own_alive <= 2 and opp_alive <= 2:
        return 0.2

    # Losing badly
    if opp_alive > own_alive + 1:
        return 0.5

    return 0.35


async def _call_gemini(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[types.Tool],
    error_context: Optional[str] = None,
    temperature: float = 0.35,
) -> dict:
    """Make a single Gemini call and extract the function call result.

    Returns the parsed arguments dict from the first function call.
    """
    messages = []
    if error_context:
        user_prompt += (
            f"\n\n⚠️ PREVIOUS ERROR: The server rejected our last choice with: "
            f'"{error_context}". Pick a different valid action.'
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools,
        temperature=temperature,
        max_output_tokens=2048,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
            )
        ),
    )

    # Retry with backoff for transient 429/503 errors
    last_exc = None
    for attempt in range(3):
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=config,
                ),
                timeout=_DECISION_TIMEOUT_SECONDS,
            )
            break  # success
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str or "UNAVAILABLE" in err_str:
                wait = 2 ** attempt * 3  # 3s, 6s, 12s
                logger.warning("Gemini %s (attempt %d/3), retrying in %ds...", err_str[:80], attempt + 1, wait)
                last_exc = e
                await asyncio.sleep(wait)
            else:
                raise
    else:
        raise last_exc  # all retries exhausted

    # Extract function call from response
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                fc = part.function_call
                logger.info("Gemini called %s with args: %s", fc.name, dict(fc.args))
                return {"name": fc.name, "args": dict(fc.args)}

    # No function call found - raise to trigger MCTS fallback
    resp_text = ""
    try:
        resp_text = response.text or ""
    except Exception:
        resp_text = str(response)
    raise GeminiInvalidChoice(f"Gemini did not return a function call. Response: {resp_text[:200]}")


async def find_best_move_gemini(battle) -> list[dict]:
    """Main entry point: pick the best move(s) using Gemini.

    Uses a hybrid approach:
    - The move scorer computes scores for all options
    - If one option clearly dominates (high score, big gap to #2), use it directly
    - Only consult Gemini when there's a genuine strategic dilemma

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

    # Handle force-switch: use scorer to pick best switch target
    if not view.active_slots and not view.is_team_preview:
        best = get_best_action(view)
        logger.info("Force-switch: scorer picked %s", best)
        return [{"decision": best, "slot": 0}]

    # --- Auto-play: use scorer for obvious decisions (singles only) ---
    if not view.is_team_preview and len(view.active_slots) == 1:
        scored_moves, scored_switches, threat = score_all_actions(view, 0)

        if scored_moves or scored_switches:
            # Find the best overall action
            best_move = scored_moves[0] if scored_moves else None
            best_switch = scored_switches[0] if scored_switches else None
            best_move_score = best_move.score if best_move else -1
            best_switch_score = best_switch.score if best_switch else -1

            # Determine best action and runner-up
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

            # Auto-play when the answer is obvious:
            # - Best score >= 50 (decent move) AND gap >= 15 (clearly better than alternatives)
            # - OR best score >= 70 (strong move regardless)
            # - OR best move is immune and a switch is available
            # - OR only one option exists
            auto_play = False
            reason = ""

            if best_move and best_move.is_immune and best_switch:
                # Our best move is immune — must switch
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

            # Close call — log what we're sending to Gemini
            logger.info(
                "GEMINI CONSULT: top=%s(%.0f) vs runner=%.0f, gap=%.0f — asking Gemini",
                best_decision, best_score, runner_up, gap,
            )

    # --- Gemini decision for complex situations + team preview ---
    client = get_client(
        auth_mode=FoulPlayConfig.gemini_auth_mode,
        api_key_override=FoulPlayConfig.gemini_api_key,
    )
    model = get_model_name()

    # Build prompts
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
        result = await _call_gemini(client, model, system_prompt, user_prompt, tools, error_context, temperature=temperature)
    except asyncio.TimeoutError:
        logger.error("Gemini decision timed out after %.1fs", _DECISION_TIMEOUT_SECONDS)
        raise GeminiTimeout(f"Gemini call timed out after {_DECISION_TIMEOUT_SECONDS}s")
    except Exception as exc:
        logger.error("Gemini decision failed: %s", exc)
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
