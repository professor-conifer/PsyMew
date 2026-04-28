"""Claude decision engine — main entry point for move selection.

find_best_move_claude() builds a GeminiBattleView, runs MCTS search in
parallel for game-theoretic data, calls Claude with a distribution-style
tool, and samples the final action from the LLM's weighted preferences.

The LLM ALWAYS makes the final decision — no auto-play thresholds
override it. The only exceptions are force-switch (fainted Pokemon) and
single-option (one legal move, no switches).
"""

import asyncio
import logging
import random
from typing import Optional

from fp.claude.client import get_async_client, get_model_name
from fp.gemini.errors import GeminiInvalidChoice, GeminiTimeout
from fp.gemini.prompt import (
    build_system_prompt,
    build_turn_prompt,
    build_team_preview_prompt,
)
from fp.claude.tools import build_tools
from fp.gemini.view import GeminiBattleView

logger = logging.getLogger(__name__)

_DECISION_TIMEOUT_SECONDS = 25.0


def _parse_distribution(
    distribution: list[dict], view: GeminiBattleView
) -> dict[str, float]:
    """Convert LLM weight output to a normalized probability distribution."""
    raw: dict[str, float] = {}

    for entry in distribution:
        if not isinstance(entry, dict):
            continue
        option = entry.get("option", "")
        weight = entry.get("weight", 0)
        if not option or not isinstance(weight, (int, float)) or weight <= 0:
            continue
        raw[option] = float(weight)

    if not raw:
        logger.warning("No valid weights in Claude distribution, using uniform")
        from fp.claude.tools import _build_all_option_strings

        all_options = _build_all_option_strings(view, 0)
        return {opt: 1.0 / len(all_options) for opt in all_options}

    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _sharpen_distribution(
    probs: dict[str, float], exponent: float
) -> dict[str, float]:
    """Sharpen probability distribution by applying an exponent.

    Higher exponent = wider gap between high and low weight options.
    exponent=1.0 = no change, exponent=2.0 = square the weights.
    """
    if exponent == 1.0 or not probs:
        return probs
    if len(probs) <= 1:
        return probs
    sharpened = {k: v ** exponent for k, v in probs.items()}
    total = sum(sharpened.values())
    if total <= 0:
        return probs
    return {k: v / total for k, v in sharpened.items()}


def _compute_sharpen_exponent(view) -> float:
    """Compute sharpening exponent based on game state."""
    if view is None:
        return 2.0
    own_alive = sum(1 for p in view.own_team if p.hp > 0)
    opp_alive = 6
    if view.snapshot and view.snapshot.opponent_active_slots:
        opp_fainted = sum(
            1 for o in view.snapshot.opponent_active_slots.values() if o.fainted
        )
        opp_alive = max(1, 6 - opp_fainted)
    if own_alive <= 2 and opp_alive <= 2:
        return 2.5
    if opp_alive > own_alive + 1:
        return 1.5
    return 2.0


def _sample_action(probs: dict[str, float]) -> str:
    """Sample an action string from the normalized probability distribution."""
    options = list(probs.keys())
    weights = [probs[opt] for opt in options]
    return random.choices(options, weights=weights, k=1)[0]


def _parse_action_from_option(
    option_str: str,
    entry: dict,
    view: GeminiBattleView,
    slot_idx: int = 0,
) -> str:
    """Convert a sampled option string + entry metadata to a Showdown decision."""
    parts = option_str.split(" ", 1)
    action_type = parts[0]

    if action_type == "switch" and len(parts) > 1:
        target = parts[1]
        return f"switch {target}"

    move_id = parts[1] if len(parts) > 1 else "struggle"
    result_parts = [f"move {move_id}"]

    target = entry.get("target")
    if target is not None:
        result_parts.append(str(target))

    gimmick = entry.get("gimmick", "none")
    if gimmick and gimmick != "none":
        result_parts.append(gimmick)

    return " ".join(result_parts)


def _lookup_entry_for_option(
    distribution: list[dict], option_str: str
) -> dict:
    """Find the distribution entry matching the sampled option string."""
    for entry in distribution:
        if entry.get("option") == option_str:
            return entry
    return {}


def _compute_temperature(view) -> float:
    """Compute dynamic temperature based on game state."""
    if view is None:
        return 0.65

    own_alive = sum(1 for p in view.own_team if p.hp > 0)

    opp_alive = 6
    if view.snapshot and view.snapshot.opponent_active_slots:
        opp_fainted = sum(
            1 for o in view.snapshot.opponent_active_slots.values() if o.fainted
        )
        opp_alive = max(1, 6 - opp_fainted)

    if own_alive <= 2 and opp_alive <= 2:
        return 0.3
    if opp_alive > own_alive + 1:
        return 0.85
    return 0.65


async def _call_claude(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
    error_context: Optional[str] = None,
    temperature: float = 0.65,
) -> dict:
    """Make a single Claude call and extract the tool use result."""
    if error_context:
        user_prompt += (
            f"\n\nATTENTION — Previous action was rejected by the server: "
            f'"{error_context}". Adjust your distribution accordingly.'
        )

    from config import FoulPlayConfig as _cfg

    thinking_budget = (
        _cfg.claude_thinking_budget
        if hasattr(_cfg, "claude_thinking_budget")
        else 0
    )

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
        create_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

    last_exc = None
    for attempt in range(3):
        try:
            response = await asyncio.wait_for(
                client.messages.create(**create_kwargs),
                timeout=_DECISION_TIMEOUT_SECONDS,
            )
            break
        except Exception as e:
            err_str = str(e)
            if any(
                tag in err_str
                for tag in ("429", "503", "529", "overloaded")
            ):
                wait = 2**attempt * 3
                logger.warning(
                    "Claude %s (attempt %d/3), retrying in %ds...",
                    err_str[:80],
                    attempt + 1,
                    wait,
                )
                last_exc = e
                await asyncio.sleep(wait)
            else:
                raise
    else:
        raise last_exc

    if response.content:
        for block in response.content:
            if block.type == "tool_use":
                logger.info(
                    "Claude called %s — distribution entries: %d",
                    block.name,
                    len(block.input.get("distribution", [])),
                )
                return {"name": block.name, "args": block.input}

    resp_text = ""
    try:
        for block in response.content:
            if hasattr(block, "text"):
                resp_text += block.text
    except Exception:
        resp_text = str(response)
    raise GeminiInvalidChoice(
        f"Claude did not return a tool call. Response: {resp_text[:200]}"
    )


async def _run_mcts_parallel(battle, loop) -> Optional[object]:
    """Run MCTS search for data in a thread pool."""
    try:
        from fp.search.main import run_mcts_for_data

        return await loop.run_in_executor(None, run_mcts_for_data, battle)
    except Exception as exc:
        logger.warning("MCTS data search failed: %s", exc)
        return None


async def find_best_move_claude(battle) -> list[dict]:
    """Pick the best action(s) using Claude with distribution-based reasoning.

    The LLM receives ALL available data — heuristic scores, threat assessment,
    battle history, strategic context, AND MCTS game-theoretic search insights.
    It outputs relative preference weights (1-10) for each viable option.
    The actual action is sampled from the normalized distribution.

    The LLM ALWAYS decides. Auto-play is eliminated — exceptions exist only
    for forced situations.
    """
    from config import FoulPlayConfig
    from fp.gemini.move_scorer import score_switch, ThreatInfo

    view = GeminiBattleView.from_battle(battle)

    # --- Forced situations ---

    if not view.active_slots and not view.is_team_preview:
        targets = view.legal_switch_targets
        if not targets:
            return [{"decision": "move struggle", "slot": 0}]
        scored = [score_switch(t, view, ThreatInfo()) for t in targets]
        scored.sort(key=lambda s: s.score, reverse=True)
        best_name = scored[0].pokemon_name
        logger.info(
            "FORCE-SWITCH: %s fainted, switching to %s (score=%.0f)",
            battle.user.active.name if battle.user.active else "?",
            best_name,
            scored[0].score,
        )
        return [{"decision": f"switch {best_name}", "slot": 0}]

    if not view.is_team_preview and len(view.active_slots) == 1:
        slot = view.active_slots[0]
        if not slot.force_switch:
            legal_moves = [
                m for m in slot.legal_moves if not m.disabled and m.pp > 0
            ]
            switch_targets = (
                [p for p in view.legal_switch_targets]
                if not slot.trapped
                else []
            )
            if len(legal_moves) == 1 and not switch_targets:
                move_id = legal_moves[0].id
                logger.info("SINGLE-OPTION: only move=%s, no switches", move_id)
                return [{"decision": f"move {move_id}", "slot": 0}]

    # --- Start MCTS in parallel ---
    loop = asyncio.get_event_loop()
    mcts_task = asyncio.ensure_future(_run_mcts_parallel(battle, loop))

    # --- Build prompts ---
    client = get_async_client(
        auth_mode=FoulPlayConfig.claude_auth_mode,
        api_key_override=FoulPlayConfig.claude_api_key,
    )
    model = get_model_name()

    format_rules_text = getattr(battle, "format_rules_text", "")
    format_meta_context = getattr(battle, "format_meta_context", "")
    system_prompt = build_system_prompt(
        view.format_info, format_rules_text, format_meta_context
    )

    if view.is_team_preview:
        user_prompt = build_team_preview_prompt(view)
    else:
        user_prompt = build_turn_prompt(view)

    # Await MCTS data and inject into prompt
    mcts_data = None
    try:
        mcts_data = await asyncio.wait_for(mcts_task, timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("MCTS data search timed out, proceeding without it")
    except Exception as exc:
        logger.warning("MCTS data search error: %s", exc)

    if mcts_data is not None and not view.is_team_preview:
        mcts_block = mcts_data.to_prompt_block()
        if mcts_block:
            user_prompt += "\n\n" + mcts_block

    tools = build_tools(view)
    error_context = getattr(battle, "last_server_error", None)
    temperature = _compute_temperature(view)

    # --- Call LLM ---
    try:
        result = await _call_claude(
            client, model, system_prompt, user_prompt, tools,
            error_context, temperature=temperature,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Claude decision timed out after %.1fs", _DECISION_TIMEOUT_SECONDS
        )
        raise GeminiTimeout(
            f"Claude call timed out after {_DECISION_TIMEOUT_SECONDS}s"
        )
    except Exception as exc:
        logger.error("Claude decision failed: %s", exc)
        raise

    if hasattr(battle, "last_server_error"):
        battle.last_server_error = None

    func_name = result.get("name", "")
    args = result.get("args", {})

    # --- Parse result ---
    if func_name == "choose_leads":
        team_order = _parse_team_preview(args, view)
        return [{"team_order": team_order}]

    elif func_name == "choose_action_distribution":
        distribution = args.get("distribution", [])
        probs = _parse_distribution(distribution, view)
        sharpen = _compute_sharpen_exponent(view)
        sharp_probs = _sharpen_distribution(probs, sharpen)
        sampled_option = _sample_action(sharp_probs)
        entry = _lookup_entry_for_option(distribution, sampled_option)
        decision_str = _parse_action_from_option(sampled_option, entry, view, 0)

        logger.info(
            "Claude Distribution (%d options, sharpen=%.1f): sampled=%s, raw=%s",
            len(probs),
            sharpen,
            sampled_option,
            ", ".join(f"{o}={w:.2f}" for o, w in probs.items()),
        )

        return [{"decision": decision_str, "slot": 0}]

    elif func_name == "choose_actions_distribution":
        actions = []
        for i in range(len(view.active_slots)):
            slot_key = f"slot_{i + 1}"
            slot_dist = args.get(slot_key, [])
            if isinstance(slot_dist, list) and slot_dist:
                probs = _parse_distribution(slot_dist, view)
                sharpen = _compute_sharpen_exponent(view)
                sharp_probs = _sharpen_distribution(probs, sharpen)
                sampled_option = _sample_action(sharp_probs)
                entry = _lookup_entry_for_option(slot_dist, sampled_option)
                decision_str = _parse_action_from_option(
                    sampled_option, entry, view, i
                )
                actions.append({"decision": decision_str, "slot": i})
            else:
                logger.warning(
                    "Missing distribution for %s, defaulting to struggle", slot_key
                )
                actions.append({"decision": "move struggle", "slot": i})
        return actions

    else:
        logger.warning("Unknown function call: %s", func_name)
        return [{"decision": "move struggle", "slot": 0}]


def _parse_team_preview(args: dict, view: GeminiBattleView) -> str:
    """Convert choose_leads tool call to a Showdown team order string."""
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
