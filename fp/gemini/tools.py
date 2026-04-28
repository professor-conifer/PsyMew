"""Build Gemini function-calling tool schemas from the current battle view.

The primary tool is choose_action_distribution, which lets the LLM
output relative preference weights (1-10) across multiple moves and
switches. Weights are normalized into a probability distribution and
the actual move is sampled from it — making the bot unpredictable
without sacrificing strategic quality.

Rebuilt fresh each turn so enum values always match the current
battle state.
"""

import logging

from google.genai import types

from fp.gemini.view import GeminiBattleView

logger = logging.getLogger(__name__)


def _build_all_option_strings(view: GeminiBattleView, slot_idx: int) -> list[str]:
    """Build the complete list of legal option strings for one active slot.

    Returns strings like "move earthquake", "move protect",
    "switch garchomp" — each becomes an enum value the LLM can reference.
    """
    options: list[str] = []
    slot = view.active_slots[slot_idx]

    if not slot.force_switch:
        for move in slot.legal_moves:
            if not move.disabled and move.pp > 0:
                options.append(f"move {move.id}")

    if not slot.trapped:
        for p in view.legal_switch_targets:
            options.append(f"switch {p.name}")

    # If nothing is legal, provide struggle as the only option
    if not options:
        options.append("move struggle")

    return options


def _build_distribution_item_schema(
    view: GeminiBattleView, slot_idx: int
) -> dict:
    """Build the JSON schema for one distribution entry."""
    option_strings = _build_all_option_strings(view, slot_idx)
    slot = view.active_slots[slot_idx]

    properties: dict = {
        "option": {
            "type": "STRING",
            "description": "The move or switch option (e.g. 'move earthquake', 'switch garchomp')",
            "enum": option_strings,
        },
        "weight": {
            "type": "INTEGER",
            "description": (
                "Relative strategic preference for this option (1-10). "
                "Higher = stronger preference. The actual move is sampled "
                "from the normalized distribution. Spread weights across "
                "genuinely viable alternatives to remain unpredictable."
            ),
        },
    }

    # Gimmick flag per option
    gimmick_options: list[str] = ["none"]
    if slot.can_terastallize:
        gimmick_options.append("terastallize")
    if slot.can_mega_evo:
        gimmick_options.append("mega")
    if slot.can_dynamax:
        gimmick_options.append("dynamax")
    if slot.can_z_move:
        gimmick_options.append("zmove")

    if len(gimmick_options) > 1:
        properties["gimmick"] = {
            "type": "STRING",
            "description": "Optional gimmick to activate with this option",
            "enum": gimmick_options,
        }

    # Target selection for doubles/triples
    if view.format_info and view.format_info.gametype not in ("singles",):
        all_targets: set[int] = set()
        for move in slot.legal_moves:
            if not move.disabled and move.pp > 0:
                targets = view.legal_move_targets(slot_idx, move.id)
                if targets:
                    all_targets.update(targets)

        if all_targets:
            properties["target"] = {
                "type": "INTEGER",
                "description": (
                    "Target index for the move. "
                    "Negative = opponent slots (-1=left, -2=right), "
                    "Positive = own slots (1=self, 2=partner)."
                ),
                "enum": sorted(all_targets),
            }

    return {
        "type": "OBJECT",
        "description": "One option with its relative preference weight",
        "properties": properties,
        "required": ["option", "weight"],
    }


def _build_distribution_tool(
    view: GeminiBattleView,
) -> types.FunctionDeclaration:
    """Build the distribution-style action selection tool."""
    if view.slot_count == 1:
        item_schema = _build_distribution_item_schema(view, 0)
        return types.FunctionDeclaration(
            name="choose_action_distribution",
            description=(
                "Output your relative strategic preferences across legal moves "
                "and switches as weights (1-10). Higher weight = stronger "
                "preference. Weights are normalized into probabilities and "
                "the actual action is sampled from the distribution. "
                "This mixed-strategy approach makes you unpredictable. "
                "Include ALL viable options, not just your top pick. "
                "Only omit options you've ruled out entirely."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "distribution": {
                        "type": "ARRAY",
                        "description": (
                            "List of options with preference weights. "
                            "Unlisted options receive weight 0. "
                            "Weights are relative — they will be normalized."
                        ),
                        "items": item_schema,
                    },
                },
                "required": ["distribution"],
            },
        )
    else:
        # Multi-slot (doubles/triples): per-slot distribution arrays
        slot_properties = {}
        slot_required = []
        for i in range(len(view.active_slots)):
            key = f"slot_{i + 1}"
            item_schema = _build_distribution_item_schema(view, i)
            slot_properties[key] = {
                "type": "ARRAY",
                "description": (
                    f"Distribution for slot {i + 1} "
                    f"({view.active_slots[i].pokemon.species})"
                ),
                "items": item_schema,
            }
            slot_required.append(key)

        return types.FunctionDeclaration(
            name="choose_actions_distribution",
            description=(
                f"Output relative strategic preferences for each of "
                f"{len(view.active_slots)} active Pokemon. Each slot "
                f"gets its own distribution array with weights (1-10)."
            ),
            parameters={
                "type": "OBJECT",
                "properties": slot_properties,
                "required": slot_required,
            },
        )


def _build_team_preview_tool(view: GeminiBattleView) -> types.FunctionDeclaration:
    """Build the team preview lead-selection tool."""
    team_size = len(view.own_team)
    valid_indices = list(range(1, team_size + 1))

    return types.FunctionDeclaration(
        name="choose_leads",
        description=(
            f"Choose your lead Pokemon for team preview. "
            f"Select {view.pick_count} Pokemon by their team indices. "
            f"The first Pokemon listed will be your lead."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "lead_order": {
                    "type": "ARRAY",
                    "description": (
                        f"Ordered list of {view.pick_count} team indices "
                        f"(1-{team_size})"
                    ),
                    "items": {
                        "type": "INTEGER",
                        "enum": valid_indices,
                    },
                },
            },
            "required": ["lead_order"],
        },
    )


def build_tools(
    view: GeminiBattleView, include_search: bool = False
) -> list[types.Tool]:
    """Build the Gemini tool declarations for this turn.

    Parameters
    ----------
    view : GeminiBattleView
        Current battle state snapshot.
    include_search : bool
        Whether to include google_search. Cannot be combined with
        function calling in the same request.

    Returns
    -------
    list[types.Tool]
        Tool declarations for Gemini function-calling.
    """
    if view.is_team_preview:
        func_decl = _build_team_preview_tool(view)
    else:
        func_decl = _build_distribution_tool(view)

    if include_search:
        return [types.Tool(google_search=types.GoogleSearch())]

    return [types.Tool(function_declarations=[func_decl])]
