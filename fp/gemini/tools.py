"""Build Gemini function-calling tool schemas from the current battle view.

Every parameter uses enum constraints so Gemini cannot hallucinate
illegal moves or targets. Rebuilt fresh each turn.
"""

import logging
from typing import Optional

from google.genai import types

from fp.gemini.view import GeminiBattleView

logger = logging.getLogger(__name__)


def _build_slot_action_schema(view: GeminiBattleView, slot_idx: int) -> dict:
    """Build the JSON schema for one active slot's action."""
    slot = view.active_slots[slot_idx]

    # Legal move IDs (excluding disabled)
    legal_move_ids = [m.id for m in slot.legal_moves if not m.disabled and m.pp > 0]

    # Legal switch targets
    switch_targets = [p.name for p in view.legal_switch_targets]

    # Build action_type enum
    action_types = []
    if legal_move_ids and not slot.force_switch:
        action_types.append("move")
    if switch_targets and not slot.trapped:
        action_types.append("switch")
    if not action_types:
        # Forced to struggle or pass
        action_types = ["move"]
        legal_move_ids = ["struggle"]

    properties = {
        "action_type": {
            "type": "STRING",
            "description": "Whether to use a move or switch Pokemon",
            "enum": action_types,
        },
    }
    required = ["action_type"]

    if legal_move_ids:
        properties["move_id"] = {
            "type": "STRING",
            "description": f"The move to use (slot {slot_idx + 1}: {slot.pokemon.species})",
            "enum": legal_move_ids,
        }

    if switch_targets:
        properties["switch_target"] = {
            "type": "STRING",
            "description": "Pokemon to switch to",
            "enum": switch_targets,
        }

    # Targeting for doubles/triples
    if view.format_info and view.format_info.gametype != "singles":
        # Build a union of all possible targets across legal moves
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
                    "Positive = own slots (1=self, 2=partner). "
                    "Required for single-target moves in doubles/triples."
                ),
                "enum": sorted(all_targets),
            }

    # Gimmick flags
    gimmick_options = []
    if slot.can_terastallize:
        gimmick_options.append("terastallize")
    if slot.can_mega_evo:
        gimmick_options.append("mega")
    if slot.can_dynamax:
        gimmick_options.append("dynamax")
    if slot.can_z_move:
        gimmick_options.append("zmove")

    if gimmick_options:
        gimmick_options.append("none")
        properties["gimmick"] = {
            "type": "STRING",
            "description": "Optional gimmick to activate this turn (only one per battle)",
            "enum": gimmick_options,
        }

    return {
        "type": "OBJECT",
        "description": f"Action for active slot {slot_idx + 1} ({slot.pokemon.species})",
        "properties": properties,
        "required": required,
    }


def _build_choose_action_tool(view: GeminiBattleView) -> types.FunctionDeclaration:
    """Build the main action-selection tool."""
    if view.slot_count == 1:
        # Singles: single action object
        schema = _build_slot_action_schema(view, 0)
        return types.FunctionDeclaration(
            name="choose_action",
            description=(
                "Choose your action for this turn. "
                "Use 'move' to attack or 'switch' to swap Pokemon. "
                "Only use moves/targets from the provided enums."
            ),
            parameters=schema,
        )
    else:
        # Multi-active: array of actions
        slot_schemas = []
        for i in range(len(view.active_slots)):
            slot_schemas.append(_build_slot_action_schema(view, i))

        properties = {}
        required = []
        for i, schema in enumerate(slot_schemas):
            key = f"slot_{i + 1}"
            properties[key] = schema
            required.append(key)

        return types.FunctionDeclaration(
            name="choose_actions",
            description=(
                f"Choose actions for all {len(view.active_slots)} active Pokemon this turn. "
                "Each slot must have an action. Only use moves/targets from the provided enums."
            ),
            parameters={
                "type": "OBJECT",
                "properties": properties,
                "required": required,
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
                    "description": f"Ordered list of {view.pick_count} team indices (1-{team_size})",
                    "items": {
                        "type": "INTEGER",
                        "enum": valid_indices,
                    },
                },
            },
            "required": ["lead_order"],
        },
    )


def build_tools(view: GeminiBattleView, include_search: bool = False) -> list[types.Tool]:
    """Build the Gemini tool declarations for this turn.

    Parameters
    ----------
    view : GeminiBattleView
        Current battle state snapshot.
    include_search : bool
        Whether to include google_search. Note: Google Search cannot be
        combined with function calling in the same request, so this is
        False by default. Search grounding is used separately at battle
        start via format_research.py.

    Returns
    -------
    list[types.Tool]
        Tool declarations for Gemini function-calling.
    """
    if view.is_team_preview:
        func_decl = _build_team_preview_tool(view)
    else:
        func_decl = _build_choose_action_tool(view)

    if include_search:
        # Search-only mode (no function calling)
        return [types.Tool(google_search=types.GoogleSearch())]

    return [types.Tool(function_declarations=[func_decl])]
