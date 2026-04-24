"""Build Anthropic tool schemas from the current battle view.

Converts the same battle-state-aware tool definitions used by Gemini
into Anthropic's tool format (JSON Schema-based input_schema).
Rebuilt fresh each turn, with strict enum constraints so Claude
cannot hallucinate illegal moves or targets.
"""

import logging
from typing import Optional

from fp.gemini.view import GeminiBattleView

logger = logging.getLogger(__name__)


def _build_slot_action_schema(view: GeminiBattleView, slot_idx: int) -> dict:
    """Build the JSON schema properties for one active slot's action."""
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
        action_types = ["move"]
        legal_move_ids = ["struggle"]

    properties = {
        "action_type": {
            "type": "string",
            "description": "Whether to use a move or switch Pokemon",
            "enum": action_types,
        },
    }
    required = ["action_type"]

    if legal_move_ids:
        properties["move_id"] = {
            "type": "string",
            "description": f"The move to use (slot {slot_idx + 1}: {slot.pokemon.species})",
            "enum": legal_move_ids,
        }

    if switch_targets:
        properties["switch_target"] = {
            "type": "string",
            "description": "Pokemon to switch to",
            "enum": switch_targets,
        }

    # Targeting for doubles/triples
    if view.format_info and view.format_info.gametype != "singles":
        all_targets: set[int] = set()
        for move in slot.legal_moves:
            if not move.disabled and move.pp > 0:
                targets = view.legal_move_targets(slot_idx, move.id)
                if targets:
                    all_targets.update(targets)

        if all_targets:
            properties["target"] = {
                "type": "integer",
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
            "type": "string",
            "description": "Optional gimmick to activate this turn (only one per battle)",
            "enum": gimmick_options,
        }

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _build_choose_action_tool(view: GeminiBattleView) -> dict:
    """Build the main action-selection tool for Anthropic format."""
    if view.slot_count == 1:
        schema = _build_slot_action_schema(view, 0)
        return {
            "name": "choose_action",
            "description": (
                "Choose your action for this turn. "
                "Use 'move' to attack or 'switch' to swap Pokemon. "
                "Only use moves/targets from the provided enums."
            ),
            "input_schema": schema,
            "strict": True,
        }
    else:
        properties = {}
        required = []
        for i in range(len(view.active_slots)):
            key = f"slot_{i + 1}"
            properties[key] = _build_slot_action_schema(view, i)
            required.append(key)

        return {
            "name": "choose_actions",
            "description": (
                f"Choose actions for all {len(view.active_slots)} active Pokemon this turn. "
                "Each slot must have an action. Only use moves/targets from the provided enums."
            ),
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "strict": True,
        }


def _build_team_preview_tool(view: GeminiBattleView) -> dict:
    """Build the team preview lead-selection tool for Anthropic format."""
    team_size = len(view.own_team)
    valid_indices = list(range(1, team_size + 1))

    return {
        "name": "choose_leads",
        "description": (
            f"Choose your lead Pokemon for team preview. "
            f"Select {view.pick_count} Pokemon by their team indices. "
            f"The first Pokemon listed will be your lead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lead_order": {
                    "type": "array",
                    "description": f"Ordered list of {view.pick_count} team indices (1-{team_size})",
                    "items": {
                        "type": "integer",
                        "enum": valid_indices,
                    },
                },
            },
            "required": ["lead_order"],
        },
        "strict": True,
    }


def build_tools(view: GeminiBattleView) -> list[dict]:
    """Build the Anthropic tool declarations for this turn.

    Parameters
    ----------
    view : GeminiBattleView
        Current battle state snapshot.

    Returns
    -------
    list[dict]
        Tool declarations in Anthropic format.
    """
    if view.is_team_preview:
        return [_build_team_preview_tool(view)]
    else:
        return [_build_choose_action_tool(view)]
