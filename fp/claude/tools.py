"""Build Anthropic tool schemas from the current battle view.

Outputs distribution-style tools (choose_action_distribution) that let
Claude assign relative preference weights (1-10) across multiple moves
and switches. Weights are normalized and the actual move is sampled —
producing unpredictable mixed strategies.

Converts the same view-aware logic used by Gemini into Anthropic's
JSON Schema-based input_schema format with strict mode.
"""

import logging

from fp.gemini.view import GeminiBattleView

logger = logging.getLogger(__name__)


def _build_all_option_strings(view: GeminiBattleView, slot_idx: int) -> list[str]:
    """Build the complete list of legal option strings for one active slot."""
    options: list[str] = []
    slot = view.active_slots[slot_idx]

    if not slot.force_switch:
        for move in slot.legal_moves:
            if not move.disabled and move.pp > 0:
                options.append(f"move {move.id}")

    if not slot.trapped:
        for p in view.legal_switch_targets:
            options.append(f"switch {p.name}")

    if not options:
        options.append("move struggle")

    return options


def _build_distribution_item_schema(
    view: GeminiBattleView, slot_idx: int
) -> dict:
    """Build the JSON Schema for one distribution entry."""
    option_strings = _build_all_option_strings(view, slot_idx)
    slot = view.active_slots[slot_idx]

    properties: dict = {
        "option": {
            "type": "string",
            "description": "The move or switch option (e.g. 'move earthquake', 'switch garchomp')",
            "enum": option_strings,
        },
        "weight": {
            "type": "integer",
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
            "type": "string",
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
                "type": "integer",
                "description": (
                    "Target index for the move. "
                    "Negative = opponent slots (-1=left, -2=right), "
                    "Positive = own slots (1=self, 2=partner)."
                ),
                "enum": sorted(all_targets),
            }

    return {
        "type": "object",
        "properties": properties,
        "required": ["option", "weight"],
    }


def _build_distribution_tool(view: GeminiBattleView) -> dict:
    """Build the distribution-style action selection tool in Anthropic format."""
    if view.slot_count == 1:
        item_schema = _build_distribution_item_schema(view, 0)
        return {
            "name": "choose_action_distribution",
            "description": (
                "Output your relative strategic preferences across legal moves "
                "and switches as weights (1-10). Higher weight = stronger "
                "preference. Weights are normalized into probabilities and "
                "the actual action is sampled from the distribution. "
                "This mixed-strategy approach makes you unpredictable. "
                "Include ALL viable options, not just your top pick. "
                "Only omit options you've ruled out entirely."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "distribution": {
                        "type": "array",
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
            "strict": True,
        }
    else:
        # Multi-slot (doubles/triples): per-slot distribution arrays
        slot_properties = {}
        slot_required = []
        for i in range(len(view.active_slots)):
            key = f"slot_{i + 1}"
            item_schema = _build_distribution_item_schema(view, i)
            slot_properties[key] = {
                "type": "array",
                "description": (
                    f"Distribution for slot {i + 1} "
                    f"({view.active_slots[i].pokemon.species})"
                ),
                "items": item_schema,
            }
            slot_required.append(key)

        return {
            "name": "choose_actions_distribution",
            "description": (
                f"Output relative strategic preferences for each of "
                f"{len(view.active_slots)} active Pokemon. Each slot "
                f"gets its own distribution array with weights (1-10)."
            ),
            "input_schema": {
                "type": "object",
                "properties": slot_properties,
                "required": slot_required,
            },
            "strict": True,
        }


def _build_team_preview_tool(view: GeminiBattleView) -> dict:
    """Build the team preview lead-selection tool in Anthropic format."""
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
                    "description": (
                        f"Ordered list of {view.pick_count} team indices "
                        f"(1-{team_size})"
                    ),
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
        return [_build_distribution_tool(view)]
