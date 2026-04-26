"""Doubles-aware read-only battle view for Gemini.

GeminiBattleView is built fresh each turn from:
  (a) battle.request_json  → own team + own active slots
  (b) msg_parser output    → opponent slots, field, conditions
  (c) battle.format_info   → format metadata

Never reads battler.active.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from data import all_move_json, pokedex
from fp.gemini.format_detection import FormatInfo
from fp.gemini.format_rules import get_move_target_semantics
from fp.gemini.msg_parser import ParsedSnapshot, parse_msg_list
from fp.helpers import normalize_name

logger = logging.getLogger(__name__)


@dataclass
class LegalMove:
    """A legal move for one active slot."""

    id: str
    name: str
    disabled: bool = False
    pp: int = 0
    max_pp: int = 0
    can_z: bool = False
    target_type: str = "normal"  # from moves.json
    base_power: int = 0
    move_type: str = "normal"
    category: str = "physical"


@dataclass
class OwnPokemon:
    """Summary of an own-side Pokemon from request_json."""

    name: str
    species: str
    hp: int
    max_hp: int
    hp_pct: float
    level: int
    status: Optional[str]
    active: bool
    index: int  # 1-based team index for Showdown /switch command
    item: str
    ability: str
    tera_type: Optional[str] = None
    types: list[str] = field(default_factory=list)
    moves: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


@dataclass
class ActiveSlotView:
    """View of one own active slot with legal moves and gimmick flags."""

    slot_index: int  # 0-based
    pokemon: OwnPokemon
    legal_moves: list[LegalMove]
    trapped: bool = False
    force_switch: bool = False
    can_mega_evo: bool = False
    can_ultra_burst: bool = False
    can_dynamax: bool = False
    can_terastallize: Optional[str] = None  # tera type string if available
    can_z_move: bool = False


@dataclass
class GeminiBattleView:
    """Complete read-only battle snapshot for Gemini decision-making."""

    format_info: FormatInfo
    turn: int
    slot_count: int

    # Own side
    active_slots: list[ActiveSlotView] = field(default_factory=list)
    own_team: list[OwnPokemon] = field(default_factory=list)

    # Opponent side (from msg_parser)
    snapshot: Optional[ParsedSnapshot] = None

    # Team preview
    is_team_preview: bool = False
    opponent_preview_team: list[str] = field(default_factory=list)
    pick_count: int = 6

    # Force switch slots (indices of slots that must switch)
    force_switch_slots: list[int] = field(default_factory=list)

    # Format rules text (verified)
    format_rules_text: str = ""

    # Current meta knowledge for this format (fetched at battle start)
    format_meta_context: str = ""

    # Readable battle history for Gemini context (capped at last 8 turns)
    battle_history: str = ""

    # Cross-turn strategic memory and opponent profiling
    strategic_context: Optional[object] = None   # StrategicContext
    opponent_profile: Optional[object] = None    # OpponentProfile

    @property
    def legal_switch_targets(self) -> list[OwnPokemon]:
        """Reserve Pokemon that are alive and not currently active."""
        active_names = {slot.pokemon.name for slot in self.active_slots}
        return [
            p for p in self.own_team
            if not p.active and p.hp > 0 and p.name not in active_names
        ]

    def legal_move_targets(self, slot_index: int, move_id: str) -> Optional[list[int]]:
        """Return valid Showdown target indices for a move, or None if no targeting needed."""
        move_data = all_move_json.get(move_id, {})
        target_type = move_data.get("target", "normal")
        target_map = get_move_target_semantics(self.format_info.gen, self.format_info.gametype)
        return target_map.get(target_type)

    def brief_summary(self) -> str:
        """Short one-line battle state for the strategic context update prompt."""
        own_alive = sum(1 for p in self.own_team if p.hp > 0)
        opp_fainted = 0
        if self.snapshot and self.snapshot.opponent_active_slots:
            opp_fainted = sum(1 for o in self.snapshot.opponent_active_slots.values() if o.fainted)
        opp_alive = max(1, 6 - opp_fainted)
        active = self.active_slots[0].pokemon if self.active_slots else None
        opp_active = None
        if self.snapshot and self.snapshot.opponent_active_slots:
            for o in self.snapshot.opponent_active_slots.values():
                if not o.fainted:
                    opp_active = o
                    break
        return (
            f"Turn {self.turn}. Us: {own_alive} alive. Opponent: ~{opp_alive} alive. "
            f"Our active: {active.species if active else '?'} "
            f"({active.hp_pct if active else '?'}% HP). "
            f"Opp active: {opp_active.species if opp_active else '?'} "
            f"({opp_active.hp_pct if opp_active else '?'}% HP)."
        )

    @classmethod
    def from_battle(cls, battle) -> "GeminiBattleView":
        """Build a fresh view from battle state.

        Parameters
        ----------
        battle : Battle
            The battle object (reads request_json, msg_list, format_info, turn).
        """
        format_info = battle.format_info
        request_json = battle.request_json

        if request_json is None:
            logger.warning("GeminiBattleView: request_json is None")
            return cls(
                format_info=format_info or FormatInfo(
                    gametype="singles", gen=9, format_name="unknown",
                    is_random=False, is_vgc=False, is_battle_factory=False,
                    has_team_preview=False, slot_count=1, pick_count=6,
                ),
                turn=getattr(battle, "turn", 0) or 0,
                slot_count=1,
            )

        # --- Parse own team ---
        own_team = []
        side_pokemon = request_json.get("side", {}).get("pokemon", [])
        for i, pkmn in enumerate(side_pokemon):
            condition = pkmn.get("condition", "100/100")
            hp_parts = condition.split()[0]
            status = condition.split()[1] if len(condition.split()) > 1 else None

            if "fnt" in condition:
                hp = 0
                max_hp = 1
            elif "/" in hp_parts:
                hp_cur, hp_max = hp_parts.split("/")
                hp = int(hp_cur)
                max_hp = int(hp_max)
            else:
                hp = int(hp_parts) if hp_parts.isdigit() else 100
                max_hp = 100

            details = pkmn.get("details", "")
            species = normalize_name(details.split(",")[0])

            # Parse level from details (e.g. "Garchomp, L78, M")
            pkmn_level = 100
            for detail_part in details.split(","):
                detail_part = detail_part.strip()
                if detail_part.startswith("L") and detail_part[1:].isdigit():
                    pkmn_level = int(detail_part[1:])
                    break

            # Look up types from pokedex
            pkmn_dex = pokedex.get(species, {})
            pkmn_types = [t.capitalize() for t in pkmn_dex.get("types", [])]

            own_pkmn = OwnPokemon(
                name=species,
                species=species,
                hp=hp,
                max_hp=max_hp,
                hp_pct=round(100.0 * hp / max_hp, 1) if max_hp > 0 else 0.0,
                level=pkmn_level,
                status=status if status != "fnt" else "fnt",
                active=pkmn.get("active", False),
                index=i + 1,  # 1-based for Showdown
                item=normalize_name(pkmn.get("item", "")),
                ability=normalize_name(pkmn.get("ability", "")),
                tera_type=pkmn.get("teraType"),
                types=pkmn_types,
                moves=[normalize_name(m) for m in pkmn.get("moves", [])],
                stats=pkmn.get("stats", {}),
            )
            own_team.append(own_pkmn)

        # --- Team preview mode ---
        is_team_preview = request_json.get("teamPreview", False)
        if is_team_preview:
            return cls(
                format_info=format_info,
                turn=getattr(battle, "turn", 0) or 0,
                slot_count=format_info.slot_count if format_info else 1,
                own_team=own_team,
                is_team_preview=True,
                opponent_preview_team=getattr(battle, "_opp_preview_names", []),
                pick_count=format_info.pick_count if format_info else len(own_team),
                format_rules_text=getattr(battle, "format_rules_text", ""),
            )

        # --- Active slots ---
        active_data = request_json.get("active", [])
        force_switch_data = request_json.get("forceSwitch", [])
        active_slots = []
        force_switch_slots = []

        for slot_idx, active in enumerate(active_data):
            # Find the corresponding team member
            active_pkmn = None
            for p in own_team:
                if p.active and p.hp > 0:
                    # Match by checking if this is the slot_idx-th active
                    active_count = sum(
                        1 for pp in own_team[:own_team.index(p) + 1] if pp.active
                    )
                    if active_count == slot_idx + 1:
                        active_pkmn = p
                        break

            if active_pkmn is None:
                # Fallback: pick first active
                for p in own_team:
                    if p.active:
                        active_pkmn = p
                        break

            if active_pkmn is None:
                continue

            # Parse legal moves
            legal_moves = []
            for move_data in active.get("moves", []):
                move_id = normalize_name(move_data.get("id", move_data.get("move", "")))
                move_json = all_move_json.get(move_id, {})
                legal_moves.append(LegalMove(
                    id=move_id,
                    name=move_data.get("move", move_id),
                    disabled=move_data.get("disabled", False),
                    pp=move_data.get("pp", 0),
                    max_pp=move_data.get("maxpp", 0),
                    can_z=bool(move_data.get("canZMove")),
                    target_type=move_json.get("target", "normal"),
                    base_power=move_json.get("basePower", 0),
                    move_type=move_json.get("type", "Normal").lower(),
                    category=move_json.get("category", "Physical").lower(),
                ))

            slot_view = ActiveSlotView(
                slot_index=slot_idx,
                pokemon=active_pkmn,
                legal_moves=legal_moves,
                trapped=active.get("trapped", False) or active.get("maybeTrapped", False),
                can_mega_evo=active.get("canMegaEvo", False),
                can_ultra_burst=active.get("canUltraBurst", False),
                can_dynamax=active.get("canDynamax", False),
                can_terastallize=active.get("canTerastallize"),
                can_z_move=bool(active.get("canZMove")),
            )

            # Check force switch
            if slot_idx < len(force_switch_data) and force_switch_data[slot_idx]:
                slot_view.force_switch = True
                force_switch_slots.append(slot_idx)

            active_slots.append(slot_view)

        # --- Parse opponent state from cumulative msg log ---
        msg_source = getattr(battle, 'gemini_msg_log', None) or battle.msg_list
        snapshot = parse_msg_list(
            msg_source,
            battle.user.name if hasattr(battle, "user") else "p1",
        )

        # --- Build readable battle history ---
        user_name = battle.user.name if hasattr(battle, "user") else "p1"
        battle_history = _build_battle_history(msg_source, user_name)

        return cls(
            format_info=format_info,
            turn=getattr(battle, "turn", 0) or 0,
            slot_count=len(active_slots) or (format_info.slot_count if format_info else 1),
            active_slots=active_slots,
            own_team=own_team,
            snapshot=snapshot,
            force_switch_slots=force_switch_slots,
            format_rules_text=getattr(battle, "format_rules_text", ""),
            format_meta_context=getattr(battle, "format_meta_context", ""),
            battle_history=battle_history,
            strategic_context=getattr(battle, "strategic_context", None),
            opponent_profile=getattr(battle, "opponent_profile", None),
        )


def _build_battle_history(msg_list: list[str], user_name: str) -> str:
    """Convert raw Showdown protocol lines into a structured battle history.

    Produces two sections:
    1. STRATEGIC SUMMARY: Score, KOs, key events (~10 lines)
    2. RECENT TURNS: Last 4 turns of detailed play-by-play

    This replaces the old raw chronological dump with higher signal density.
    """
    opp_name = "p2" if user_name == "p1" else "p1"

    def _side_label(ident: str) -> str:
        """'p1a: Sceptile' → 'Your Sceptile' or 'Opposing Sceptile'."""
        parts = ident.split(":")
        side = parts[0].strip()[:2]
        name = parts[1].strip() if len(parts) > 1 else "?"
        return f"Your {name}" if side == user_name else f"Opposing {name}"

    def _is_own(ident: str) -> bool:
        return ident.split(":")[0].strip()[:2] == user_name

    # --- First pass: extract strategic summary data ---
    own_fainted = []
    opp_fainted = []
    own_revealed = set()
    opp_revealed = set()
    key_events = []
    current_turn = 0

    for line in msg_list:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        action = parts[1].strip()

        try:
            if action == "turn" and len(parts) >= 3:
                current_turn = int(parts[2].strip())

            elif action in ("switch", "drag") and len(parts) >= 4:
                species = parts[3].strip().split(",")[0]
                if _is_own(parts[2]):
                    own_revealed.add(species)
                else:
                    opp_revealed.add(species)

            elif action == "faint" and len(parts) >= 3:
                name = parts[2].split(":")[-1].strip() if ":" in parts[2] else "?"
                if _is_own(parts[2]):
                    own_fainted.append(f"{name} (turn {current_turn})")
                else:
                    opp_fainted.append(f"{name} (turn {current_turn})")

            elif action == "-terastallize" and len(parts) >= 4:
                who = _side_label(parts[2])
                tera_type = parts[3].strip()
                key_events.append(f"Turn {current_turn}: {who} Terastallized -> {tera_type}")

            elif action == "-mega" and len(parts) >= 3:
                who = _side_label(parts[2])
                key_events.append(f"Turn {current_turn}: {who} Mega Evolved")

        except (IndexError, ValueError):
            continue

    # --- Second pass: extract recent turns (last 4) ---
    recent_turns = []
    turn_buffer = []
    turns_seen = []

    for line in msg_list:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        action = parts[1].strip()

        try:
            if action == "turn" and len(parts) >= 3:
                if turn_buffer:
                    turns_seen.append(turn_buffer)
                turn_buffer = [f"\n--- Turn {parts[2].strip()} ---"]

            elif action == "move" and len(parts) >= 4:
                who = _side_label(parts[2])
                move = parts[3].strip()
                turn_buffer.append(f"  {who} used {move}")

            elif action == "switch" and len(parts) >= 4:
                who = _side_label(parts[2])
                details = parts[3].strip().split(",")[0]
                turn_buffer.append(f"  {who} switched in -> {details}")

            elif action == "-damage" and len(parts) >= 4:
                who = _side_label(parts[2])
                hp_str = parts[3].strip().split()[0]
                turn_buffer.append(f"    {who} -> {hp_str} HP")

            elif action == "faint" and len(parts) >= 3:
                who = _side_label(parts[2])
                turn_buffer.append(f"  ** {who} fainted! **")

            elif action == "-supereffective":
                turn_buffer.append("    (Super effective!)")

            elif action == "-boost" and len(parts) >= 5:
                who = _side_label(parts[2])
                stat = parts[3].strip()
                amount = parts[4].strip()
                turn_buffer.append(f"    {who} {stat} +{amount}")

            elif action == "-status" and len(parts) >= 4:
                who = _side_label(parts[2])
                status = parts[3].strip()
                turn_buffer.append(f"    {who} got {status}")

        except (IndexError, ValueError):
            continue

    if turn_buffer:
        turns_seen.append(turn_buffer)

    # --- Build output ---
    output = []

    # Strategic summary
    own_alive_count = len(own_revealed) - len(own_fainted)
    opp_alive_count = len(opp_revealed) - len(opp_fainted)

    if own_revealed or opp_revealed:
        output.append("GAME STATE SUMMARY:")
        if own_revealed:
            output.append(f"  Your revealed: {', '.join(sorted(own_revealed))}")
        if opp_revealed:
            output.append(f"  Opponent revealed: {', '.join(sorted(opp_revealed))}")
        if own_fainted:
            output.append(f"  Your KOs lost: {', '.join(own_fainted)}")
        if opp_fainted:
            output.append(f"  Opponent KOs scored: {', '.join(opp_fainted)}")
        if key_events:
            for event in key_events[-5:]:  # last 5 key events
                output.append(f"  {event}")
        output.append("")

    # Recent turns (last 4)
    if turns_seen:
        output.append("RECENT TURNS:")
        for turn_lines in turns_seen[-4:]:
            output.extend(turn_lines)

    return "\n".join(output) if output else "(No battle history yet)"
