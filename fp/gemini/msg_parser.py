"""Read-only parser over battle.msg_list for Gemini's battle view.

Produces a ParsedSnapshot from raw Pokemon Showdown protocol lines.
Never reads battler.active — only parses the accumulated msg_list.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from data import pokedex
from fp.helpers import normalize_name

logger = logging.getLogger(__name__)


@dataclass
class OppSlotInfo:
    """Opponent's active slot state, parsed from protocol messages."""

    species: str = "unknown"
    types: list[str] = field(default_factory=list)
    hp_pct: float = 100.0
    level: int = 100
    status: Optional[str] = None  # brn, par, slp, psn, tox, frz, fnt
    boosts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    revealed_moves: list[str] = field(default_factory=list)
    revealed_item: Optional[str] = None
    revealed_ability: Optional[str] = None
    tera_type: Optional[str] = None
    is_dynamaxed: bool = False
    has_substitute: bool = False
    fainted: bool = False


@dataclass
class FieldState:
    """Global field conditions."""

    weather: Optional[str] = None
    weather_turns: int = -1
    terrain: Optional[str] = None
    terrain_turns: int = 0
    trick_room: bool = False
    trick_room_turns: int = 0
    gravity: bool = False


@dataclass
class ParsedSnapshot:
    """Complete parsed state from msg_list for Gemini consumption."""

    opponent_active_slots: dict[int, OppSlotInfo] = field(default_factory=dict)
    field_state: FieldState = field(default_factory=FieldState)
    own_side_conditions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    opp_side_conditions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    own_gimmicks_used: set[str] = field(default_factory=set)
    opp_gimmicks_used: set[str] = field(default_factory=set)
    commander_active: bool = False


def _parse_pokemon_ident(ident: str) -> tuple[str, int]:
    """Parse 'p2a: Garchomp' → ('p2', 0) or 'p2b: Toxapex' → ('p2', 1)."""
    side_slot = ident.split(":")[0].strip()
    side = side_slot[:2]  # p1 or p2
    slot_char = side_slot[2] if len(side_slot) > 2 else "a"
    slot_idx = ord(slot_char) - ord("a")
    return side, slot_idx


def _parse_hp_status(condition: str) -> tuple[float, Optional[str]]:
    """Parse '157/300 brn' → (52.3, 'brn') or '0 fnt' → (0.0, 'fnt')."""
    parts = condition.strip().split()
    hp_str = parts[0]
    status = parts[1] if len(parts) > 1 else None

    if hp_str == "0" or "fnt" in condition:
        return 0.0, "fnt"

    if "/" in hp_str:
        current, total = hp_str.split("/")
        try:
            return round(100.0 * int(current) / int(total), 1), status
        except (ValueError, ZeroDivisionError):
            return 100.0, status

    # Percentage format (e.g. "75/100")
    try:
        return float(hp_str), status
    except ValueError:
        return 100.0, status


def parse_msg_list(msg_list: list[str], user_name: str) -> ParsedSnapshot:
    """Parse accumulated protocol messages into a ParsedSnapshot.

    Parameters
    ----------
    msg_list : list[str]
        Raw protocol lines from battle.msg_list.
    user_name : str
        The bot's player identifier (e.g. "p1").
    """
    snap = ParsedSnapshot()
    opp_name = "p2" if user_name == "p1" else "p1"

    def _get_opp_slot(slot_idx: int) -> OppSlotInfo:
        if slot_idx not in snap.opponent_active_slots:
            snap.opponent_active_slots[slot_idx] = OppSlotInfo()
        return snap.opponent_active_slots[slot_idx]

    for line in msg_list:
        split = line.split("|")
        if len(split) < 2:
            continue

        action = split[1].strip()

        try:
            # --- Switch / Drag ---
            if action in ("switch", "drag") and len(split) >= 5:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    # Species from details: "Garchomp, L78, M"
                    details = split[3].strip()
                    info.species = normalize_name(details.split(",")[0])
                    # Parse level from details (e.g. "L78")
                    for part in details.split(","):
                        part = part.strip()
                        if part.startswith("L") and part[1:].isdigit():
                            info.level = int(part[1:])
                            break
                    # Look up types from pokedex
                    dex_entry = pokedex.get(info.species, {})
                    info.types = [t.capitalize() for t in dex_entry.get("types", [])]
                    hp_pct, status = _parse_hp_status(split[4])
                    info.hp_pct = hp_pct
                    info.status = status
                    info.fainted = hp_pct <= 0
                    # Reset boosts on switch
                    info.boosts = defaultdict(int)
                    info.revealed_moves = []
                    info.tera_type = None
                    info.is_dynamaxed = False

            # --- Damage / Heal ---
            elif action in ("-damage", "-heal") and len(split) >= 4:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    hp_pct, status = _parse_hp_status(split[3])
                    info.hp_pct = hp_pct
                    if status:
                        info.status = status
                    info.fainted = hp_pct <= 0

            # --- Move ---
            elif action == "move" and len(split) >= 4:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    move_name = normalize_name(split[3].strip())
                    info = _get_opp_slot(slot_idx)
                    if move_name not in info.revealed_moves:
                        info.revealed_moves.append(move_name)

            # --- Boost / Unboost ---
            elif action in ("-boost", "-unboost") and len(split) >= 5:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    stat = split[3].strip()
                    try:
                        amount = int(split[4].strip())
                    except ValueError:
                        amount = 1
                    info = _get_opp_slot(slot_idx)
                    if action == "-boost":
                        info.boosts[stat] = min(6, info.boosts[stat] + amount)
                    else:
                        info.boosts[stat] = max(-6, info.boosts[stat] - amount)

            # --- Status ---
            elif action == "-status" and len(split) >= 4:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.status = split[3].strip()

            elif action == "-curestatus" and len(split) >= 4:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.status = None

            # --- Faint ---
            elif action == "faint" and len(split) >= 3:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.fainted = True
                    info.hp_pct = 0.0
                    info.status = "fnt"

            # --- Item ---
            elif action in ("-item", "-enditem") and len(split) >= 4:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    if action == "-item":
                        info.revealed_item = normalize_name(split[3].strip())
                    else:
                        info.revealed_item = None

            # --- Ability ---
            elif action in ("-ability",) and len(split) >= 4:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.revealed_ability = normalize_name(split[3].strip())

            # --- Terastallize ---
            elif action == "-terastallize" and len(split) >= 4:
                side, slot_idx = _parse_pokemon_ident(split[2])
                tera_type = split[3].strip()
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.tera_type = tera_type
                    snap.opp_gimmicks_used.add("terastallize")
                elif side == user_name:
                    snap.own_gimmicks_used.add("terastallize")

            # --- Mega ---
            elif action == "-mega" and len(split) >= 3:
                side, _ = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    snap.opp_gimmicks_used.add("mega")
                elif side == user_name:
                    snap.own_gimmicks_used.add("mega")

            # --- Substitute ---
            elif action == "-start" and len(split) >= 4 and "substitute" in split[3].lower():
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.has_substitute = True

            elif action == "-end" and len(split) >= 4 and "substitute" in split[3].lower():
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.has_substitute = False

            # --- Dynamax ---
            elif action == "-start" and len(split) >= 4 and "dynamax" in split[3].lower():
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.is_dynamaxed = True
                    snap.opp_gimmicks_used.add("dynamax")
                elif side == user_name:
                    snap.own_gimmicks_used.add("dynamax")

            elif action == "-end" and len(split) >= 4 and "dynamax" in split[3].lower():
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.is_dynamaxed = False

            # --- Z-Power ---
            elif action == "-zpower" and len(split) >= 3:
                side, _ = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    snap.opp_gimmicks_used.add("zpower")
                elif side == user_name:
                    snap.own_gimmicks_used.add("zpower")

            # --- Weather ---
            elif action == "-weather" and len(split) >= 3:
                weather = normalize_name(split[2].strip())
                if weather == "none":
                    snap.field_state.weather = None
                else:
                    snap.field_state.weather = weather

            # --- Terrain (fieldstart / fieldend) ---
            elif action == "-fieldstart" and len(split) >= 3:
                field_effect = normalize_name(split[2].strip())
                if "terrain" in field_effect:
                    snap.field_state.terrain = field_effect
                elif "trickroom" in field_effect:
                    snap.field_state.trick_room = True
                elif "gravity" in field_effect:
                    snap.field_state.gravity = True

            elif action == "-fieldend" and len(split) >= 3:
                field_effect = normalize_name(split[2].strip())
                if "terrain" in field_effect:
                    snap.field_state.terrain = None
                elif "trickroom" in field_effect:
                    snap.field_state.trick_room = False
                elif "gravity" in field_effect:
                    snap.field_state.gravity = False

            # --- Side conditions (sidestart / sideend) ---
            elif action == "-sidestart" and len(split) >= 4:
                side_str = split[2].strip()
                condition = normalize_name(split[3].strip())
                if opp_name in side_str:
                    snap.opp_side_conditions[condition] += 1
                else:
                    snap.own_side_conditions[condition] += 1

            elif action == "-sideend" and len(split) >= 4:
                side_str = split[2].strip()
                condition = normalize_name(split[3].strip())
                if opp_name in side_str:
                    snap.opp_side_conditions.pop(condition, None)
                else:
                    snap.own_side_conditions.pop(condition, None)

            # --- Commander (Tatsugiri) ---
            elif action == "-activate" and len(split) >= 4:
                if "commander" in split[3].lower():
                    snap.commander_active = True

            # --- Clear boosts ---
            elif action == "-clearallboost":
                for info in snap.opponent_active_slots.values():
                    info.boosts = defaultdict(int)

            elif action == "-clearboost" and len(split) >= 3:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.boosts = defaultdict(int)

            elif action == "-clearnegativeboost" and len(split) >= 3:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    info.boosts = defaultdict(int, {
                        k: v for k, v in info.boosts.items() if v > 0
                    })

            # --- Replace (Zoroark illusion break) ---
            elif action == "replace" and len(split) >= 4:
                side, slot_idx = _parse_pokemon_ident(split[2])
                if side == opp_name:
                    info = _get_opp_slot(slot_idx)
                    details = split[3].strip()
                    info.species = normalize_name(details.split(",")[0])

        except (IndexError, ValueError) as exc:
            logger.debug("msg_parser: skipping malformed line '%s': %s", line, exc)
            continue

    return snap
