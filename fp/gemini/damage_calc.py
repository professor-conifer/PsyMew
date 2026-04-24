"""Simplified damage estimation and speed tier analysis for Gemini prompts.

Provides approximate damage ranges so Gemini can reason about KOs and 2HKOs.
Not intended to be exact (poke-engine handles that for MCTS) -- just good enough
for informed decision-making.

Also provides speed comparison utilities for turn ordering awareness.
"""

import logging
import math
from typing import Optional

from data import all_move_json, pokedex
from fp.gemini.prompt import _TYPE_CHART

logger = logging.getLogger(__name__)


def _calc_stat(base: int, level: int, ev: int = 85, iv: int = 31, nature_mult: float = 1.0) -> int:
    """Calculate a non-HP stat value."""
    return int((math.floor(((2 * base + iv + math.floor(ev / 4)) * level) / 100) + 5) * nature_mult)


def _calc_hp(base: int, level: int, ev: int = 85, iv: int = 31) -> int:
    """Calculate HP stat value."""
    if base == 1:  # Shedinja
        return 1
    return math.floor(((2 * base + iv + math.floor(ev / 4)) * level) / 100) + level + 10


def estimate_opp_stat_range(base_stat: int, level: int, is_speed: bool = False) -> tuple[int, int]:
    """Estimate opponent stat range assuming unknown EVs/nature.

    Returns (min_reasonable, max_reasonable):
    - min: 0 EVs, hindering nature (0.9x)
    - max: 252 EVs, boosting nature (1.1x)
    """
    min_stat = _calc_stat(base_stat, level, ev=0, nature_mult=0.9)
    max_stat = _calc_stat(base_stat, level, ev=252, nature_mult=1.1)
    return (min_stat, max_stat)


def _type_effectiveness(atk_type: str, def_types: list[str]) -> float:
    """Calculate type effectiveness multiplier."""
    chart = _TYPE_CHART.get(atk_type.lower(), {})
    mult = 1.0
    for dt in def_types:
        mult *= chart.get(dt.lower(), 1.0)
    return mult


def _check_defender_ability_immunity(
    defender_ability: Optional[str], move_type: str
) -> Optional[str]:
    """Check if the defender's ability makes it immune to this move type.

    Returns a label string like "IMMUNE (Lightning Rod)" if immune, else None.
    """
    if not defender_ability:
        return None
    ability = defender_ability.lower().replace(" ", "").replace("-", "")
    mt = move_type.lower()

    # Electric immunities
    if mt == "electric" and ability in ("lightningrod", "voltabsorb", "motordrive"):
        return f"IMMUNE ({defender_ability})"

    # Water immunities
    if mt == "water" and ability in ("waterabsorb", "stormdrain", "dryskin"):
        return f"IMMUNE ({defender_ability})"

    # Fire immunity
    if mt == "fire" and ability == "flashfire":
        return f"IMMUNE ({defender_ability})"

    # Ground immunity
    if mt == "ground" and ability == "levitate":
        return f"IMMUNE ({defender_ability})"

    # Grass immunity
    if mt == "grass" and ability == "sapsipper":
        return f"IMMUNE ({defender_ability})"

    return None


def _apply_defender_ability_modifier(
    defender_ability: Optional[str],
    move_type: str,
    is_special: bool,
    type_eff: float,
    is_contact: bool = False,
) -> float:
    """Return a damage multiplier from the defender's ability (non-immunity effects)."""
    if not defender_ability:
        return 1.0
    ability = defender_ability.lower().replace(" ", "").replace("-", "")
    mt = move_type.lower()

    # Thick Fat: halves Fire and Ice
    if ability == "thickfat" and mt in ("fire", "ice"):
        return 0.5

    # Heatproof: halves Fire
    if ability == "heatproof" and mt == "fire":
        return 0.5

    # Filter / Solid Rock / Prism Armor: 0.75x on super-effective
    if ability in ("filter", "solidrock", "prismarmor") and type_eff > 1.0:
        return 0.75

    # Fluffy: halves contact damage, but 2x Fire
    if ability == "fluffy":
        if mt == "fire":
            return 2.0
        if is_contact:
            return 0.5

    # Ice Scales: halves special damage
    if ability == "icescales" and is_special:
        return 0.5

    # Fur Coat: halves physical damage
    if ability == "furcoat" and not is_special:
        return 0.5

    # Multiscale / Shadow Shield: halves damage at full HP (we can't check HP here,
    # but caller can factor this in via defender_hp_pct)

    return 1.0


def estimate_damage_pct(
    atk_stat: int,
    def_stat: int,
    base_power: int,
    level: int,
    move_type: str,
    attacker_types: list[str],
    defender_types: list[str],
    is_special: bool = False,
    weather: Optional[str] = None,
    attacker_item: Optional[str] = None,
    attacker_ability: Optional[str] = None,
    defender_ability: Optional[str] = None,
    defender_max_hp: Optional[int] = None,
    defender_hp_pct: float = 100.0,
    is_contact: bool = False,
) -> tuple[float, float]:
    """Estimate damage as a percentage of defender's max HP.

    Returns (min_pct, max_pct) representing the damage roll range.
    Uses the standard Pokemon damage formula with common modifiers.
    Returns (0.0, 0.0) if the defender is immune via ability.
    """
    if base_power <= 0:
        return (0.0, 0.0)

    # Check defender ability immunity first
    if _check_defender_ability_immunity(defender_ability, move_type):
        return (0.0, 0.0)

    if def_stat <= 0:
        def_stat = 1

    # Base damage formula
    damage = ((2 * level / 5 + 2) * base_power * atk_stat / def_stat) / 50 + 2

    # STAB
    is_stab = move_type.lower() in [t.lower() for t in attacker_types]
    if is_stab:
        damage *= 1.5

    # Type effectiveness
    eff = _type_effectiveness(move_type, defender_types)
    damage *= eff

    # Defender ability modifier (non-immunity: Thick Fat, Filter, etc.)
    def_ability_mult = _apply_defender_ability_modifier(
        defender_ability, move_type, is_special, eff, is_contact
    )
    damage *= def_ability_mult

    # Common item modifiers
    if attacker_item:
        item = attacker_item.lower().replace(" ", "")
        if item == "lifeorb":
            damage *= 1.3
        elif item == "choiceband" and not is_special:
            damage *= 1.5
        elif item == "choicespecs" and is_special:
            damage *= 1.5

    # Common attacker ability modifiers
    if attacker_ability:
        ability = attacker_ability.lower().replace(" ", "")
        if ability in ("hugepower", "purepower"):
            if not is_special:
                damage *= 2.0
        elif ability == "adaptability":
            # STAB becomes 2x instead of 1.5x (net +0.33x on top of the 1.5 already applied)
            if is_stab:
                damage *= (2.0 / 1.5)

    # Weather
    if weather:
        weather_lower = weather.lower()
        move_type_lower = move_type.lower()
        if weather_lower in ("sunnyday", "desolateland", "sun"):
            if move_type_lower == "fire":
                damage *= 1.5
            elif move_type_lower == "water":
                damage *= 0.5
        elif weather_lower in ("raindance", "primordialsea", "rain"):
            if move_type_lower == "water":
                damage *= 1.5
            elif move_type_lower == "fire":
                damage *= 0.5

    # Estimate HP if not provided
    if defender_max_hp is None:
        # Rough estimate -- assume level 100 with base HP of 80
        defender_max_hp = 300  # reasonable average

    # Damage roll range: 85% to 100%
    min_dmg = damage * 0.85
    max_dmg = damage * 1.0

    min_pct = round(100.0 * min_dmg / defender_max_hp, 1)
    max_pct = round(100.0 * max_dmg / defender_max_hp, 1)

    return (min_pct, max_pct)


def format_damage_hint(min_pct: float, max_pct: float, defender_hp_pct: float) -> str:
    """Format a damage estimate into a human-readable hint string."""
    if max_pct <= 0:
        return ""

    hint = f"~{min_pct:.0f}-{max_pct:.0f}%"

    # KO analysis
    if min_pct >= defender_hp_pct:
        hint += " (GUARANTEED KO!)"
    elif max_pct >= defender_hp_pct:
        hint += " (possible KO)"
    elif min_pct * 2 >= defender_hp_pct:
        hint += " (2HKO)"
    elif max_pct * 2 >= defender_hp_pct:
        hint += " (likely 2HKO)"
    elif min_pct * 3 >= defender_hp_pct:
        hint += " (3HKO)"

    return hint


def compute_damage_for_move(
    move_id: str,
    move_type: str,
    move_category: str,
    move_bp: int,
    attacker_stats: dict,
    attacker_types: list[str],
    attacker_level: int,
    attacker_item: Optional[str],
    attacker_ability: Optional[str],
    defender_types: list[str],
    defender_hp_pct: float,
    defender_level: int = 100,
    defender_base_stats: Optional[dict] = None,
    defender_ability: Optional[str] = None,
    weather: Optional[str] = None,
) -> str:
    """Compute damage hint for a single move against the current opponent.

    Returns a formatted string like "~45-53% (2HKO)" or "" if not applicable.
    If the defender's ability grants immunity, returns "IMMUNE (AbilityName)".
    """
    if move_bp <= 0 or move_category.lower() == "status":
        return ""

    # Check ability immunity first — surface this prominently
    if defender_ability:
        immunity = _check_defender_ability_immunity(defender_ability, move_type)
        if immunity:
            return immunity

    is_special = move_category.lower() == "special"

    # Check if move makes contact (from move data)
    move_json = all_move_json.get(move_id, {})
    flags = move_json.get("flags", {})
    is_contact = bool(flags.get("contact"))

    # Attacker stat
    if is_special:
        atk_stat = attacker_stats.get("spa", attacker_stats.get("special-attack", 100))
    else:
        atk_stat = attacker_stats.get("atk", attacker_stats.get("attack", 100))

    # Defender stat -- estimate from base stats
    if defender_base_stats:
        if is_special:
            def_base = defender_base_stats.get("spd", defender_base_stats.get("special-defense", 80))
        else:
            def_base = defender_base_stats.get("def", defender_base_stats.get("defense", 80))

        hp_base = defender_base_stats.get("hp", 80)
        # Use average investment estimate (128 EVs)
        def_stat = _calc_stat(def_base, defender_level, ev=128)
        defender_max_hp = _calc_hp(hp_base, defender_level, ev=128)
    else:
        def_stat = 200  # reasonable default
        defender_max_hp = 300

    min_pct, max_pct = estimate_damage_pct(
        atk_stat=atk_stat,
        def_stat=def_stat,
        base_power=move_bp,
        level=attacker_level,
        move_type=move_type,
        attacker_types=attacker_types,
        defender_types=defender_types,
        is_special=is_special,
        weather=weather,
        attacker_item=attacker_item,
        attacker_ability=attacker_ability,
        defender_ability=defender_ability,
        defender_max_hp=defender_max_hp,
        defender_hp_pct=defender_hp_pct,
        is_contact=is_contact,
    )

    return format_damage_hint(min_pct, max_pct, defender_hp_pct)


def compute_speed_analysis(
    own_speed: int,
    own_name: str,
    opp_species: str,
    opp_level: int,
    opp_boosts: dict,
    opp_status: Optional[str],
    trick_room: bool,
    own_tailwind: bool,
    opp_tailwind: bool,
) -> str:
    """Generate a speed comparison analysis string.

    Returns a multi-line string describing who moves first and why.
    """
    # Look up opponent base speed
    dex_entry = pokedex.get(opp_species, {})
    base_stats = dex_entry.get("baseStats", dex_entry.get("basestats", {}))
    base_spe = base_stats.get("spe", base_stats.get("speed", 80))

    min_spe, max_spe = estimate_opp_stat_range(base_spe, opp_level)

    # Apply boosts
    spe_boost = opp_boosts.get("spe", 0)
    if spe_boost > 0:
        boost_mult = (2 + spe_boost) / 2
    elif spe_boost < 0:
        boost_mult = 2 / (2 - spe_boost)
    else:
        boost_mult = 1.0
    min_spe = int(min_spe * boost_mult)
    max_spe = int(max_spe * boost_mult)

    # Paralysis halves speed
    if opp_status == "par":
        min_spe = min_spe // 2
        max_spe = max_spe // 2

    # Tailwind doubles speed
    effective_own = own_speed
    if own_tailwind:
        effective_own *= 2
    if opp_tailwind:
        min_spe *= 2
        max_spe *= 2

    lines = []
    lines.append(f"  Your {own_name}: Spe={effective_own}" +
                 (" (Tailwind)" if own_tailwind else ""))
    lines.append(f"  Opposing {opp_species}: Est. Spe ~{min_spe}-{max_spe} (base {base_spe})" +
                 (" (paralyzed, halved)" if opp_status == "par" else "") +
                 (" (Tailwind)" if opp_tailwind else "") +
                 (f" (boost {'+' if spe_boost > 0 else ''}{spe_boost})" if spe_boost != 0 else ""))

    if trick_room:
        if effective_own < min_spe:
            lines.append("  -> Trick Room ACTIVE: You move FIRST (slower = first)")
        elif effective_own > max_spe:
            lines.append("  -> Trick Room ACTIVE: You move LAST (faster = last)")
        else:
            lines.append("  -> Trick Room ACTIVE: Speed tie range, uncertain order")
    else:
        if effective_own > max_spe:
            lines.append("  -> You move FIRST")
        elif effective_own < min_spe:
            lines.append("  -> Opponent moves FIRST")
        else:
            lines.append("  -> Speed tie range -- uncertain who moves first")

    return "\n".join(lines)
