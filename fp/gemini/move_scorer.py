"""Move scoring engine — computes numeric quality scores for each legal action.

Scores every legal move and switch target 0-100 based on hard math:
  - Type effectiveness multiplier
  - STAB bonus
  - Base power / damage estimate
  - Ability interactions (immunities, modifiers)
  - Defender HP (can we KO?)
  - Opponent threat assessment (can THEY KO us? should we switch?)
  - Speed tier awareness (who moves first?)
  - Set prediction from RandomBattleTeamDatasets (unrevealed moves/abilities)
  - Status moves get contextual scores

The top-scored action is used as the fallback when Gemini fails,
and scores are surfaced in the prompt so Gemini has clear guidance.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from data import all_move_json, pokedex
from fp.gemini.damage_calc import (
    _check_defender_ability_immunity,
    _type_effectiveness,
    estimate_damage_pct,
    estimate_opp_stat_range,
    _calc_stat,
    _calc_hp,
    _apply_defender_ability_modifier,
)
from fp.gemini.view import GeminiBattleView, ActiveSlotView, LegalMove, OwnPokemon

logger = logging.getLogger(__name__)


@dataclass
class ScoredMove:
    """A legal move with its computed quality score."""
    move_id: str
    score: float  # 0-100 (used for fallback/MCTS, not shown in AI prompt)
    reason: str   # short explanation
    damage_pct: tuple[float, float] = (0.0, 0.0)  # (min%, max%)
    is_immune: bool = False
    type_mult: float = 1.0   # raw type effectiveness multiplier
    is_stab: bool = False    # whether this is a STAB move


@dataclass
class ScoredSwitch:
    """A switch target with its computed quality score."""
    pokemon_name: str
    score: float  # 0-100
    reason: str


@dataclass
class ThreatInfo:
    """What the opponent can do to us this turn."""
    best_move_name: str = ""
    best_move_type: str = ""
    estimated_damage_pct: float = 0.0  # average expected damage
    can_ko: bool = False
    can_2hko: bool = False
    they_outspeed: bool = False  # True if opponent likely moves first
    predicted_moves: list[str] = field(default_factory=list)  # from set data
    predicted_ability: Optional[str] = None
    predicted_item: Optional[str] = None


def _get_base_stats(species: str) -> Optional[dict]:
    """Look up base stats from pokedex."""
    entry = pokedex.get(species, {})
    return entry.get("baseStats", entry.get("basestats"))


def _get_opp_info(view: GeminiBattleView):
    """Get the primary opponent's full info."""
    if not view.snapshot or not view.snapshot.opponent_active_slots:
        return None
    for _, opp in sorted(view.snapshot.opponent_active_slots.items()):
        if not opp.fainted:
            return opp
    return None


def _predict_opp_set(opp_species: str, revealed_moves: list, revealed_ability: Optional[str],
                     revealed_item: Optional[str], is_random: bool) -> dict:
    """Predict opponent's unrevealed moves, ability, item from set data.

    Returns dict with keys: moves, ability, item (all Optional).
    """
    predicted = {"moves": [], "ability": None, "item": None}

    if not is_random:
        return predicted

    try:
        from data.pkmn_sets import RandomBattleTeamDatasets
        if not RandomBattleTeamDatasets.pkmn_sets:
            return predicted

        # Get all possible sets for this species
        raw_sets = RandomBattleTeamDatasets.pkmn_sets.get(opp_species, [])
        if not raw_sets:
            # Try normalized name
            from fp.helpers import normalize_name
            for key in RandomBattleTeamDatasets.pkmn_sets:
                if normalize_name(key) == normalize_name(opp_species):
                    raw_sets = RandomBattleTeamDatasets.pkmn_sets[key]
                    break

        if not raw_sets:
            return predicted

        # Filter sets to those matching revealed info
        revealed_set = set(m.lower() for m in revealed_moves if m)
        matching_sets = []
        for pset in raw_sets:
            set_moves = set(m.lower() for m in pset.pkmn_moveset.moves)
            # All revealed moves must be in the set
            if revealed_set.issubset(set_moves):
                if revealed_ability and pset.pkmn_set.ability.lower() != revealed_ability.lower():
                    continue
                if revealed_item and pset.pkmn_set.item.lower() != revealed_item.lower():
                    continue
                matching_sets.append(pset)

        if not matching_sets:
            # Fall back to all sets
            matching_sets = raw_sets[:5]

        # Aggregate predictions from matching sets
        move_counts = {}
        ability_counts = {}
        item_counts = {}
        total_weight = 0

        for pset in matching_sets:
            weight = pset.pkmn_set.count
            total_weight += weight
            for mv in pset.pkmn_moveset.moves:
                mv_lower = mv.lower()
                if mv_lower not in revealed_set:
                    move_counts[mv_lower] = move_counts.get(mv_lower, 0) + weight
            ability_counts[pset.pkmn_set.ability] = ability_counts.get(pset.pkmn_set.ability, 0) + weight
            item_counts[pset.pkmn_set.item] = item_counts.get(pset.pkmn_set.item, 0) + weight

        # Top predicted moves (not yet revealed)
        predicted["moves"] = sorted(move_counts, key=move_counts.get, reverse=True)[:4]

        # Most likely ability
        if not revealed_ability and ability_counts:
            predicted["ability"] = max(ability_counts, key=ability_counts.get)

        # Most likely item
        if not revealed_item and item_counts:
            predicted["item"] = max(item_counts, key=item_counts.get)

    except Exception as exc:
        logger.debug("Set prediction failed for %s: %s", opp_species, exc)

    return predicted


def compute_threat(
    view: GeminiBattleView,
    slot_idx: int = 0,
) -> ThreatInfo:
    """Compute how threatening the opponent is to our active Pokemon.

    Answers: What's their best move? Can they KO us? Do they outspeed?
    """
    threat = ThreatInfo()

    if slot_idx >= len(view.active_slots):
        return threat

    opp = _get_opp_info(view)
    if opp is None:
        return threat

    slot = view.active_slots[slot_idx]
    pkmn = slot.pokemon
    own_types = [t.lower() for t in pkmn.types] if pkmn.types else []

    # --- Predict unrevealed moves ---
    is_random = view.format_info and view.format_info.is_random
    predicted = _predict_opp_set(
        opp.species,
        opp.revealed_moves or [],
        opp.revealed_ability,
        opp.revealed_item,
        is_random,
    )
    threat.predicted_moves = predicted["moves"]
    threat.predicted_ability = predicted["ability"]
    threat.predicted_item = predicted["item"]

    # Combine revealed + predicted moves
    all_opp_moves = list(opp.revealed_moves or []) + predicted["moves"]

    # --- Compute opponent's best damage against us ---
    opp_types = [t.lower() for t in opp.types] if opp.types else []
    opp_base_stats = _get_base_stats(opp.species)

    # Our defensive stats
    own_base_stats = _get_base_stats(pkmn.species)
    if pkmn.stats:
        own_hp = pkmn.max_hp
        own_def = pkmn.stats.get("def", pkmn.stats.get("defense", 200))
        own_spd = pkmn.stats.get("spd", pkmn.stats.get("special-defense", 200))
    elif own_base_stats:
        own_hp = _calc_hp(own_base_stats.get("hp", 80), pkmn.level, ev=128)
        own_def = _calc_stat(own_base_stats.get("def", own_base_stats.get("defense", 80)), pkmn.level, ev=128)
        own_spd = _calc_stat(own_base_stats.get("spd", own_base_stats.get("special-defense", 80)), pkmn.level, ev=128)
    else:
        own_hp = 300
        own_def = 200
        own_spd = 200

    best_dmg = 0.0
    best_move_name = ""
    best_move_type = ""

    for mv_name in all_opp_moves:
        mv_data = all_move_json.get(mv_name, {})
        mv_bp = mv_data.get("basePower", 0)
        if mv_bp <= 0:
            continue

        mv_type = mv_data.get("type", "Normal").lower()
        mv_cat = mv_data.get("category", "Physical").lower()
        is_special = mv_cat == "special"

        # Opponent's attack stat (estimated)
        if opp_base_stats:
            if is_special:
                atk_base = opp_base_stats.get("spa", opp_base_stats.get("special-attack", 80))
            else:
                atk_base = opp_base_stats.get("atk", opp_base_stats.get("attack", 80))
            opp_atk = _calc_stat(atk_base, opp.level, ev=128)
        else:
            opp_atk = 200

        def_stat = own_spd if is_special else own_def

        # Type effectiveness against us
        type_mult = _type_effectiveness(mv_type, own_types)
        if type_mult == 0:
            continue

        # STAB
        stab = 1.5 if mv_type in opp_types else 1.0

        # Raw damage
        raw = ((2 * opp.level / 5 + 2) * mv_bp * opp_atk / max(def_stat, 1)) / 50 + 2
        raw *= stab * type_mult

        # Estimate as % of our remaining HP
        our_current_hp = own_hp * pkmn.hp_pct / 100.0
        avg_dmg_pct = (raw * 0.925) / max(our_current_hp, 1) * 100  # avg roll

        if avg_dmg_pct > best_dmg:
            best_dmg = avg_dmg_pct
            best_move_name = mv_name
            best_move_type = mv_type

    threat.best_move_name = best_move_name
    threat.best_move_type = best_move_type
    threat.estimated_damage_pct = round(best_dmg, 1)
    threat.can_ko = best_dmg >= 100
    threat.can_2hko = best_dmg >= 50

    # --- Speed comparison ---
    own_spe = 0
    if pkmn.stats:
        own_spe = pkmn.stats.get("spe", pkmn.stats.get("speed", 0))

    if own_spe > 0 and opp_base_stats:
        opp_base_spe = opp_base_stats.get("spe", opp_base_stats.get("speed", 80))
        _, max_opp_spe = estimate_opp_stat_range(opp_base_spe, opp.level)

        # Apply boosts
        spe_boost = (opp.boosts or {}).get("spe", 0)
        if spe_boost > 0:
            max_opp_spe = int(max_opp_spe * (2 + spe_boost) / 2)

        # Paralysis
        if opp.status == "par":
            max_opp_spe //= 2

        # Trick Room inverts
        trick_room = view.snapshot.field_state.trick_room if view.snapshot else False
        if trick_room:
            threat.they_outspeed = own_spe > max_opp_spe  # reversed in TR
        else:
            threat.they_outspeed = max_opp_spe >= own_spe

    return threat


def score_move(
    move: LegalMove,
    slot: ActiveSlotView,
    view: GeminiBattleView,
    weather: Optional[str] = None,
    threat: Optional[ThreatInfo] = None,
) -> ScoredMove:
    """Score a single move 0-100 against the current opponent."""

    opp = _get_opp_info(view)
    if opp is None:
        # No opponent info — can't score, give neutral score
        return ScoredMove(move.id, 50.0, "no opponent data")

    opp_types = [t.lower() for t in opp.types] if opp.types else []
    own_types = [t.lower() for t in slot.pokemon.types] if slot.pokemon.types else []
    move_type = move.move_type.lower()

    # --- Status moves: contextual scoring ---
    # ONLY check category, NOT base_power — BP=0 moves like Seismic Toss are damage moves
    if move.category.lower() == "status":
        # Status moves are blocked by Substitute — score them near zero
        if opp.has_substitute:
            move_json = all_move_json.get(move.id, {})
            flags = move_json.get("flags", {})
            bypasses_sub = flags.get("sound") or flags.get("bypasssub")
            ability = (slot.pokemon.ability or "").lower().replace(" ", "")
            if ability == "infiltrator":
                bypasses_sub = True
            if not bypasses_sub:
                return ScoredMove(move.id, 2.0, "blocked by Substitute")
        return _score_status_move(move, slot, view, opp)

    # --- Fixed-damage moves (BP=0 but category != status) ---
    if move.base_power == 0:
        return _score_fixed_damage_move(move, slot, view, opp)

    # --- Check ability immunity first ---
    if opp.revealed_ability:
        immunity = _check_defender_ability_immunity(opp.revealed_ability, move_type)
        if immunity:
            return ScoredMove(move.id, 0.0, f"IMMUNE ({opp.revealed_ability})", is_immune=True)

    # --- Type effectiveness ---
    type_mult = _type_effectiveness(move_type, opp_types)

    if type_mult == 0:
        return ScoredMove(move.id, 0.0, "type IMMUNE", is_immune=True)

    # --- STAB ---
    is_stab = move_type in own_types
    stab_mult = 1.5 if is_stab else 1.0

    # --- Compute actual damage estimate ---
    pkmn = slot.pokemon
    is_special = move.category.lower() == "special"

    if pkmn.stats:
        if is_special:
            atk_stat = pkmn.stats.get("spa", pkmn.stats.get("special-attack", 100))
        else:
            atk_stat = pkmn.stats.get("atk", pkmn.stats.get("attack", 100))
    else:
        atk_stat = 100

    opp_base_stats = _get_base_stats(opp.species)
    if opp_base_stats:
        if is_special:
            def_base = opp_base_stats.get("spd", opp_base_stats.get("special-defense", 80))
        else:
            def_base = opp_base_stats.get("def", opp_base_stats.get("defense", 80))
        hp_base = opp_base_stats.get("hp", 80)
        def_stat = _calc_stat(def_base, opp.level, ev=128)
        opp_max_hp = _calc_hp(hp_base, opp.level, ev=128)
    else:
        def_stat = 200
        opp_max_hp = 300

    # Check contact for ability modifiers
    move_json = all_move_json.get(move.id, {})
    flags = move_json.get("flags", {})
    is_contact = bool(flags.get("contact"))

    # Defender ability modifier
    def_ability_mult = 1.0
    if opp.revealed_ability:
        def_ability_mult = _apply_defender_ability_modifier(
            opp.revealed_ability, move_type, is_special, type_mult, is_contact
        )

    # Attacker item/ability modifiers
    item_mult = 1.0
    item = (pkmn.item or "").lower().replace(" ", "")
    if item == "lifeorb":
        item_mult = 1.3
    elif item == "choiceband" and not is_special:
        item_mult = 1.5
    elif item == "choicespecs" and is_special:
        item_mult = 1.5

    ability_mult = 1.0
    ability = (pkmn.ability or "").lower().replace(" ", "")
    if ability in ("hugepower", "purepower") and not is_special:
        ability_mult = 2.0
    elif ability == "adaptability" and is_stab:
        ability_mult = 2.0 / 1.5  # upgrades STAB from 1.5 to 2.0

    # Weather modifiers
    weather_mult = 1.0
    if weather:
        wl = weather.lower()
        if wl in ("sunnyday", "desolateland", "sun"):
            if move_type == "fire":
                weather_mult = 1.5
            elif move_type == "water":
                weather_mult = 0.5
        elif wl in ("raindance", "primordialsea", "rain"):
            if move_type == "water":
                weather_mult = 1.5
            elif move_type == "fire":
                weather_mult = 0.5

    # Full damage calc
    bp = move.base_power
    raw_damage = ((2 * pkmn.level / 5 + 2) * bp * atk_stat / max(def_stat, 1)) / 50 + 2
    raw_damage *= stab_mult * type_mult * def_ability_mult * item_mult * ability_mult * weather_mult

    min_dmg_pct = round(100.0 * raw_damage * 0.85 / max(opp_max_hp, 1), 1)
    max_dmg_pct = round(100.0 * raw_damage / max(opp_max_hp, 1), 1)

    # --- Build score ---
    # Base score from expected damage as % of remaining HP
    avg_dmg_pct = (min_dmg_pct + max_dmg_pct) / 2
    opp_remaining = opp.hp_pct

    # Score = how much of their remaining HP we take
    if opp_remaining > 0:
        damage_ratio = min(avg_dmg_pct / opp_remaining, 1.5)  # cap at 150% (overkill diminishing)
    else:
        damage_ratio = 0

    score = damage_ratio * 60  # 60 points from damage (KO = 60)

    # Bonus for super-effective
    if type_mult >= 4:
        score += 15
    elif type_mult >= 2:
        score += 10

    # Bonus for STAB
    if is_stab:
        score += 5

    # Penalty for resisted
    if type_mult <= 0.25:
        score -= 20
    elif type_mult <= 0.5:
        score -= 10

    # KO bonus — if we can KO, big bonus
    if min_dmg_pct >= opp_remaining:
        score += 20  # guaranteed KO
    elif max_dmg_pct >= opp_remaining:
        score += 10  # possible KO

    # Accuracy penalty for inaccurate moves
    accuracy = move_json.get("accuracy", 100)
    if isinstance(accuracy, (int, float)) and accuracy < 100:
        score *= accuracy / 100

    # Build reason string
    reason_parts = []
    if type_mult >= 2:
        reason_parts.append(f"{type_mult:.0f}x SE")
    elif type_mult < 1:
        reason_parts.append(f"{type_mult}x resist")
    if is_stab:
        reason_parts.append("STAB")
    if min_dmg_pct >= opp_remaining:
        reason_parts.append("GUARANTEED KO")
    elif max_dmg_pct >= opp_remaining:
        reason_parts.append("possible KO")
    reason_parts.append(f"~{min_dmg_pct:.0f}-{max_dmg_pct:.0f}%")

    # --- Threat-aware adjustments ---
    if threat:
        # If they can KO us and we outspeed: prioritize KO moves (race condition)
        if threat.can_ko and not threat.they_outspeed and min_dmg_pct >= opp.hp_pct:
            score += 15  # we can KO them before they KO us — go for it
            reason_parts.append("RACE WIN")

        # If they can KO us and they outspeed: we're dead unless we switch
        if threat.can_ko and threat.they_outspeed:
            score -= 10
            reason_parts.append("they KO first")

        # Priority moves get a massive boost when opponent outspeeds and threatens KO
        move_priority = move_json.get("priority", 0)
        if move_priority > 0 and threat.they_outspeed and threat.can_ko:
            score += 35  # priority is the only way to move first — highest situational value
            reason_parts.append(f"+{move_priority} priority — only way to survive")
        elif move_priority > 0 and threat.they_outspeed:
            score += 12
            reason_parts.append(f"+{move_priority} priority")

    # Recoil penalty — but NOT if the move KOs (recoil doesn't matter on a KO)
    if flags.get("recoil") and min_dmg_pct < opp_remaining:
        score -= 5

    # Cap at 0-100
    score = max(0.0, min(100.0, score))

    return ScoredMove(
        move_id=move.id,
        score=round(score, 1),
        reason=", ".join(reason_parts),
        damage_pct=(min_dmg_pct, max_dmg_pct),
        type_mult=type_mult,
        is_stab=is_stab,
    )


def _score_fixed_damage_move(move: LegalMove, slot: ActiveSlotView, view: GeminiBattleView, opp) -> ScoredMove:
    """Score fixed-damage moves like Seismic Toss, Night Shade, Super Fang, Counter, etc.

    These have base_power=0 but category != status. They deal damage through
    formulas that bypass the standard damage calc.
    """
    move_id = move.id.lower()
    pkmn = slot.pokemon
    opp_types = [t.lower() for t in opp.types] if opp.types else []

    # Type immunity check (e.g., Seismic Toss is Fighting — immune to Ghost)
    move_type = move.move_type.lower()
    type_mult = _type_effectiveness(move_type, opp_types)
    if type_mult == 0:
        return ScoredMove(move.id, 0.0, "type IMMUNE", is_immune=True)

    # Ability immunity check
    if opp.revealed_ability:
        immunity = _check_defender_ability_immunity(opp.revealed_ability, move_type)
        if immunity:
            return ScoredMove(move.id, 0.0, f"IMMUNE ({opp.revealed_ability})", is_immune=True)

    # Estimate damage for known fixed-damage patterns
    opp_base_stats = _get_base_stats(opp.species)
    opp_max_hp = 300  # default
    if opp_base_stats:
        hp_base = opp_base_stats.get("hp", 80)
        opp_max_hp = _calc_hp(hp_base, opp.level, ev=128)

    if move_id in ("seismictoss", "nightshade"):
        # Deals damage = user's level
        fixed_dmg = pkmn.level
        dmg_pct = round(100.0 * fixed_dmg / max(opp_max_hp, 1), 1)
        # Consistent, reliable damage — very valuable for walls
        score = min(55 + (dmg_pct * 0.5), 75)
        if dmg_pct >= opp.hp_pct:
            score = 80
            return ScoredMove(move.id, score, f"GUARANTEED KO, fixed {fixed_dmg} HP (~{dmg_pct}%)",
                              damage_pct=(dmg_pct, dmg_pct))
        n_hits = int(opp.hp_pct / max(dmg_pct, 1)) + 1
        return ScoredMove(move.id, round(score, 1), f"fixed {fixed_dmg} HP (~{dmg_pct}%, {n_hits}HKO)",
                          damage_pct=(dmg_pct, dmg_pct))

    elif move_id in ("superfang", "naturesmadness"):
        # Deals 50% of target's CURRENT HP
        dmg_pct = opp.hp_pct / 2
        score = 50 if opp.hp_pct > 30 else 30  # less valuable at low HP
        return ScoredMove(move.id, score, f"halves HP (~{dmg_pct:.0f}%)",
                          damage_pct=(dmg_pct, dmg_pct))

    elif move_id == "finalgambit":
        # User faints, deals damage = user's current HP
        own_hp = pkmn.max_hp * pkmn.hp_pct / 100 if pkmn.max_hp else 200
        dmg_pct = round(100.0 * own_hp / max(opp_max_hp, 1), 1)
        if dmg_pct >= opp.hp_pct:
            return ScoredMove(move.id, 65, f"KO trade (you faint too, ~{dmg_pct}%)",
                              damage_pct=(dmg_pct, dmg_pct))
        return ScoredMove(move.id, 20, f"you faint, only ~{dmg_pct}% dmg",
                          damage_pct=(dmg_pct, dmg_pct))

    elif move_id in ("counter", "mirrorcoat", "metalburst"):
        # Deals 2x (or 1.5x) the damage received — unpredictable
        return ScoredMove(move.id, 35, "reflects damage (situational)")

    elif move_id == "endeavor":
        # Brings target to user's HP — great at low HP
        if pkmn.hp_pct <= 10:
            return ScoredMove(move.id, 60, f"equalizes to {pkmn.hp_pct}% HP")
        return ScoredMove(move.id, 15, f"equalizes to {pkmn.hp_pct}% (too healthy)")

    # Unknown BP=0 damage move — give it a moderate score
    return ScoredMove(move.id, 30, "fixed/variable damage")


def _score_status_move(move: LegalMove, slot: ActiveSlotView, view: GeminiBattleView, opp) -> ScoredMove:
    """Score a status move based on full battle context."""
    move_id = move.id.lower()

    opp_conditions = view.snapshot.opp_side_conditions if view.snapshot else {}
    own_conditions = view.snapshot.own_side_conditions if view.snapshot else {}
    turn = view.turn or 0

    # Compute threat for setup safety checks
    threat = compute_threat(view, slot.slot_index)

    # --- Entry hazards ---
    if move_id == "stealthrock":
        if any("stealthrock" in k for k in opp_conditions):
            return ScoredMove(move.id, 5.0, "already up")
        # Earlier = more total chip damage
        if turn <= 2:
            return ScoredMove(move.id, 78.0, "Stealth Rock turn 1-2 — maximum chip value")
        if turn <= 5:
            return ScoredMove(move.id, 65.0, "Stealth Rock early game")
        return ScoredMove(move.id, 48.0, "Stealth Rock mid/late game")

    if move_id == "spikes":
        existing_layers = opp_conditions.get("spikes", 0)
        if existing_layers >= 3:
            return ScoredMove(move.id, 5.0, "max Spikes already up")
        base_scores = [68.0, 55.0, 40.0]
        score = base_scores[existing_layers]
        if turn > 6:
            score = max(score - 10, 25.0)
        label = ["layer 1 (high value)", "layer 2 (good value)", "layer 3 (diminishing)"][existing_layers]
        return ScoredMove(move.id, score, f"Spikes {label}")

    if move_id == "toxicspikes":
        existing = opp_conditions.get("toxicspikes", 0)
        if existing >= 2:
            return ScoredMove(move.id, 5.0, "max Toxic Spikes already up")
        return ScoredMove(move.id, 55.0 if existing == 0 else 42.0, f"Toxic Spikes layer {existing + 1}")

    if move_id == "stickyweb":
        if any("stickyweb" in k for k in opp_conditions):
            return ScoredMove(move.id, 5.0, "already up")
        if turn <= 3:
            return ScoredMove(move.id, 70.0, "Sticky Web — speed control")
        return ScoredMove(move.id, 45.0, "Sticky Web")

    # --- Hazard removal ---
    if move_id in ("defog", "rapidspin", "mortalspin", "tidyup", "courtchange"):
        if own_conditions:
            hazard_count = len(own_conditions)
            return ScoredMove(move.id, min(40.0 + hazard_count * 10, 65.0), f"hazard removal ({hazard_count} hazards up)")
        return ScoredMove(move.id, 10.0, "hazard removal (no hazards up)")

    # --- Setup moves — score based on safety ---
    _SETUP_MOVES = {
        "swordsdance", "nastyplot", "dragondance", "quiverdance",
        "calmmind", "bulkup", "irondefense", "shellsmash",
        "tailglow", "coil", "geomancy", "shiftgear", "growth",
        "agility", "autotomize", "rockpolish", "victorydance",
    }
    if move_id in _SETUP_MOVES:
        if threat.can_ko and threat.they_outspeed:
            return ScoredMove(move.id, 8.0, "setup unsafe — they KO before we move")
        if threat.can_ko:
            return ScoredMove(move.id, 28.0, "setup risky — they can KO on the switch")
        if threat.can_2hko:
            return ScoredMove(move.id, 48.0, "setup possible — they 2HKO, predict switch")
        # Safe setup opportunity
        return ScoredMove(move.id, 72.0, "setup opportunity — opponent cannot KO")

    # --- Tailwind (doubles/VGC) ---
    if move_id == "tailwind":
        field = view.snapshot.field_state if view.snapshot else None
        if field and getattr(field, "tailwind_own", False):
            return ScoredMove(move.id, 5.0, "Tailwind already active")
        return ScoredMove(move.id, 72.0, "Tailwind — speed control")

    # --- Trick Room ---
    if move_id == "trickroom":
        field = view.snapshot.field_state if view.snapshot else None
        if field and field.trick_room:
            return ScoredMove(move.id, 30.0, "cancel Trick Room")
        return ScoredMove(move.id, 65.0, "Trick Room setup")

    # --- Recovery ---
    if move_id in ("recover", "roost", "softboiled", "moonlight", "morningsun",
                    "synthesis", "slackoff", "milkdrink", "shoreup", "rest",
                    "leechseed", "ingrain", "aquaring"):
        hp = slot.pokemon.hp_pct
        if hp <= 35:
            return ScoredMove(move.id, 62.0, f"recovery (critical: {hp:.0f}% HP)")
        elif hp <= 60:
            return ScoredMove(move.id, 45.0, f"recovery (low: {hp:.0f}% HP)")
        elif hp <= 80:
            return ScoredMove(move.id, 28.0, f"recovery (moderate: {hp:.0f}% HP)")
        return ScoredMove(move.id, 8.0, f"recovery (not needed: {hp:.0f}% HP)")

    # --- Status-inflicting moves ---
    if move_id in ("thunderwave", "toxic", "willowisp", "glare", "stunspore",
                    "nuzzle", "yawn", "sleeppowder", "spore", "hypnosis",
                    "darkvoid", "lovelykiss", "sing", "grasswhistle"):
        if opp.status:
            return ScoredMove(move.id, 2.0, "already statused")
        if opp.has_substitute:
            return ScoredMove(move.id, 5.0, "blocked by Substitute")
        if move_id in ("spore", "sleeppowder", "darkvoid"):
            return ScoredMove(move.id, 58.0, "sleep — best status move")
        if move_id in ("hypnosis", "lovelykiss", "sing", "grasswhistle", "yawn"):
            accuracy = all_move_json.get(move.id, {}).get("accuracy", 100)
            base = 50.0 if isinstance(accuracy, bool) or accuracy >= 100 else 35.0
            return ScoredMove(move.id, base, f"sleep (acc:{accuracy}%)")
        return ScoredMove(move.id, 42.0, "status move")

    # --- Protect variants ---
    if move_id in ("protect", "detect", "spikyshield", "banefulbunker",
                    "kingsshield", "silktrap", "obstruct", "maxguard"):
        return ScoredMove(move.id, 22.0, "protect — scouting/stalling")

    # --- Pivots (status-category, e.g. Teleport, Parting Shot) ---
    if move_id in ("partingshot", "teleport", "batonpass"):
        return ScoredMove(move.id, 40.0, "pivot/status")

    # --- Item disruption ---
    if move_id in ("trick", "switcheroo"):
        return ScoredMove(move.id, 32.0, "item disruption")

    # --- Substitute ---
    if move_id == "substitute":
        if slot.pokemon.hp_pct >= 50:
            return ScoredMove(move.id, 42.0, "substitute (healthy)")
        return ScoredMove(move.id, 12.0, "substitute (low HP)")

    # --- Encore ---
    if move_id == "encore":
        return ScoredMove(move.id, 40.0, "encore — lock into status/setup")

    # --- Taunt ---
    if move_id == "taunt":
        return ScoredMove(move.id, 38.0, "taunt — prevent status moves")

    # --- Default ---
    return ScoredMove(move.id, 25.0, "status move")


def score_switch(
    target: OwnPokemon,
    view: GeminiBattleView,
    threat: Optional[ThreatInfo] = None,
    current_pokemon: Optional[OwnPokemon] = None,
) -> ScoredSwitch:
    """Score a switch target based on defensive and offensive matchup."""
    opp = _get_opp_info(view)
    if opp is None:
        return ScoredSwitch(target.name, 30.0, "no opponent data")

    opp_types = [t.lower() for t in opp.types] if opp.types else []
    own_types = [t.lower() for t in target.types] if target.types else []

    score = 30.0  # base switch score (switching has inherent cost)
    reasons = []

    # If opponent threatens KO on our current mon, switching is more urgent
    if threat and threat.can_ko and threat.they_outspeed:
        score += 10  # switching is much more valuable when we'd die staying in
        reasons.append("current mon in danger")

    # PENALTY: if current Pokemon is already walling well, switching is wasteful
    if current_pokemon and current_pokemon.hp_pct >= 50:
        current_types = [t.lower() for t in current_pokemon.types] if current_pokemon.types else []
        current_walls = True
        for opp_type in opp_types:
            mult = _type_effectiveness(opp_type, current_types)
            if mult > 1:
                current_walls = False
                break
        if current_walls and not (threat and threat.can_ko):
            score -= 12  # already in a good position, don't switch
            reasons.append("current mon walls fine")

    # --- Defensive: how well do we take opponent's STAB? ---
    for opp_type in opp_types:
        mult = _type_effectiveness(opp_type, own_types)
        if mult == 0:
            score += 20
            reasons.append(f"immune to {opp_type.capitalize()}")
        elif mult <= 0.25:
            score += 12
            reasons.append(f"4x resist {opp_type.capitalize()}")
        elif mult <= 0.5:
            score += 8
            reasons.append(f"resist {opp_type.capitalize()}")
        elif mult >= 4:
            score -= 25
            reasons.append(f"4x WEAK to {opp_type.capitalize()}")
        elif mult >= 2:
            score -= 15
            reasons.append(f"WEAK to {opp_type.capitalize()}")

    # --- Offensive: do we threaten the opponent? ---
    best_offensive_mult = 0
    if target.moves:
        for mv_name in target.moves[:4]:
            mv_data = all_move_json.get(mv_name, {})
            mv_type = mv_data.get("type", "Normal").lower()
            mv_bp = mv_data.get("basePower", 0)
            if mv_bp > 0:
                mult = _type_effectiveness(mv_type, opp_types)
                is_stab = mv_type in own_types
                effective_mult = mult * (1.5 if is_stab else 1.0)
                if effective_mult > best_offensive_mult:
                    best_offensive_mult = effective_mult

    if best_offensive_mult >= 3:  # SE + STAB
        score += 15
        reasons.append("strong offensive matchup")
    elif best_offensive_mult >= 2:
        score += 8
        reasons.append("SE coverage")
    elif best_offensive_mult <= 0.5:
        score -= 5
        reasons.append("no good attacks")

    # --- HP penalty ---
    if target.hp_pct <= 25:
        score -= 15
        reasons.append("very low HP")
    elif target.hp_pct <= 50:
        score -= 5
        reasons.append("low HP")

    # --- Check against opponent's predicted coverage moves ---
    if threat and threat.predicted_moves:
        for pred_mv in threat.predicted_moves[:2]:  # top 2 predicted
            mv_data = all_move_json.get(pred_mv, {})
            mv_type = mv_data.get("type", "Normal").lower()
            mv_bp = mv_data.get("basePower", 0)
            if mv_bp > 0:
                mult = _type_effectiveness(mv_type, own_types)
                if mult >= 2:
                    score -= 8
                    reasons.append(f"predicted {pred_mv} hits hard")
                    break  # one warning is enough

    # --- Hazard damage penalty ---
    if view.snapshot:
        own_conditions = view.snapshot.own_side_conditions
        hazard_dmg = 0
        if any("stealthrock" in k for k in own_conditions):
            # Stealth Rock damage based on type
            rock_mult = _type_effectiveness("rock", own_types)
            hazard_dmg += 12.5 * rock_mult
        spike_layers = sum(v for k, v in own_conditions.items() if k == "spikes")
        if spike_layers > 0 and "flying" not in own_types:
            hazard_dmg += [0, 12.5, 16.7, 25][min(spike_layers, 3)]

        if hazard_dmg >= 25:
            score -= 10
            reasons.append(f"takes ~{hazard_dmg:.0f}% hazard dmg")
        elif hazard_dmg > 0:
            score -= 3

    score = max(0.0, min(100.0, score))
    return ScoredSwitch(target.name, round(score, 1), ", ".join(reasons) if reasons else "neutral matchup")


def score_all_actions(
    view: GeminiBattleView,
    slot_idx: int = 0,
) -> tuple[list[ScoredMove], list[ScoredSwitch], ThreatInfo]:
    """Score all legal moves and switch targets for a given active slot.

    Returns (scored_moves, scored_switches, threat_info) with moves/switches sorted by score descending.
    """
    threat = ThreatInfo()

    if slot_idx >= len(view.active_slots):
        return [], [], threat

    slot = view.active_slots[slot_idx]
    weather = None
    if view.snapshot and view.snapshot.field_state.weather:
        weather = view.snapshot.field_state.weather

    # Compute opponent threat FIRST — this informs move and switch scoring
    threat = compute_threat(view, slot_idx)

    # Score moves
    scored_moves = []
    for move in slot.legal_moves:
        if move.disabled or move.pp <= 0:
            continue
        scored = score_move(move, slot, view, weather, threat)
        scored_moves.append(scored)

    scored_moves.sort(key=lambda m: m.score, reverse=True)

    # Score switches (pass current active Pokemon so we can penalize unnecessary switches)
    scored_switches = []
    if not slot.trapped:
        current_pkmn = slot.pokemon
        for target in view.legal_switch_targets:
            scored = score_switch(target, view, threat, current_pokemon=current_pkmn)
            scored_switches.append(scored)

    scored_switches.sort(key=lambda s: s.score, reverse=True)

    return scored_moves, scored_switches, threat


def get_best_action(view: GeminiBattleView, slot_idx: int = 0) -> str:
    """Return the best action as a Showdown decision string.

    Used as the fallback when Gemini fails — always picks the
    mathematically best option instead of blindly defaulting to move 1.
    """
    scored_moves, scored_switches, _threat = score_all_actions(view, slot_idx)

    best_move_score = scored_moves[0].score if scored_moves else -1
    best_switch_score = scored_switches[0].score if scored_switches else -1

    # Prefer moves unless a switch is significantly better
    # (switching has tempo cost, so it needs to be clearly better)
    if best_switch_score > best_move_score + 10 and scored_switches:
        return f"switch {scored_switches[0].pokemon_name}"

    if scored_moves:
        best = scored_moves[0]
        if best.is_immune and scored_switches:
            # Our best move is immune — must switch
            return f"switch {scored_switches[0].pokemon_name}"
        return f"move {best.move_id}"

    if scored_switches:
        return f"switch {scored_switches[0].pokemon_name}"

    return "move struggle"
