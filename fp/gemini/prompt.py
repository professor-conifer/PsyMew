"""Build system and turn prompts for Gemini decision-making."""

import logging
from typing import Optional

from data import pokedex
from fp.gemini.format_detection import FormatInfo
from fp.gemini.view import GeminiBattleView

logger = logging.getLogger(__name__)


def build_system_prompt(
    format_info: FormatInfo,
    format_rules_text: str,
    format_meta_context: str = "",
) -> str:
    """Build the system prompt for the Gemini decision engine."""
    is_doubles = getattr(format_info, "gametype", "singles") in ("doubles", "triples")

    meta_block = ""
    if format_meta_context:
        meta_block = f"\nCURRENT META CONTEXT:\n{format_meta_context}\n"

    doubles_block = ""
    if is_doubles:
        doubles_block = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOUBLES / VGC SPECIFIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Fake Out timing: use it to buy a free turn for your main attacker, not just for chip damage
- Protect is information: when the opponent uses Protect, they expected your attack — punish the reveal next turn
- Tailwind vs Trick Room: identify which archetype they're running by turn 2 and adapt immediately
- Spread moves hit both targets (Rock Slide, Earthquake, Heat Wave) — prioritize when both targets are vulnerable
- Redirection (Rage Powder / Follow Me) blocks targeted moves — predict it and redirect toward the supporter instead
- Target selection matters: if their Pokemon A is immune to your move, targeting Pokemon B may not be correct either — read the board
- Position advantage: having the better Tailwind/TR control is often more valuable than immediate damage
"""

    return f"""You are PsyMew — a master-level competitive Pokémon battle AI designed to challenge and defeat world-class players on Pokémon Showdown.

FORMAT: {format_info.format_name} (Gen {format_info.gen}, {format_info.gametype})
{format_rules_text}
{meta_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR ROLE: STRATEGIC COMMANDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are not a move picker. You are a strategic general. Every turn you receive battle data — damage estimates, type matchups, speed tiers, threat assessments. This data informs your reasoning. It does not make your decision.

DECISION FRAMEWORK — evaluate in this order every turn:

1. WIN CONDITION
   What is my path to winning? Which of my Pokémon can sweep, wall, or outlast to victory?
   What does that win condition need: specific threats removed, hazards up, a gimmick saved?
   Am I protecting it, or sacrificing it early for short-term gain?

2. OPPONENT'S WIN CONDITION
   What is their clearest path to winning? Which Pokémon is their key threat?
   What do I need to deny them? Do I have a safe answer, or must I play around it?

3. THE PREDICTION GAME
   What does my opponent expect me to do here?
   Is there a better play they won't anticipate?
   Creating 50/50s — situations where their read determines the outcome — is elite play.
   If they guess right, you break even. If they guess wrong, you gain a massive advantage.
   Identify the 50/50 and choose the branch that punishes their most likely mistake.

4. TEMPO AND MOMENTUM
   Am I forcing their hand or reacting to theirs?
   Maintaining tempo means threatening rather than answering.
   Pivot moves (U-turn / Volt Switch / Flip Turn) steal tempo and gather information.
   Entry hazards compound every turn — Stealth Rock at turn 1 may be the strongest play
   even if a damage move looks better on paper right now.

5. RESOURCE MANAGEMENT
   HP, PP, and gimmicks (Tera / Mega / Z-Move / Dynamax) are finite.
   Do NOT spend Tera to win a turn you would have won anyway.
   Save gimmicks for the moment they create an unwinnable scenario for the opponent.
   Endgame (2v2 or 1v1): stop approximating — calculate exact KOs from the damage ranges given.
{doubles_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ELITE PLAY PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Force decisions: create situations where every option the opponent has is a bad guess for them
- Deny free information: don't reveal your win condition before it's too late to stop
- Win condition protection takes priority over short-term gains
- Speed tier 50/50s: when opponent's speed is within ~10% of yours after EV/nature variance, treat it as a live coin flip and play accordingly
- Chip damage compounds: Burn / Poison / Rocks / Spikes / Sand wins long games without direct KOs
- Never trade your answer to their win condition for neutral KO value elsewhere on their team
- Pivoting out of a bad matchup is often stronger than clicking the best move — momentum matters
- Setup turns are free turns: if you can use Dragon Dance / Swords Dance / Calm Mind safely, that's often worth more than any direct attack

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOVE DATA FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Move entries below show: [Type] — effectiveness | STAB flag | estimated damage vs opponent HP | PP remaining.
Damage estimates use your current stats and predicted opponent stats. Use them as inputs to your reasoning — not as a decision order.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISTRIBUTION-BASED OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You output RELATIVE PREFERENCES as integer weights (1-10) using the
choose_action_distribution tool. The actual move is sampled from your
weighted distribution — this makes you unpredictable. Opponents cannot
exploit deterministic patterns because you mix strategies naturally.

Weight assignment guidance:
- Assign higher weights to options you believe are strategically stronger
- Spread weights across GENUINELY VIABLE alternatives — if two moves
  are close in value, give them similar weights
- Only concentrate weight heavily (8-10) when one option clearly dominates
- Include ALL options you consider viable, even if they're not your top pick
- Omit only options you've ruled out entirely (immune moves, suicidal plays)
- The distribution IS your strategy — think about what mix of plays your
  opponent will struggle to predict, not just what you'd do in one game

If the provided MCTS data strongly favors an option your strategic
analysis disagrees with, trust your reasoning — the search engine
makes assumptions about unrevealed sets that may not match reality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MCTS SEARCH INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each turn you will receive game-theoretic search data from our battle
engine. The engine runs millions of simulated rollouts across multiple
possible opponent team configurations (determinizations). For each
option, you'll see:
  - Visit percentage: how often the search engine selected this option
  - Average score: expected outcome when this option was played (0=loss, 1=win)

Use this data as strategic input alongside everything else. The search
works under assumptions about their unrevealed moves, items, and
abilities — it may overfit to common configurations. Your judgment
about what this specific opponent actually has takes priority.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. You MUST use the choose_action_distribution tool. Never respond with text.
2. ONLY reference options from the enum values in the tool schema — no exceptions.
3. NEVER assign weight to a move against a type-immune target (shown as IMMUNE).
4. Move data and MCTS stats are inputs to your reasoning — not a decision order.
"""


def _get_opp_base_stats(opp_species: str) -> Optional[dict]:
    """Look up base stats for an opponent species from the pokedex."""
    dex_entry = pokedex.get(opp_species, {})
    return dex_entry.get("baseStats", dex_entry.get("basestats"))


def _get_weather_for_calc(view: GeminiBattleView) -> Optional[str]:
    """Extract weather string for damage calc from view snapshot."""
    if view.snapshot and view.snapshot.field_state.weather:
        return view.snapshot.field_state.weather
    return None


def _build_game_phase_guidance(view: GeminiBattleView) -> str:
    """Generate situation-specific strategy guidance based on game state."""
    own_alive = sum(1 for p in view.own_team if p.hp > 0)

    # Count opponent alive from snapshot
    opp_alive = 6  # assume 6 if unknown
    if view.snapshot and view.snapshot.opponent_active_slots:
        # Count non-fainted in known slots
        known_fainted = sum(1 for o in view.snapshot.opponent_active_slots.values() if o.fainted)
        opp_alive = max(1, 6 - known_fainted)  # rough estimate

    guidance = []

    # Advantage assessment
    if own_alive > opp_alive + 1:
        guidance.append("POSITION: You have a numbers advantage. Play conservatively -- trade 1-for-1 and protect your remaining Pokemon.")
    elif opp_alive > own_alive + 1:
        guidance.append("POSITION: You are BEHIND. You must find a sweep opportunity or make a game-changing play. Identify if any of your remaining Pokemon can clean up their weakened team.")

    # Game phase
    turn = view.turn or 0
    if turn <= 3:
        guidance.append("PHASE: Early game. Prioritize information gathering, hazard setting, and positioning. Avoid overcommitting your win condition.")
    elif own_alive <= 2 or opp_alive <= 2:
        guidance.append("PHASE: ENDGAME. Every HP point matters. Use the damage estimates below to calculate exact KOs before committing.")

    # Hazard warnings
    if view.snapshot:
        own_conditions = view.snapshot.own_side_conditions
        if any("stealthrock" in k for k in own_conditions):
            guidance.append("WARNING: Stealth Rock is on YOUR side. Minimize switching or prioritize hazard removal.")
        spike_count = sum(v for k, v in own_conditions.items() if "spikes" in k)
        if spike_count >= 2:
            guidance.append(f"WARNING: {spike_count} layers of Spikes on YOUR side. Switching is very costly.")

    return "\n".join(guidance) if guidance else ""


def _ko_label(min_pct: float, max_pct: float) -> str:
    """Convert damage range into a human-readable KO label."""
    if min_pct >= 100:
        return "GUARANTEED KO"
    if max_pct >= 100:
        return f"possible KO ({min_pct:.0f}–{max_pct:.0f}%)"
    if max_pct >= 50:
        return f"2HKO ({min_pct:.0f}–{max_pct:.0f}%)"
    return f"~{min_pct:.0f}–{max_pct:.0f}% dmg"


def _effectiveness_label(type_mult: float) -> str:
    if type_mult == 0:
        return "IMMUNE"
    if type_mult >= 4:
        return "4x effective"
    if type_mult >= 2:
        return "2x effective"
    if type_mult <= 0.25:
        return "0.25x resisted"
    if type_mult <= 0.5:
        return "0.5x resisted"
    return "neutral"


def build_turn_prompt(view: GeminiBattleView) -> str:
    """Build the per-turn user prompt describing current battle state."""
    from fp.gemini.move_scorer import score_all_actions

    lines = []

    # --- Strategic memory (cross-turn context) ---
    if view.strategic_context is not None:
        mem_block = view.strategic_context.to_prompt_block()
        if mem_block:
            lines.append(mem_block)
            lines.append("")

    lines.append(f"=== TURN {view.turn} ===\n")

    # --- Situational phase guidance ---
    phase_guidance = _build_game_phase_guidance(view)
    if phase_guidance:
        lines.append(phase_guidance)
        lines.append("")

    # --- Battle history ---
    if view.battle_history and view.battle_history != "(No battle history yet)":
        lines.append("BATTLE LOG (previous turns):")
        lines.append(view.battle_history)
        lines.append("")

    # --- Own active slots with raw move data ---
    lines.append("YOUR ACTIVE POKEMON:")
    threat = None
    scored_switches = []

    for slot in view.active_slots:
        pkmn = slot.pokemon
        type_str = "/".join(pkmn.types) if pkmn.types else "???"

        lines.append(
            f"  Slot {slot.slot_index + 1}: {pkmn.species} [{type_str}] "
            f"({pkmn.hp_pct}% HP{', ' + pkmn.status if pkmn.status else ''}) "
            f"Ability:{pkmn.ability} Item:{pkmn.item}"
        )

        if slot.force_switch:
            lines.append("    ** MUST SWITCH (fainted/forced) **")
            continue

        # Get scored data (scores kept internally for fallback, not shown to AI)
        scored_moves, scored_switches, threat = score_all_actions(view, slot.slot_index)

        # Build a lookup for score data by move ID
        score_map = {sm.move_id: sm for sm in scored_moves}

        lines.append("    Moves:")
        for move in slot.legal_moves:
            sm = score_map.get(move.id)
            pp_str = f"PP:{move.pp}/{move.max_pp}"
            disabled = " [DISABLED]" if move.disabled else ""
            move_type = move.move_type.capitalize()

            if sm is None:
                lines.append(f"      - {move.id} [{move_type}] — {pp_str}{disabled}")
                continue

            if sm.is_immune:
                lines.append(f"      - {move.id} [{move_type}]: IMMUNE — do not use{disabled}")
                continue

            type_mult = getattr(sm, "type_mult", 1.0)
            effectiveness = _effectiveness_label(type_mult)
            stab = " | STAB" if getattr(sm, "is_stab", False) else ""
            min_pct, max_pct = sm.damage_pct
            ko_label = _ko_label(min_pct, max_pct) if move.base_power > 0 or move.category != "status" else ""
            dmg_part = f" | {ko_label}" if ko_label else ""

            lines.append(
                f"      - {move.id} [{move_type}] — {effectiveness}{stab}{dmg_part} | {pp_str}{disabled}"
            )

        # Gimmick availability
        gimmicks = []
        if slot.can_terastallize:
            gimmicks.append(f"Terastallize → {slot.can_terastallize} type")
        if slot.can_mega_evo:
            gimmicks.append("Mega Evolve")
        if slot.can_dynamax:
            gimmicks.append("Dynamax")
        if slot.can_z_move:
            gimmicks.append("Z-Move")
        if gimmicks:
            lines.append(f"    Gimmicks available: {', '.join(gimmicks)}")

        if slot.trapped:
            lines.append("    ** TRAPPED — cannot switch **")

    # --- Threat assessment ---
    if threat and threat.best_move_name:
        threat_lines = ["\nTHREAT ASSESSMENT:"]
        speed_str = "OUTSPEEDS you" if threat.they_outspeed else "you outspeed them"
        threat_lines.append(f"  Speed: {speed_str}")
        threat_lines.append(
            f"  Their best move: {threat.best_move_name} [{threat.best_move_type.capitalize()}]"
            f" = ~{threat.estimated_damage_pct:.0f}% damage to you"
        )
        if threat.can_ko and threat.they_outspeed:
            threat_lines.append("  !! CRITICAL: They KO you before you move — switch or use priority !!")
        elif threat.can_ko:
            threat_lines.append("  !! WARNING: They can KO you — KO them first or switch !!")
        elif threat.can_2hko:
            threat_lines.append("  They 2HKO you — don't stay unless you can KO them first")

        if threat.predicted_moves:
            threat_lines.append(f"  Predicted unrevealed moves: {', '.join(threat.predicted_moves[:3])}")
        if threat.predicted_ability:
            threat_lines.append(f"  Predicted ability: {threat.predicted_ability}")
        if threat.predicted_item:
            threat_lines.append(f"  Predicted item: {threat.predicted_item}")

        lines.extend(threat_lines)

    # --- Switch options ---
    if scored_switches:
        lines.append("\n    AVAILABLE SWITCHES:")
        for ss in scored_switches:
            target = next((p for p in view.legal_switch_targets if p.name == ss.pokemon_name), None)
            if target:
                type_str = "/".join(target.types) if target.types else "???"
                lines.append(
                    f"      - {ss.pokemon_name} [{type_str}] {target.hp_pct}% HP — {ss.reason}"
                )

    # --- Opponent active slots ---
    if view.snapshot and view.snapshot.opponent_active_slots:
        lines.append("\nOPPONENT'S ACTIVE POKEMON:")
        for slot_idx, opp in sorted(view.snapshot.opponent_active_slots.items()):
            if opp.fainted:
                lines.append(f"  Slot {slot_idx + 1}: {opp.species} (FAINTED)")
                continue

            type_str = "/".join(opp.types) if opp.types else "???"
            status_str = f", {opp.status}" if opp.status else ""
            tera_str = f", Tera:{opp.tera_type}" if opp.tera_type else ""
            dmax_str = ", DYNAMAXED" if opp.is_dynamaxed else ""
            sub_str = ", HAS SUBSTITUTE" if opp.has_substitute else ""
            lines.append(
                f"  Slot {slot_idx + 1}: {opp.species} [{type_str}] "
                f"({opp.hp_pct}% HP, Lv{opp.level}{status_str}{tera_str}{dmax_str}{sub_str})"
            )

            if opp.revealed_ability:
                lines.append(f"    Ability: {opp.revealed_ability}")
                _ability_immunities = {
                    "lightningrod": "Electric", "voltabsorb": "Electric", "motordrive": "Electric",
                    "waterabsorb": "Water", "stormdrain": "Water", "dryskin": "Water",
                    "flashfire": "Fire", "levitate": "Ground", "sapsipper": "Grass",
                }
                norm_ability = opp.revealed_ability.lower().replace(" ", "").replace("-", "")
                immune_type = _ability_immunities.get(norm_ability)
                if immune_type:
                    lines.append(f"    !! IMMUNE to {immune_type} moves via {opp.revealed_ability} !!")
            if opp.revealed_item:
                lines.append(f"    Item: {opp.revealed_item}")
            if opp.revealed_moves:
                lines.append(f"    Known moves: {', '.join(opp.revealed_moves)}")

            boosts = {k: v for k, v in opp.boosts.items() if v != 0}
            if boosts:
                boost_strs = [f"{k}:{'+' if v > 0 else ''}{v}" for k, v in boosts.items()]
                lines.append(f"    Boosts: {', '.join(boost_strs)}")

    # --- Field conditions ---
    if view.snapshot:
        field = view.snapshot.field_state
        field_lines = []
        if field.weather:
            field_lines.append(f"Weather: {field.weather}")
        if field.terrain:
            field_lines.append(f"Terrain: {field.terrain}")
        if field.trick_room:
            field_lines.append("Trick Room ACTIVE")
        if field.gravity:
            field_lines.append("Gravity ACTIVE")
        if field_lines:
            lines.append("\nFIELD: " + " | ".join(field_lines))

        if view.snapshot.own_side_conditions:
            own_conds = [f"{k}(x{v})" if v > 1 else k
                         for k, v in view.snapshot.own_side_conditions.items()]
            lines.append(f"YOUR SIDE: {', '.join(own_conds)}")
        if view.snapshot.opp_side_conditions:
            opp_conds = [f"{k}(x{v})" if v > 1 else k
                         for k, v in view.snapshot.opp_side_conditions.items()]
            lines.append(f"OPPONENT'S SIDE: {', '.join(opp_conds)}")

        if view.snapshot.opp_gimmicks_used:
            lines.append(f"OPPONENT GIMMICKS USED: {', '.join(view.snapshot.opp_gimmicks_used)}")
        if view.snapshot.own_gimmicks_used:
            lines.append(f"YOUR GIMMICKS USED: {', '.join(view.snapshot.own_gimmicks_used)}")

    # --- Full team summary ---
    lines.append("\nYOUR FULL TEAM:")
    for p in view.own_team:
        type_str = "/".join(p.types) if p.types else "???"
        status_info = f", {p.status}" if p.status and p.status != "fnt" else ""
        if p.hp <= 0:
            lines.append(f"  {p.name} [{type_str}]: FAINTED")
        else:
            spe_str = ""
            if p.stats:
                spe = p.stats.get("spe", p.stats.get("speed"))
                if spe:
                    spe_str = f" Spe:{spe}"
            lines.append(
                f"  {p.name} [{type_str}]: {p.hp_pct}% HP{status_info} "
                f"Moves: {', '.join(p.moves[:4])}{spe_str}"
            )

    # --- Opponent profile (if we've learned something) ---
    if view.opponent_profile is not None:
        profile_block = view.opponent_profile.to_prompt_block()
        if profile_block:
            lines.append("")
            lines.append(profile_block)

    return "\n".join(lines)


# Type effectiveness chart (attacker → defender multipliers)
_TYPE_CHART = {
    "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0, "dark": 2, "steel": 2, "fairy": 0.5},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
    "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
    "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
    "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
    "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5},
}


def _get_effectiveness_hint(move_type: str, view: GeminiBattleView) -> str:
    """Compute type effectiveness of a move against the current opponent(s)."""
    if not view.snapshot or not view.snapshot.opponent_active_slots:
        return ""

    hints = []
    for _, opp in sorted(view.snapshot.opponent_active_slots.items()):
        if opp.fainted or not opp.types:
            continue
        multiplier = 1.0
        chart = _TYPE_CHART.get(move_type.lower(), {})
        for def_type in opp.types:
            multiplier *= chart.get(def_type.lower(), 1.0)
        if multiplier >= 4:
            hints.append(f" [4x vs {opp.species}]")
        elif multiplier >= 2:
            hints.append(f" [2x vs {opp.species}]")
        elif multiplier <= 0:
            hints.append(f" [IMMUNE {opp.species}]")
        elif multiplier <= 0.25:
            hints.append(f" [0.25x vs {opp.species}]")
        elif multiplier <= 0.5:
            hints.append(f" [0.5x vs {opp.species}]")

    return "".join(hints)


def _compute_multiplier(move_type: str, view: GeminiBattleView) -> float:
    """Return the raw effectiveness multiplier of a move type vs the opponent's active."""
    if not view.snapshot or not view.snapshot.opponent_active_slots:
        return 1.0
    for _, opp in sorted(view.snapshot.opponent_active_slots.items()):
        if opp.fainted or not opp.types:
            continue
        multiplier = 1.0
        chart = _TYPE_CHART.get(move_type.lower(), {})
        for def_type in opp.types:
            multiplier *= chart.get(def_type.lower(), 1.0)
        return multiplier
    return 1.0


# Defensive type chart: for each defending type, what attack types are SE against it?
_DEFENSIVE_WEAKNESSES = {
    "normal": ["fighting"],
    "fire": ["water", "ground", "rock"],
    "water": ["electric", "grass"],
    "electric": ["ground"],
    "grass": ["fire", "ice", "poison", "flying", "bug"],
    "ice": ["fire", "fighting", "rock", "steel"],
    "fighting": ["flying", "psychic", "fairy"],
    "poison": ["ground", "psychic"],
    "ground": ["water", "grass", "ice"],
    "flying": ["electric", "ice", "rock"],
    "psychic": ["bug", "ghost", "dark"],
    "bug": ["fire", "flying", "rock"],
    "rock": ["water", "grass", "fighting", "ground", "steel"],
    "ghost": ["ghost", "dark"],
    "dragon": ["ice", "dragon", "fairy"],
    "dark": ["fighting", "bug", "fairy"],
    "steel": ["fire", "fighting", "ground"],
    "fairy": ["poison", "steel"],
}

_DEFENSIVE_RESISTANCES = {
    "normal": [],
    "fire": ["fire", "grass", "ice", "bug", "steel", "fairy"],
    "water": ["fire", "water", "ice", "steel"],
    "electric": ["electric", "flying", "steel"],
    "grass": ["water", "electric", "grass", "ground"],
    "ice": ["ice"],
    "fighting": ["bug", "rock", "dark"],
    "poison": ["grass", "fighting", "poison", "bug", "fairy"],
    "ground": ["poison", "rock"],
    "flying": ["grass", "fighting", "bug"],
    "psychic": ["fighting", "psychic"],
    "bug": ["grass", "fighting", "ground"],
    "rock": ["normal", "fire", "poison", "flying"],
    "ghost": ["poison", "bug"],
    "dragon": ["fire", "water", "electric", "grass"],
    "dark": ["ghost", "dark"],
    "steel": ["normal", "grass", "ice", "flying", "psychic", "bug", "rock", "dragon", "steel", "fairy"],
    "fairy": ["fighting", "bug", "dark"],
}

_DEFENSIVE_IMMUNITIES = {
    "normal": ["ghost"],
    "flying": ["ground"],
    "ground": ["electric"],
    "ghost": ["normal", "fighting"],
    "dark": ["psychic"],
    "steel": ["poison"],
    "fairy": ["dragon"],
}


def _get_defensive_matchup(own_types: list[str], view: GeminiBattleView) -> str:
    """Compute how well own_types handle the opponent's likely STAB types."""
    if not view.snapshot or not view.snapshot.opponent_active_slots:
        return ""

    for _, opp in sorted(view.snapshot.opponent_active_slots.items()):
        if opp.fainted or not opp.types:
            continue

        opp_stab_types = [t.lower() for t in opp.types]
        own_types_lower = [t.lower() for t in own_types]

        immunities = []
        resists = []
        weak_to = []

        for stab in opp_stab_types:
            # Check each defending type
            is_immune = False
            multiplier = 1.0
            for dt in own_types_lower:
                if stab in _DEFENSIVE_IMMUNITIES.get(dt, []):
                    is_immune = True
                    break
                if stab in _DEFENSIVE_WEAKNESSES.get(dt, []):
                    multiplier *= 2.0
                elif stab in _DEFENSIVE_RESISTANCES.get(dt, []):
                    multiplier *= 0.5

            if is_immune:
                immunities.append(stab.capitalize())
            elif multiplier >= 2:
                weak_to.append(f"{stab.capitalize()}({multiplier:.0f}x)")
            elif multiplier <= 0.5:
                resists.append(stab.capitalize())

        parts = []
        if immunities:
            parts.append(f"IMMUNE to {', '.join(immunities)}")
        if resists:
            parts.append(f"resists {', '.join(resists)}")
        if weak_to:
            parts.append(f"WEAK to {', '.join(weak_to)}")

        if not parts:
            return "neutral vs opponent STAB"
        return " | ".join(parts)

    return ""


def build_team_preview_prompt(view: GeminiBattleView) -> str:
    """Build the team preview prompt for lead selection."""
    lines = ["=== TEAM PREVIEW ===\n"]

    lines.append(f"Format: {view.format_info.format_name} ({view.format_info.gametype})")
    lines.append(f"Pick {view.pick_count} Pokemon. The first will be your lead.\n")

    lines.append("YOUR TEAM:")
    for i, p in enumerate(view.own_team):
        type_str = "/".join(p.types) if p.types else "???"
        lines.append(
            f"  {i + 1}. {p.species} [{type_str}] Ability:{p.ability} Item:{p.item} "
            f"Tera:{p.tera_type or 'N/A'} "
            f"Moves: {', '.join(p.moves[:4])}"
        )

    if view.opponent_preview_team:
        lines.append("\nOPPONENT'S TEAM:")
        for i, name in enumerate(view.opponent_preview_team):
            lines.append(f"  {i + 1}. {name}")

    lines.append(
        f"\nChoose the optimal {view.pick_count} Pokemon and lead order "
        f"for this matchup using the choose_leads tool."
    )
    return "\n".join(lines)
