"""Hardcoded rule cards for Pokemon Showdown formats.

These serve as a baseline for Gemini's system prompt. At battle start,
format_research.py verifies them against live Google Search results and
patches any discrepancies.

Also exports move targeting semantics used by tools.py.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Move target type → valid Showdown target indices
# In doubles/triples, Showdown target indices are:
#   -1 = opponent slot 1 (left), -2 = opponent slot 2 (right)
#   +1 = ally slot 1 (self/left), +2 = ally slot 2 (right)
#   For triples: -3, -2, -1, 1, 2, 3
# In singles, target is always implicit (no index needed).
# ---------------------------------------------------------------------------

# Mapping from moves.json "target" field to list of valid Showdown target indices
# These are for DOUBLES; triples extends similarly.
DOUBLES_TARGET_MAP = {
    "normal": [-1, -2, 2],       # any adjacent (opponents + ally)
    "any": [-1, -2, 2],          # same as normal for most purposes
    "adjacentFoe": [-1, -2],     # adjacent opponents only
    "allAdjacentFoes": None,     # no target selection needed (hits all foes)
    "allAdjacent": None,         # hits all adjacent (foes + ally)
    "adjacentAlly": [2],         # partner only
    "adjacentAllyOrSelf": [1, 2],  # self or partner
    "self": None,                # no target needed
    "all": None,                 # entire field
    "allySide": None,            # own side
    "foeSide": None,             # opponent's side
    "allyTeam": None,            # own team
    "randomNormal": None,        # random adjacent
    "scripted": None,            # auto-targeted
}

SINGLES_TARGET_MAP = {
    # In singles, no target index is ever needed
    k: None for k in DOUBLES_TARGET_MAP
}

TRIPLES_TARGET_MAP = {
    "normal": [-1, -2, -3, 2, 3],
    "any": [-1, -2, -3, 2, 3],
    "adjacentFoe": [-1, -2, -3],
    "allAdjacentFoes": None,
    "allAdjacent": None,
    "adjacentAlly": [2, 3],
    "adjacentAllyOrSelf": [1, 2, 3],
    "self": None,
    "all": None,
    "allySide": None,
    "foeSide": None,
    "allyTeam": None,
    "randomNormal": None,
    "scripted": None,
}


def get_move_target_semantics(gen: int, gametype: str) -> dict[str, Optional[list[int]]]:
    """Return the target mapping for the given gametype."""
    if gametype == "triples":
        return TRIPLES_TARGET_MAP
    elif gametype == "doubles":
        return DOUBLES_TARGET_MAP
    else:
        return SINGLES_TARGET_MAP


# ---------------------------------------------------------------------------
# Rule cards — keyed by (gen, format_substring)
# Lookup: iterate entries, pick first where format_substring is in format_name
# ---------------------------------------------------------------------------

_RULE_CARDS: dict[tuple[int, str], str] = {
    # --- Gen 9 ---
    (9, "randombattle"): """Gen 9 Random Battle (Singles)
- Gametype: Singles (1v1 active)
- Team: 6 random Pokemon, random sets, random levels
- No team preview
- Gimmicks: Terastallize (once per battle), no Dynamax, no Mega, no Z-Moves
- Clauses: Sleep Clause, HP Percentage Mod, Cancel Mod
- Pokemon may have any legal ability/item/moves for their set
- Levels vary (typically 70-100) based on Pokemon viability
""",

    (9, "vgc"): """Gen 9 VGC (Doubles)
- Gametype: Doubles (2v2 active)
- Team: Bring 6, pick 4 at team preview
- Gimmicks: Terastallize (once per battle), no Dynamax, no Mega, no Z-Moves
- Move targeting: must specify target for single-target moves (-1, -2 for foes, 2 for ally)
- Spread moves hit both opponents (reduced damage to 75%)
- Clauses: Species Clause, Item Clause, check current regulation for restricted/banned Pokemon
- Current regulations may restrict certain legendaries — verify via search
""",

    (9, "ou"): """Gen 9 OU (Singles)
- Gametype: Singles (1v1 active)
- Team: 6 Pokemon, team preview
- Gimmicks: Terastallize (once per battle), no Dynamax, no Mega, no Z-Moves
- Clauses: Sleep Clause, Species Clause, OHKO Clause, Evasion Clause, Moody Clause
- Various Pokemon/abilities/moves are banned (Ubers) — verify current banlist via search
""",

    (9, "doublesou"): """Gen 9 Doubles OU
- Gametype: Doubles (2v2 active)
- Team: 6 Pokemon, team preview, all 6 battle
- Gimmicks: Terastallize (once per battle), no Dynamax, no Mega, no Z-Moves
- Move targeting: specify target for single-target moves
- Spread moves: 75% damage to each target
- Clauses: Species Clause, OHKO Clause, Evasion Clause, Sleep Clause
""",

    (9, "battlefactory"): """Gen 9 Battle Factory
- Gametype: Singles (1v1 active)
- Team: 6 random rental Pokemon from a tier pool, team preview
- Gimmicks: Terastallize (once per battle)
- Clauses: Sleep Clause, Species Clause, OHKO Clause
- Sets are predefined rental sets, not custom
""",

    (9, "doublesrandombattle"): """Gen 9 Random Doubles Battle
- Gametype: Doubles (2v2 active)
- Team: 6 random Pokemon, random sets, random levels
- No team preview
- Gimmicks: Terastallize (once per battle), no Dynamax, no Mega, no Z-Moves
- Move targeting: must specify target for single-target moves
- Spread moves: 75% damage
""",

    (9, "nationaldex"): """Gen 9 National Dex (Singles)
- Gametype: Singles (1v1 active)
- Team: 6 Pokemon, team preview
- Gimmicks: Terastallize, Mega Evolution (with Mega Stone), Z-Moves (with Z-Crystal)
  - Only one gimmick per battle (tera OR mega OR z-move)
- Clauses: Sleep Clause, Species Clause, OHKO Clause, Evasion Clause, Moody Clause
- Includes Pokemon/moves/abilities from all gens in National Dex format
""",

    (9, "1v1"): """Gen 9 1v1
- Gametype: Singles (1v1 active)
- Team: Bring 3, pick 1 at team preview
- Gimmicks: Terastallize
- Single Pokemon battle — no switching
- Clauses: Species Clause, Accuracy Moves Clause, Sleep Clause, OHKO Clause
""",

    (9, "lc"): """Gen 9 Little Cup (Singles)
- Gametype: Singles (1v1 active)
- Team: 6 Pokemon, all must be first-stage/baby Pokemon at level 5
- Gimmicks: Terastallize
- Clauses: Sleep Clause, Species Clause, OHKO Clause, Evasion Clause
""",

    (9, "ag"): """Gen 9 Anything Goes (Singles)
- Gametype: Singles (1v1 active)
- Team: 6 Pokemon, team preview
- Gimmicks: Terastallize
- No bans — any Pokemon, move, ability, or item is legal
- No clauses except Endless Battle Clause
""",

    (9, "balancedhackmons"): """Gen 9 Balanced Hackmons (Singles)
- Gametype: Singles (1v1 active)
- Team: 6 Pokemon with any legal moves/abilities
- Gimmicks: Terastallize
- Any ability on any Pokemon, any move on any Pokemon
- Bans: specific OP combos — verify via search
""",

    (9, "metronome"): """Gen 9 Metronome Battle
- Gametype: Doubles (2v2 active)
- Team: 6 Pokemon, all moves are Metronome
- Every turn, each Pokemon uses Metronome (calls random move)
- Gimmicks: None usually
""",

    # --- Gen 8 ---
    (8, "randombattle"): """Gen 8 Random Battle (Singles)
- Gametype: Singles
- Gimmicks: Dynamax (once per battle, lasts 3 turns), no Mega, no Z-Moves, no Tera
- Max Moves replace regular moves during Dynamax
- Sleep Clause, HP Percentage Mod
""",

    (8, "ou"): """Gen 8 OU (Singles)
- Gametype: Singles
- Gimmicks: Dynamax (banned in Gen 8 OU!), no Mega, no Z-Moves, no Tera
- Note: Dynamax is BANNED in Gen 8 OU Smogon tier
- Clauses: Sleep Clause, Species Clause, OHKO Clause, Evasion Clause, Dynamax Clause
""",

    (8, "vgc"): """Gen 8 VGC (Doubles)
- Gametype: Doubles
- Gimmicks: Dynamax (once per battle, 3 turns), no Mega, no Z-Moves, no Tera
- Max Moves replace moves; targeting still required
- Bring 6, pick 4
""",

    # --- Gen 7 ---
    (7, "randombattle"): """Gen 7 Random Battle (Singles)
- Gametype: Singles
- Gimmicks: Mega Evolution (with Mega Stone, once per battle), Z-Moves (with Z-Crystal, once per battle)
- No Dynamax, no Tera
""",

    (7, "ou"): """Gen 7 OU (Singles)
- Gametype: Singles
- Gimmicks: Mega Evolution, Z-Moves
- Clauses: Sleep Clause, Species Clause, OHKO Clause, Evasion Clause
""",

    # --- Gen 6 ---
    (6, "randombattle"): """Gen 6 Random Battle (Singles)
- Gametype: Singles
- Gimmicks: Mega Evolution only
- No Z-Moves, no Dynamax, no Tera
""",

    (6, "ou"): """Gen 6 OU (Singles)
- Gametype: Singles
- Gimmicks: Mega Evolution
- Clauses: Sleep Clause, Species Clause, OHKO Clause, Evasion Clause
""",

    # --- Gen 4 ---
    (4, "ou"): """Gen 4 OU (Singles)
- Gametype: Singles
- No gimmicks (no Mega, no Z, no Dynamax, no Tera)
- No team preview (Gen 4 and earlier)
- Clauses: Sleep Clause, Species Clause, OHKO Clause, Evasion Clause
- Stealth Rock is extremely dominant
""",

    (4, "randombattle"): """Gen 4 Random Battle (Singles)
- Gametype: Singles
- No gimmicks, no team preview
""",
}

# Generic fallback cards per generation
_GEN_FALLBACK: dict[int, str] = {
    9: """Gen 9 Format
- Gimmicks: Terastallize (once per battle)
- No Dynamax, no Mega, no Z-Moves (unless National Dex)
""",
    8: """Gen 8 Format
- Gimmicks: Dynamax (once per battle, 3 turns) — may be banned in Smogon tiers
- No Mega, no Z-Moves, no Tera
""",
    7: """Gen 7 Format
- Gimmicks: Mega Evolution (once per battle), Z-Moves (once per battle)
- No Dynamax, no Tera
""",
    6: """Gen 6 Format
- Gimmicks: Mega Evolution (once per battle)
- No Z-Moves, no Dynamax, no Tera
""",
    5: """Gen 5 Format
- No gimmicks
""",
    4: """Gen 4 Format
- No gimmicks, no team preview
""",
    3: """Gen 3 Format
- No gimmicks, no team preview, no abilities on some Pokemon
""",
    2: """Gen 2 Format
- No gimmicks, no team preview, no items hold effects are limited
""",
    1: """Gen 1 Format
- No gimmicks, no team preview, no held items, no abilities
- Special stat (no split), 1/256 miss glitch, freeze is permanent
""",
}


def get_rule_card(gen: int, format_name: str) -> str:
    """Look up the best-matching rule card for this format.

    Returns the card text (never None — falls back to a generic gen card).
    """
    lower = format_name.lower()

    # Try exact (gen, substring) matches in priority order
    for (card_gen, card_key), card_text in _RULE_CARDS.items():
        if card_gen == gen and card_key in lower:
            return card_text.strip()

    # Fallback to generic gen card
    return _GEN_FALLBACK.get(gen, f"Gen {gen} format — no specific rules card available. Use your knowledge.").strip()
