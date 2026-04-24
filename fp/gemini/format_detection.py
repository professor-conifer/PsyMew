"""Detect format metadata from the Pokemon Showdown format string and request JSON."""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FormatInfo:
    """Immutable snapshot of format metadata, attached to battle.format_info."""

    gametype: str  # "singles", "doubles", "triples"
    gen: int  # 1-9
    format_name: str  # raw format string e.g. "gen9vgc2025regg"
    is_random: bool
    is_vgc: bool
    is_battle_factory: bool
    has_team_preview: bool
    slot_count: int  # 1, 2, or 3
    pick_count: int  # how many pokemon to pick at team preview (4 for VGC, else team size)


_NO_TEAM_PREVIEW_GENS = {"gen1", "gen2", "gen3", "gen4"}

_GEN_RE = re.compile(r"gen(\d)")


def _detect_gen(format_name: str) -> int:
    m = _GEN_RE.search(format_name)
    if m:
        return int(m.group(1))
    return 9  # default to latest gen


def _detect_gametype_from_request(request_json: Optional[dict]) -> Optional[str]:
    """Infer gametype from the number of active slots in the request JSON."""
    if request_json is None:
        return None
    active = request_json.get("active")
    if active is None:
        return None
    n = len(active)
    if n >= 3:
        return "triples"
    elif n == 2:
        return "doubles"
    else:
        return "singles"


def _detect_gametype_from_string(format_name: str) -> str:
    """Heuristic gametype detection from the format name."""
    lower = format_name.lower()
    if "triples" in lower or "triple" in lower:
        return "triples"
    if any(kw in lower for kw in ("doubles", "vgc", "doublesou", "doublesuu",
                                   "doublesrandombattle", "2v2")):
        return "doubles"
    return "singles"


def _detect_is_vgc(format_name: str) -> bool:
    lower = format_name.lower()
    return "vgc" in lower or "battlestadium" in lower or "bss" in lower


def _detect_is_battle_factory(format_name: str) -> bool:
    return "battlefactory" in format_name.lower()


def _detect_is_random(format_name: str) -> bool:
    return "random" in format_name.lower()


def _slot_count(gametype: str) -> int:
    return {"singles": 1, "doubles": 2, "triples": 3}.get(gametype, 1)


def _detect_pick_count(format_name: str, is_vgc: bool, request_json: Optional[dict]) -> int:
    """How many Pokemon to pick at team preview."""
    if is_vgc:
        return 4  # VGC: bring 6, pick 4

    # BSS is also pick 3 from 6
    if "battlestadium" in format_name.lower() or "bss" in format_name.lower():
        return 3

    # For standard formats, pick count == team size
    if request_json and "side" in request_json:
        team = request_json["side"].get("pokemon", [])
        return len(team)

    return 6  # default


def detect_format_info(pokemon_format: str, request_json: Optional[dict] = None) -> FormatInfo:
    """Build a FormatInfo from the format string and optionally the first request JSON.

    Parameters
    ----------
    pokemon_format : str
        The Pokemon Showdown format string, e.g. "gen9randombattle".
    request_json : dict | None
        The first request JSON received from the server (may be None during
        team preview or early init).
    """
    gen = _detect_gen(pokemon_format)
    generation_str = f"gen{gen}"

    # Gametype: prefer request JSON (authoritative), fallback to string heuristic
    gametype = _detect_gametype_from_request(request_json) or _detect_gametype_from_string(pokemon_format)

    is_random = _detect_is_random(pokemon_format)
    is_vgc = _detect_is_vgc(pokemon_format)
    is_battle_factory = _detect_is_battle_factory(pokemon_format)
    has_team_preview = generation_str not in _NO_TEAM_PREVIEW_GENS and not is_random
    slot_count = _slot_count(gametype)
    pick_count = _detect_pick_count(pokemon_format, is_vgc, request_json)

    info = FormatInfo(
        gametype=gametype,
        gen=gen,
        format_name=pokemon_format,
        is_random=is_random,
        is_vgc=is_vgc,
        is_battle_factory=is_battle_factory,
        has_team_preview=has_team_preview,
        slot_count=slot_count,
        pick_count=pick_count,
    )
    logger.info("Detected format info: %s", info)
    return info
