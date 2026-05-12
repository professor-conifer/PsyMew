"""Pokémon Showdown server presets + format catalog for the GUI.

A `ServerPreset` bundles everything that has to move together when you
change Showdown servers: the websocket URL, the auth endpoint
(`action.php`), and the list of formats the server actually offers.

The Battle tab subscribes to `PS_WEBSOCKET_URI` changes and re-filters
its format dropdown from `formats_for_server()` so users never see
gen8monotype when they're connected to a server that only runs gen9ou.
"""

from __future__ import annotations

from dataclasses import dataclass

# Full Showdown format catalog — used for the official server and any
# custom URL the user types in. Curated roughly "most common first".
SHOWDOWN_FORMATS: list[str] = [
    # --- Gen 9 random / draft (no team picker needed) -----------------
    "gen9randombattle",
    "gen9randombattleblitz",
    "gen9randomdoublesbattle",
    "gen9hackmonscup",
    "gen9challengecup1v1",
    "gen9challengecup2v2",
    "gen9battlefactory",
    "gen9bssfactory",
    # --- Gen 9 singles tiers -----------------------------------------
    "gen9ou",
    "gen9uu",
    "gen9ru",
    "gen9nu",
    "gen9pu",
    "gen9zu",
    "gen9lc",
    "gen9ubers",
    "gen9ubersuu",
    "gen9anythinggoes",
    "gen9monotype",
    "gen9nationaldex",
    "gen9nationaldexuu",
    "gen9nationaldexru",
    "gen9nationaldexmonotype",
    "gen9nationaldexag",
    "gen9nationaldexubers",
    "gen9doublesou",
    "gen9doublesubers",
    "gen9doublesuu",
    "gen91v1",
    "gen92v2doubles",
    # --- Gen 9 VGC / official ----------------------------------------
    "gen9vgc2024regg",
    "gen9vgc2024regh",
    "gen9vgc2024regi",
    "gen9vgc2024reggbo3",
    "gen9battlestadiumsingles",
    "gen9battlestadiumdoubles",
    "gen9bdspou",
    # --- Older gens singles ------------------------------------------
    "gen8ou",
    "gen8nationaldex",
    "gen8nationaldexag",
    "gen8nationaldexmonotype",
    "gen8doublesou",
    "gen8randombattle",
    "gen8bdspou",
    "gen8bdsprandombattle",
    "gen7ou",
    "gen7uu",
    "gen7lc",
    "gen7anythinggoes",
    "gen7monotype",
    "gen7randombattle",
    "gen7letsgoou",
    "gen7letsgorandombattle",
    "gen6ou",
    "gen6uu",
    "gen6ubers",
    "gen6randombattle",
    "gen5ou",
    "gen5uu",
    "gen5randombattle",
    "gen4ou",
    "gen4uu",
    "gen4randombattle",
    "gen3ou",
    "gen3randombattle",
    "gen2ou",
    "gen2randombattle",
    "gen1ou",
    "gen1ubers",
    "gen1randombattle",
]


@dataclass(frozen=True)
class ServerPreset:
    name: str
    websocket: str
    # `login` is the action.php URL the bot POSTs credentials to. None
    # means "use Showdown's default (play.pokemonshowdown.com)".
    login: str | None = None
    # `formats` is the allowed list for this server. None means the full
    # SHOWDOWN_FORMATS catalog (i.e. unrestricted).
    formats: list[str] | None = None
    # Notes shown in the GUI under the preset (1–2 sentences).
    blurb: str = ""
    # Optional links shown as buttons in the Advanced tab.
    leaderboard_url: str = ""
    replays_url: str = ""
    docs_url: str = ""


SERVER_PRESETS: list[ServerPreset] = [
    ServerPreset(
        name="Official Pokémon Showdown",
        websocket="wss://sim3.psim.us/showdown/websocket",
        login=None,
        formats=None,
        blurb="The main public Showdown server. All formats available.",
        docs_url="https://pokemonshowdown.com/",
    ),
    ServerPreset(
        name="Smogtours (backup)",
        websocket="wss://smogtours.psim.us/showdown/websocket",
        login=None,
        formats=None,
        blurb="Smogon's tournament-overflow server. Same logins as the main server.",
        docs_url="https://www.smogon.com/forums/forums/tournaments.34/",
    ),
    ServerPreset(
        name="PokéAgent Challenge",
        websocket="wss://battling.pokeagentchallenge.com/showdown/websocket",
        login="https://battling.pokeagentchallenge.com/action.php?",
        formats=[
            "gen9ou",
            "gen9vgc2024regi",
            "gen4ou",
            "gen3ou",
            "gen2ou",
            "gen1ou",
        ],
        blurb=(
            "Independent AI-vs-AI ladder (NeurIPS 2025 benchmark, ongoing). "
            "Use the Long Timer option in the lobby for LLM agents."
        ),
        leaderboard_url="https://battling.pokeagentchallenge.com/ladder",
        replays_url="https://replays.pokeagentchallenge.com",
        docs_url="https://pokeagentchallenge.com/battling.html",
    ),
    ServerPreset(
        name="Custom…",
        websocket="",
        login=None,
        formats=None,
        blurb="Paste your own websocket URL and (optionally) action.php URL below.",
    ),
]


def preset_for_websocket(websocket_uri: str) -> ServerPreset:
    """Reverse-lookup which preset a saved websocket URI belongs to.

    Falls back to the Custom preset if no built-in match is found.
    """
    for preset in SERVER_PRESETS:
        if preset.websocket and preset.websocket == websocket_uri:
            return preset
    return next(p for p in SERVER_PRESETS if p.name == "Custom…")


def formats_for_server(websocket_uri: str) -> list[str]:
    """Return the format-dropdown contents to show for a given server URL."""
    preset = preset_for_websocket(websocket_uri)
    if preset.formats is not None:
        return preset.formats
    return SHOWDOWN_FORMATS


def is_teamless_format(format_name: str) -> bool:
    """True if the format generates a team for you (no team picker needed)."""
    if not format_name:
        return False
    lower = format_name.lower()
    return any(
        marker in lower
        for marker in ("random", "battlefactory", "bssfactory", "hackmonscup", "challengecup")
    )
