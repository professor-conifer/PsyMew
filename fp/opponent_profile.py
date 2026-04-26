"""Opponent behavior profiling for the AI battle engine.

Builds a lightweight model of the opponent's team archetype and tendencies
from observed battle messages. Injected into the AI's turn prompt so it can
make better predictions and force harder decisions.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TeamArchetype(Enum):
    UNKNOWN = "unknown"
    HYPER_OFFENSE = "hyper offense"
    BALANCE = "balance"
    STALL = "stall"
    TRICK_ROOM = "trick room"
    WEATHER = "weather"
    TAILWIND = "tailwind"
    HAZARD_STACK = "hazard stack"
    WEBS = "sticky webs"


_SETUP_MOVES = {
    "trickroom", "tailwind", "stealthrock", "spikes", "toxicspikes", "stickyweb",
    "raindance", "sunnyday", "sandstorm", "snowscape", "hail",
    "calmmind", "swordsdance", "dragondance", "nastyplot", "quiverdance",
    "shellsmash", "bulkup", "irondefense", "geomancy", "shiftgear",
}

_RECOVERY_MOVES = {"recover", "roost", "softboiled", "moonlight", "morningsun",
                   "synthesis", "slackoff", "milkdrink", "shoreup", "rest", "wish"}

_STATUS_MOVES = {"toxic", "willowisp", "thunderwave", "nuzzle", "glare", "stunspore",
                 "yawn", "sleeppowder", "spore", "hypnosis", "leechseed"}


@dataclass
class OpponentProfile:
    """Accumulated knowledge about the opponent's team strategy and tendencies."""

    archetype: TeamArchetype = TeamArchetype.UNKNOWN
    switch_count: int = 0
    protect_count: int = 0
    turns_observed: int = 0
    turns_they_switched_under_pressure: int = 0
    status_moves_used: list[str] = field(default_factory=list)
    setup_moves_used: list[str] = field(default_factory=list)
    revealed_strategy_notes: list[str] = field(default_factory=list)

    def update_from_msg_list(self, msg_list: list, prev_turn: int) -> None:
        """Scan new Showdown protocol messages and update tracking."""
        for line in msg_list:
            if not isinstance(line, str):
                continue

            parts = line.split("|")
            if len(parts) < 2:
                continue

            msg_type = parts[1] if len(parts) > 1 else ""

            # Opponent switches
            if msg_type == "switch" and len(parts) > 2:
                slot = parts[2] if len(parts) > 2 else ""
                if slot.startswith("p2"):
                    self.switch_count += 1

            # Opponent moves
            if msg_type == "move" and len(parts) > 3:
                slot = parts[2] if len(parts) > 2 else ""
                if not slot.startswith("p2"):
                    continue

                move_id = parts[3].lower().replace(" ", "").replace("-", "") if len(parts) > 3 else ""

                if move_id in ("protect", "detect", "kingsshield", "spikyshield",
                               "banefulbunker", "silktrap", "obstruct"):
                    self.protect_count += 1
                    if self.protect_count == 2:
                        self._add_note("opponent is using Protect frequently — predict and punish")

                if move_id == "trickroom":
                    if self.archetype == TeamArchetype.UNKNOWN:
                        self.archetype = TeamArchetype.TRICK_ROOM
                    self._add_note("opponent set Trick Room")

                if move_id == "tailwind":
                    if self.archetype == TeamArchetype.UNKNOWN:
                        self.archetype = TeamArchetype.TAILWIND
                    self._add_note("opponent set Tailwind")

                if move_id in ("raindance", "sunnyday", "sandstorm", "snowscape", "hail"):
                    if self.archetype == TeamArchetype.UNKNOWN:
                        self.archetype = TeamArchetype.WEATHER
                    self._add_note(f"opponent set weather: {move_id}")

                if move_id == "stickyweb":
                    if self.archetype == TeamArchetype.UNKNOWN:
                        self.archetype = TeamArchetype.WEBS
                    self._add_note("opponent used Sticky Web")

                if move_id in _STATUS_MOVES and move_id not in self.status_moves_used:
                    self.status_moves_used.append(move_id)
                    if len(self.status_moves_used) >= 2 and self.archetype == TeamArchetype.UNKNOWN:
                        self.archetype = TeamArchetype.STALL

                if move_id in _SETUP_MOVES and move_id not in self.setup_moves_used:
                    self.setup_moves_used.append(move_id)

            # Infer stall if opponent uses recovery
            if msg_type == "move" and len(parts) > 3:
                slot = parts[2] if len(parts) > 2 else ""
                move_id = parts[3].lower().replace(" ", "").replace("-", "") if len(parts) > 3 else ""
                if slot.startswith("p2") and move_id in _RECOVERY_MOVES:
                    if self.archetype == TeamArchetype.UNKNOWN and self.status_moves_used:
                        self.archetype = TeamArchetype.STALL
                        self._add_note("opponent using recovery + status — likely stall")

        self.turns_observed = max(self.turns_observed, prev_turn)

        # Infer HO from rapid KOs and no setup
        if (self.turns_observed >= 5 and self.switch_count <= 1
                and not self.setup_moves_used and not self.status_moves_used
                and self.archetype == TeamArchetype.UNKNOWN):
            self.archetype = TeamArchetype.HYPER_OFFENSE
            self._add_note("opponent shows hyper offense patterns — minimal switching, no setup")

    def _add_note(self, note: str) -> None:
        if note not in self.revealed_strategy_notes:
            self.revealed_strategy_notes.append(note)
            logger.debug("OpponentProfile: %s", note)

    def to_prompt_block(self) -> str:
        """Render as a prompt section at the bottom of the turn prompt."""
        if self.archetype == TeamArchetype.UNKNOWN and not self.revealed_strategy_notes:
            return ""

        lines = [f"OPPONENT PROFILE (Archetype: {self.archetype.value}):"]

        if self.switch_count > 0 and self.turns_observed > 0:
            switch_rate = self.switch_count / max(self.turns_observed, 1)
            if switch_rate > 0.3:
                lines.append(f"  Pivots frequently ({self.switch_count} switches in ~{self.turns_observed} turns) — predict switches")
            elif switch_rate < 0.1 and self.turns_observed > 4:
                lines.append(f"  Rarely switches — plays straight up, punish by staying in")

        if self.protect_count > 1:
            lines.append(f"  Used Protect {self.protect_count}x — predict and use a non-target turn or switch")

        for note in self.revealed_strategy_notes[-3:]:
            lines.append(f"  Observed: {note}")

        return "\n".join(lines)
