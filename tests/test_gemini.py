"""Unit tests for the fp.gemini module.

These tests exercise the logic of format detection, rule cards, msg parsing,
view building, tool schema construction, decision parsing, and prompt generation
without requiring a live Gemini API key.
"""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

from fp.gemini.errors import GeminiAuthError, GeminiInvalidChoice, GeminiTimeout
from fp.gemini.format_detection import detect_format_info, FormatInfo
from fp.gemini.format_rules import get_rule_card, get_move_target_semantics
from fp.gemini.msg_parser import parse_msg_list, ParsedSnapshot, OppSlotInfo
from fp.gemini.view import GeminiBattleView, LegalMove, OwnPokemon, ActiveSlotView
from fp.gemini.prompt import build_system_prompt, build_turn_prompt, build_team_preview_prompt
from fp.gemini.decision import _parse_action_part, _parse_team_preview
from fp.gemini.tutor import _sanitize_reply


# =========================================================================
# FormatDetection
# =========================================================================

class TestFormatDetection(unittest.TestCase):
    def test_gen9_random_battle(self):
        info = detect_format_info("gen9randombattle")
        self.assertEqual(info.gen, 9)
        self.assertEqual(info.gametype, "singles")
        self.assertTrue(info.is_random)
        self.assertFalse(info.is_vgc)
        self.assertFalse(info.has_team_preview)
        self.assertEqual(info.slot_count, 1)

    def test_gen9_vgc(self):
        info = detect_format_info("gen9vgc2025regg")
        self.assertEqual(info.gen, 9)
        self.assertEqual(info.gametype, "doubles")
        self.assertTrue(info.is_vgc)
        self.assertTrue(info.has_team_preview)
        self.assertEqual(info.slot_count, 2)
        self.assertEqual(info.pick_count, 4)

    def test_gen9_ou(self):
        info = detect_format_info("gen9ou")
        self.assertEqual(info.gen, 9)
        self.assertEqual(info.gametype, "singles")
        self.assertFalse(info.is_random)
        self.assertTrue(info.has_team_preview)
        self.assertEqual(info.slot_count, 1)

    def test_gen9_doubles_ou(self):
        info = detect_format_info("gen9doublesou")
        self.assertEqual(info.gen, 9)
        self.assertEqual(info.gametype, "doubles")
        self.assertEqual(info.slot_count, 2)

    def test_gen9_random_doubles(self):
        info = detect_format_info("gen9doublesrandombattle")
        self.assertEqual(info.gametype, "doubles")
        self.assertTrue(info.is_random)
        self.assertFalse(info.has_team_preview)

    def test_gen4_no_team_preview(self):
        info = detect_format_info("gen4ou")
        self.assertEqual(info.gen, 4)
        self.assertFalse(info.has_team_preview)

    def test_gen8_random_battle(self):
        info = detect_format_info("gen8randombattle")
        self.assertEqual(info.gen, 8)
        self.assertTrue(info.is_random)
        self.assertFalse(info.has_team_preview)

    def test_gen9_battle_factory(self):
        info = detect_format_info("gen9battlefactory")
        self.assertTrue(info.is_battle_factory)
        self.assertEqual(info.gen, 9)

    def test_gametype_from_request_json(self):
        # If request_json has 2 active slots, should detect doubles
        rj = {"active": [{}, {}]}
        info = detect_format_info("gen9ou", request_json=rj)
        self.assertEqual(info.gametype, "doubles")
        self.assertEqual(info.slot_count, 2)

    def test_gametype_from_request_json_singles(self):
        rj = {"active": [{}]}
        info = detect_format_info("gen9ou", request_json=rj)
        self.assertEqual(info.gametype, "singles")

    def test_unknown_format_defaults(self):
        info = detect_format_info("gen9somethingweird")
        self.assertEqual(info.gen, 9)
        self.assertEqual(info.gametype, "singles")


# =========================================================================
# FormatRules
# =========================================================================

class TestFormatRules(unittest.TestCase):
    def test_gen9_random_card(self):
        card = get_rule_card(9, "gen9randombattle")
        self.assertIn("Singles", card)
        self.assertIn("Terastallize", card)

    def test_gen9_vgc_card(self):
        card = get_rule_card(9, "gen9vgc2025regg")
        self.assertIn("Doubles", card)
        self.assertIn("pick 4", card)

    def test_gen8_ou_card(self):
        card = get_rule_card(8, "gen8ou")
        self.assertIn("Dynamax", card)

    def test_gen7_random_card(self):
        card = get_rule_card(7, "gen7randombattle")
        self.assertIn("Mega", card)

    def test_fallback_card(self):
        card = get_rule_card(5, "gen5somethingwacky")
        self.assertIn("Gen 5", card)

    def test_unknown_gen_fallback(self):
        card = get_rule_card(99, "gen99anything")
        self.assertIn("Gen 99", card)

    def test_move_target_singles(self):
        targets = get_move_target_semantics(9, "singles")
        # All targets should be None in singles
        for v in targets.values():
            self.assertIsNone(v)

    def test_move_target_doubles(self):
        targets = get_move_target_semantics(9, "doubles")
        self.assertEqual(targets["normal"], [-1, -2, 2])
        self.assertIsNone(targets["self"])
        self.assertIsNone(targets["allAdjacentFoes"])

    def test_move_target_triples(self):
        targets = get_move_target_semantics(9, "triples")
        self.assertIn(-3, targets["normal"])


# =========================================================================
# MsgParser
# =========================================================================

class TestMsgParser(unittest.TestCase):
    def test_empty_msg_list(self):
        snap = parse_msg_list([], "p1")
        self.assertEqual(len(snap.opponent_active_slots), 0)
        self.assertIsNone(snap.field_state.weather)

    def test_switch_parses_species_and_hp(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertIn(0, snap.opponent_active_slots)
        slot = snap.opponent_active_slots[0]
        self.assertEqual(slot.species, "garchomp")
        self.assertAlmostEqual(slot.hp_pct, 100.0)
        self.assertFalse(slot.fainted)

    def test_damage_updates_hp(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|-damage|p2a: Garchomp|150/300",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertAlmostEqual(slot.hp_pct, 50.0)

    def test_faint_detected(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|faint|p2a: Garchomp",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertTrue(slot.fainted)
        self.assertAlmostEqual(slot.hp_pct, 0.0)

    def test_move_revealed(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|move|p2a: Garchomp|Earthquake|p1a: Pikachu",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertIn("earthquake", slot.revealed_moves)

    def test_move_not_duplicated(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|move|p2a: Garchomp|Earthquake|p1a: Pikachu",
            "|move|p2a: Garchomp|Earthquake|p1a: Pikachu",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertEqual(slot.revealed_moves.count("earthquake"), 1)

    def test_boost(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|-boost|p2a: Garchomp|atk|2",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertEqual(slot.boosts["atk"], 2)

    def test_unboost(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|-boost|p2a: Garchomp|atk|2",
            "|-unboost|p2a: Garchomp|atk|1",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertEqual(slot.boosts["atk"], 1)

    def test_status(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|-status|p2a: Garchomp|brn",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertEqual(slot.status, "brn")

    def test_curestatus(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|-status|p2a: Garchomp|brn",
            "|-curestatus|p2a: Garchomp|brn",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertIsNone(slot.status)

    def test_weather(self):
        msgs = [
            "|-weather|Sandstorm",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertEqual(snap.field_state.weather, "sandstorm")

    def test_weather_none(self):
        msgs = [
            "|-weather|Sandstorm",
            "|-weather|none",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertIsNone(snap.field_state.weather)

    def test_terrain(self):
        msgs = [
            "|-fieldstart|move: Electric Terrain",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertIn("terrain", snap.field_state.terrain)

    def test_trick_room(self):
        msgs = [
            "|-fieldstart|move: Trick Room",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertTrue(snap.field_state.trick_room)

    def test_trick_room_ends(self):
        msgs = [
            "|-fieldstart|move: Trick Room",
            "|-fieldend|move: Trick Room",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertFalse(snap.field_state.trick_room)

    def test_side_conditions(self):
        msgs = [
            "|-sidestart|p2: Opponent|Stealth Rock",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertIn("stealthrock", snap.opp_side_conditions)

    def test_own_side_conditions(self):
        msgs = [
            "|-sidestart|p1: User|Reflect",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertIn("reflect", snap.own_side_conditions)

    def test_terastallize_opponent(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|-terastallize|p2a: Garchomp|Fire",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertEqual(slot.tera_type, "Fire")
        self.assertIn("terastallize", snap.opp_gimmicks_used)

    def test_terastallize_own(self):
        msgs = [
            "|-terastallize|p1a: Pikachu|Electric",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertIn("terastallize", snap.own_gimmicks_used)

    def test_item_revealed(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|-item|p2a: Garchomp|Leftovers",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertEqual(slot.revealed_item, "leftovers")

    def test_ability_revealed(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|-ability|p2a: Garchomp|Rough Skin",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertEqual(slot.revealed_ability, "roughskin")

    def test_own_pokemon_not_tracked_as_opponent(self):
        msgs = [
            "|switch|p1a: Pikachu|Pikachu, L82, M|200/200",
            "|move|p1a: Pikachu|Thunderbolt|p2a: Garchomp",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertEqual(len(snap.opponent_active_slots), 0)

    def test_doubles_two_slots(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|switch|p2b: Toxapex|Toxapex, L82, F|200/200",
        ]
        snap = parse_msg_list(msgs, "p1")
        self.assertIn(0, snap.opponent_active_slots)
        self.assertIn(1, snap.opponent_active_slots)
        self.assertEqual(snap.opponent_active_slots[0].species, "garchomp")
        self.assertEqual(snap.opponent_active_slots[1].species, "toxapex")

    def test_switch_resets_boosts(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|-boost|p2a: Garchomp|atk|2",
            "|switch|p2a: Toxapex|Toxapex, L82, F|200/200",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertEqual(slot.species, "toxapex")
        self.assertEqual(slot.boosts["atk"], 0)

    def test_damage_with_status(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "|-damage|p2a: Garchomp|150/300 brn",
        ]
        snap = parse_msg_list(msgs, "p1")
        slot = snap.opponent_active_slots[0]
        self.assertAlmostEqual(slot.hp_pct, 50.0)
        self.assertEqual(slot.status, "brn")

    def test_malformed_line_skipped(self):
        msgs = [
            "|switch|p2a: Garchomp|Garchomp, L78, M|300/300",
            "||",  # malformed
            "garbage line",
        ]
        snap = parse_msg_list(msgs, "p1")
        # Should not crash, should still have the switch parsed
        self.assertIn(0, snap.opponent_active_slots)


# =========================================================================
# Decision Parsing
# =========================================================================

class TestDecisionParsing(unittest.TestCase):
    def _make_view(self, gametype="singles", slot_count=1):
        fi = FormatInfo(
            gametype=gametype, gen=9, format_name="gen9ou",
            is_random=False, is_vgc=False, is_battle_factory=False,
            has_team_preview=True, slot_count=slot_count, pick_count=6,
        )
        return GeminiBattleView(
            format_info=fi, turn=1, slot_count=slot_count,
        )

    def test_parse_move(self):
        view = self._make_view()
        result = _parse_action_part(
            {"action_type": "move", "move_id": "earthquake"},
            view, slot_idx=0,
        )
        self.assertEqual(result, "move earthquake")

    def test_parse_switch(self):
        view = self._make_view()
        result = _parse_action_part(
            {"action_type": "switch", "switch_target": "garchomp"},
            view, slot_idx=0,
        )
        self.assertEqual(result, "switch garchomp")

    def test_parse_move_with_target(self):
        view = self._make_view(gametype="doubles", slot_count=2)
        result = _parse_action_part(
            {"action_type": "move", "move_id": "earthquake", "target": -1},
            view, slot_idx=0,
        )
        self.assertEqual(result, "move earthquake -1")

    def test_parse_move_with_tera(self):
        view = self._make_view()
        result = _parse_action_part(
            {"action_type": "move", "move_id": "earthquake", "gimmick": "terastallize"},
            view, slot_idx=0,
        )
        self.assertEqual(result, "move earthquake terastallize")

    def test_parse_move_with_mega(self):
        view = self._make_view()
        result = _parse_action_part(
            {"action_type": "move", "move_id": "earthquake", "gimmick": "mega"},
            view, slot_idx=0,
        )
        self.assertEqual(result, "move earthquake mega")

    def test_parse_move_with_dynamax(self):
        view = self._make_view()
        result = _parse_action_part(
            {"action_type": "move", "move_id": "earthquake", "gimmick": "dynamax"},
            view, slot_idx=0,
        )
        self.assertEqual(result, "move earthquake dynamax")

    def test_parse_move_no_gimmick(self):
        view = self._make_view()
        result = _parse_action_part(
            {"action_type": "move", "move_id": "thunderbolt", "gimmick": "none"},
            view, slot_idx=0,
        )
        self.assertEqual(result, "move thunderbolt")

    def test_parse_move_target_and_gimmick(self):
        view = self._make_view(gametype="doubles", slot_count=2)
        result = _parse_action_part(
            {"action_type": "move", "move_id": "flamethrower", "target": -2, "gimmick": "terastallize"},
            view, slot_idx=0,
        )
        self.assertEqual(result, "move flamethrower -2 terastallize")

    def test_parse_struggle_default(self):
        view = self._make_view()
        result = _parse_action_part(
            {"action_type": "move"},
            view, slot_idx=0,
        )
        self.assertEqual(result, "move struggle")


class TestTeamPreviewParsing(unittest.TestCase):
    def _make_view(self, team_size=6):
        fi = FormatInfo(
            gametype="singles", gen=9, format_name="gen9ou",
            is_random=False, is_vgc=False, is_battle_factory=False,
            has_team_preview=True, slot_count=1, pick_count=6,
        )
        own_team = [
            OwnPokemon(name=f"pkmn{i}", species=f"pkmn{i}", hp=100, max_hp=100,
                        hp_pct=100.0, status=None, active=False, index=i+1,
                        item="", ability="")
            for i in range(team_size)
        ]
        return GeminiBattleView(
            format_info=fi, turn=0, slot_count=1,
            own_team=own_team, is_team_preview=True,
        )

    def test_team_preview_order(self):
        view = self._make_view()
        result = _parse_team_preview({"lead_order": [3, 1, 4, 2, 5, 6]}, view)
        self.assertEqual(result, "314256")

    def test_team_preview_partial(self):
        view = self._make_view()
        result = _parse_team_preview({"lead_order": [3, 1]}, view)
        # Should fill in the rest: 3, 1, then 2, 4, 5, 6
        self.assertTrue(result.startswith("31"))
        self.assertEqual(len(result), 6)

    def test_team_preview_empty(self):
        view = self._make_view()
        result = _parse_team_preview({"lead_order": []}, view)
        self.assertEqual(result, "123456")

    def test_team_preview_no_key(self):
        view = self._make_view()
        result = _parse_team_preview({}, view)
        self.assertEqual(result, "123456")


# =========================================================================
# Prompt Generation
# =========================================================================

class TestPromptGeneration(unittest.TestCase):
    def _make_format_info(self):
        return FormatInfo(
            gametype="doubles", gen=9, format_name="gen9vgc2025regg",
            is_random=False, is_vgc=True, is_battle_factory=False,
            has_team_preview=True, slot_count=2, pick_count=4,
        )

    def test_system_prompt_contains_format(self):
        fi = self._make_format_info()
        prompt = build_system_prompt(fi, "Test rules text")
        self.assertIn("gen9vgc2025regg", prompt)
        self.assertIn("doubles", prompt)
        self.assertIn("Test rules text", prompt)
        self.assertIn("MUST use the function-calling tools", prompt)

    def test_turn_prompt_contains_turn_number(self):
        fi = self._make_format_info()
        pkmn = OwnPokemon(
            name="garchomp", species="garchomp", hp=300, max_hp=300,
            hp_pct=100.0, status=None, active=True, index=1,
            item="leftovers", ability="roughskin",
        )
        move = LegalMove(
            id="earthquake", name="Earthquake", pp=10, max_pp=10,
            target_type="normal", base_power=100, move_type="ground",
            category="physical",
        )
        slot = ActiveSlotView(
            slot_index=0, pokemon=pkmn, legal_moves=[move],
        )
        view = GeminiBattleView(
            format_info=fi, turn=5, slot_count=2,
            active_slots=[slot], own_team=[pkmn],
        )
        prompt = build_turn_prompt(view)
        self.assertIn("TURN 5", prompt)
        self.assertIn("garchomp", prompt)
        self.assertIn("earthquake", prompt)

    def test_team_preview_prompt(self):
        fi = self._make_format_info()
        pkmn1 = OwnPokemon(
            name="garchomp", species="garchomp", hp=300, max_hp=300,
            hp_pct=100.0, status=None, active=False, index=1,
            item="leftovers", ability="roughskin", tera_type="Fire",
            moves=["earthquake", "dragonclaw", "swordsdance", "stoneedge"],
        )
        view = GeminiBattleView(
            format_info=fi, turn=0, slot_count=2,
            own_team=[pkmn1], is_team_preview=True,
            opponent_preview_team=["pikachu", "charizard"],
            pick_count=4,
        )
        prompt = build_team_preview_prompt(view)
        self.assertIn("TEAM PREVIEW", prompt)
        self.assertIn("garchomp", prompt)
        self.assertIn("pikachu", prompt)
        self.assertIn("Pick 4", prompt)


# =========================================================================
# Tutor Sanitization
# =========================================================================

class TestTutorSanitize(unittest.TestCase):
    def test_strips_leading_slash(self):
        self.assertEqual(_sanitize_reply("/some command"), "some command")

    def test_strips_multiple_slashes(self):
        self.assertEqual(_sanitize_reply("///msg"), "msg")

    def test_caps_length(self):
        long_text = "x" * 500
        result = _sanitize_reply(long_text)
        self.assertLessEqual(len(result), 250)
        self.assertTrue(result.endswith("..."))

    def test_strips_newlines(self):
        result = _sanitize_reply("line1\nline2")
        self.assertNotIn("\n", result)

    def test_empty_returns_none(self):
        self.assertIsNone(_sanitize_reply(""))
        self.assertIsNone(_sanitize_reply("   "))

    def test_none_returns_none(self):
        # _sanitize_reply only takes str, but empty string
        self.assertIsNone(_sanitize_reply(""))


# =========================================================================
# Error classes
# =========================================================================

class TestErrors(unittest.TestCase):
    def test_auth_error(self):
        e = GeminiAuthError("no creds")
        self.assertIn("no creds", str(e))

    def test_invalid_choice(self):
        e = GeminiInvalidChoice("bad move")
        self.assertIn("bad move", str(e))

    def test_timeout(self):
        e = GeminiTimeout("too slow")
        self.assertIn("too slow", str(e))


if __name__ == "__main__":
    unittest.main()
