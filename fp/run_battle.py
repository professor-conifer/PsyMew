import json
import asyncio
import concurrent.futures
from copy import deepcopy
import logging

from data.pkmn_sets import RandomBattleTeamDatasets, TeamDatasets
from data.pkmn_sets import SmogonSets
import constants
from constants import BattleType
from config import FoulPlayConfig, SaveReplay
from fp.battle import LastUsedMove, Pokemon, Battle
from fp.battle_modifier import async_update_battle, process_battle_updates
from fp.helpers import normalize_name
from fp.search.main import find_best_move

from fp.websocket_client import PSWebsocketClient

logger = logging.getLogger(__name__)


def _use_gemini(battle=None) -> bool:
    """Check if Gemini engine should be used for the current decision."""
    return FoulPlayConfig.decision_engine == "gemini"


def _use_claude(battle=None) -> bool:
    """Check if Claude engine should be used for the current decision."""
    return FoulPlayConfig.decision_engine == "claude"


def format_decision(battle, decision):
    # Formats a decision for communication with Pokemon-Showdown
    # If the move can be used as a Z-Move, it will be

    if decision.startswith(constants.SWITCH_STRING + " "):
        switch_pokemon = decision.split("switch ")[-1]
        for pkmn in battle.user.reserve:
            if pkmn.name == switch_pokemon:
                message = "/switch {}".format(pkmn.index)
                break
        else:
            raise ValueError("Tried to switch to: {}".format(switch_pokemon))
    else:
        tera = False
        mega = False
        if decision.endswith("-tera"):
            decision = decision.replace("-tera", "")
            tera = True
        elif decision.endswith("-mega"):
            decision = decision.replace("-mega", "")
            mega = True
        message = "/choose move {}".format(decision)

        if battle.user.active.can_mega_evo and mega:
            message = "{} {}".format(message, constants.MEGA)
        elif battle.user.active.can_ultra_burst:
            message = "{} {}".format(message, constants.ULTRA_BURST)

        # only dynamax on last pokemon
        if battle.user.active.can_dynamax and all(
            p.hp == 0 for p in battle.user.reserve
        ):
            message = "{} {}".format(message, constants.DYNAMAX)

        if tera:
            message = "{} {}".format(message, constants.TERASTALLIZE)

        if battle.user.active.get_move(decision).can_z:
            message = "{} {}".format(message, constants.ZMOVE)

    return [message, str(battle.rqid)]


def format_gemini_decision(battle, action_parts: list[dict]) -> list[str]:
    """Format Gemini's action parts into a Showdown message list.

    action_parts: list of dicts like:
      [{"decision": "move earthquake -1 terastallize", "slot": 0}, ...]
      or [{"team_order": "3142"}]
    """
    # Team preview
    if action_parts and "team_order" in action_parts[0]:
        order = action_parts[0]["team_order"]
        return ["/team {}|{}".format(order, battle.rqid)]

    # Build per-slot /choose parts
    choice_parts = []
    for part in sorted(action_parts, key=lambda x: x.get("slot", 0)):
        decision = part.get("decision", "move struggle")
        tokens = decision.split()

        if tokens[0] == "switch" and len(tokens) >= 2:
            switch_name = " ".join(tokens[1:])  # handle multi-word names
            # Match directly from request_json
            side_pokemon = battle.request_json.get("side", {}).get("pokemon", [])
            found = False
            for i, p in enumerate(side_pokemon):
                details = p.get("details", "")
                species = details.split(",")[0].lower().replace(" ", "").replace("-", "")
                if species == switch_name.replace("-", ""):
                    choice_parts.append("switch {}".format(i + 1))
                    found = True
                    break
            if not found:
                logger.warning("Gemini tried to switch to unknown: %s", switch_name)
                choice_parts.append("move 1")  # fallback

        elif tokens[0] == "move" and len(tokens) >= 2:
            move_id = tokens[1]
            move_str = "move {}".format(move_id)

            # Parse remaining tokens: target (int) and gimmick (str)
            target = None
            gimmick = None
            for tok in tokens[2:]:
                try:
                    target = int(tok)
                except ValueError:
                    if tok in ("terastallize", "mega", "dynamax", "zmove"):
                        gimmick = tok

            if target is not None:
                move_str += " {}".format(target)

            if gimmick == "terastallize":
                move_str += " terastallize"
            elif gimmick == "mega":
                move_str += " mega"
            elif gimmick == "dynamax":
                move_str += " dynamax"
            elif gimmick == "zmove":
                move_str += " zmove"

            choice_parts.append(move_str)
        else:
            choice_parts.append("move 1")  # fallback

    message = "/choose " + ",".join(choice_parts)
    return [message, str(battle.rqid)]


def battle_is_finished(battle_tag, msg):
    return (
        msg.startswith(">{}".format(battle_tag))
        and (constants.WIN_STRING in msg or constants.TIE_STRING in msg)
        and constants.CHAT_STRING not in msg
    )


def extract_battle_factory_tier_from_msg(msg):
    start = msg.find("Battle Factory Tier: ") + len("Battle Factory Tier: ")
    end = msg.find("</b>", start)
    tier_name = msg[start:end]

    return normalize_name(tier_name)


async def async_pick_move(battle):
    if _use_gemini(battle):
        return await _gemini_pick_move(battle)
    if _use_claude(battle):
        return await _claude_pick_move(battle)

    battle_copy = deepcopy(battle)
    if not battle_copy.team_preview:
        battle_copy.user.update_from_request_json(battle_copy.request_json)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        best_move = await loop.run_in_executor(pool, find_best_move, battle_copy)
    battle.user.last_selected_move = LastUsedMove(
        battle.user.active.name,
        best_move.removesuffix("-tera").removesuffix("-mega"),
        battle.turn,
    )
    return format_decision(battle_copy, best_move)


async def _gemini_pick_move(battle):
    """Dispatch decision to the Gemini engine."""
    from fp.gemini.decision import find_best_move_gemini

    try:
        action_parts = await find_best_move_gemini(battle)
        logger.info("Gemini action_parts: %s", action_parts)
        # Track last selected move for speed range checks
        if action_parts and "decision" in action_parts[0]:
            dec = action_parts[0]["decision"]
            tokens = dec.split()
            if tokens[0] == "move" and len(tokens) >= 2:
                move_name = tokens[1]
                active_name = battle.user.active.name if battle.user.active else "unknown"
                battle.user.last_selected_move = LastUsedMove(
                    active_name, move_name, battle.turn,
                )
        return format_gemini_decision(battle, action_parts)
    except Exception as exc:
        import traceback
        logger.error("Gemini decision failed: %s: %s", type(exc).__name__, exc)
        logger.debug("Gemini traceback:\n%s", traceback.format_exc())

        # Smart fallback: use the move scorer to pick the best action
        try:
            from fp.gemini.view import GeminiBattleView
            from fp.gemini.move_scorer import get_best_action
            view = GeminiBattleView.from_battle(battle)
            best_decision = get_best_action(view)
            logger.info("[GEMINI FALLBACK] Scorer picked: %s", best_decision)
            active_name = battle.user.active.name if battle.user.active else "unknown"
            battle.user.last_selected_move = LastUsedMove(active_name, best_decision.split()[-1], battle.turn)
            return format_gemini_decision(battle, [{"decision": best_decision, "slot": 0}])
        except Exception as fallback_exc:
            logger.error("Scorer fallback also failed: %s", fallback_exc)

        # Last resort: force-switch or move 1
        if battle.request_json and battle.request_json.get("forceSwitch"):
            side_pokemon = battle.request_json.get("side", {}).get("pokemon", [])
            for i, p in enumerate(side_pokemon):
                cond = p.get("condition", "")
                if "fnt" not in cond and not p.get("active", False):
                    return [f"/choose switch {i + 1}", str(battle.rqid)]
        logger.info("[GEMINI EXHAUSTED] All fallbacks failed. Defaulting to move 1.")
        return ["/choose move 1", str(battle.rqid)]


async def handle_team_preview(battle, ps_websocket_client):
    if _use_gemini(battle):
        return await _gemini_handle_team_preview(battle, ps_websocket_client)
    if _use_claude(battle):
        return await _claude_handle_team_preview(battle, ps_websocket_client)

    battle_copy = deepcopy(battle)
    battle_copy.user.active = Pokemon.get_dummy()
    battle_copy.opponent.active = Pokemon.get_dummy()
    battle_copy.team_preview = True

    best_move = await async_pick_move(battle_copy)

    # because we copied the battle before sending it in, we need to update the last selected move here
    pkmn_name = battle.user.reserve[int(best_move[0].split()[1]) - 1].name
    battle.user.last_selected_move = LastUsedMove(
        "teampreview", "switch {}".format(pkmn_name), battle.turn
    )

    size_of_team = len(battle.user.reserve) + 1
    team_list_indexes = list(range(1, size_of_team))
    choice_digit = int(best_move[0].split()[-1])

    team_list_indexes.remove(choice_digit)
    message = [
        "/team {}{}|{}".format(
            choice_digit, "".join(str(x) for x in team_list_indexes), battle.rqid
        )
    ]

    await ps_websocket_client.send_message(battle.battle_tag, message)


async def _gemini_handle_team_preview(battle, ps_websocket_client):
    """Gemini-powered team preview: pick leads using the AI."""
    from fp.gemini.decision import find_best_move_gemini

    # Temporarily mark as team preview for the view builder
    battle.team_preview = True
    battle.request_json["teamPreview"] = True

    try:
        action_parts = await find_best_move_gemini(battle)
        logger.info("Gemini team preview: %s", action_parts)

        if action_parts and "team_order" in action_parts[0]:
            order = action_parts[0]["team_order"]
            message = ["/team {}|{}".format(order, battle.rqid)]
        else:
            # Fallback: default order
            size_of_team = len(battle.user.reserve) + 1
            order = "".join(str(i) for i in range(1, size_of_team))
            message = ["/team {}|{}".format(order, battle.rqid)]

        battle.user.last_selected_move = LastUsedMove(
            "teampreview", "gemini-lead-{}".format(order[:1]), battle.turn
        )
        await ps_websocket_client.send_message(battle.battle_tag, message)

    except Exception as exc:
        logger.error("Gemini team preview failed: %s — using default order", exc)
        size_of_team = len(battle.user.reserve) + 1
        order = "".join(str(i) for i in range(1, size_of_team))
        message = ["/team {}|{}".format(order, battle.rqid)]
        await ps_websocket_client.send_message(battle.battle_tag, message)

    finally:
        battle.team_preview = False


async def _claude_pick_move(battle):
    """Dispatch decision to the Claude engine."""
    from fp.claude.decision import find_best_move_claude

    try:
        action_parts = await find_best_move_claude(battle)
        logger.info("Claude action_parts: %s", action_parts)
        # Track last selected move for speed range checks
        if action_parts and "decision" in action_parts[0]:
            dec = action_parts[0]["decision"]
            tokens = dec.split()
            if tokens[0] == "move" and len(tokens) >= 2:
                move_name = tokens[1]
                active_name = battle.user.active.name if battle.user.active else "unknown"
                battle.user.last_selected_move = LastUsedMove(
                    active_name, move_name, battle.turn,
                )
        return format_gemini_decision(battle, action_parts)
    except Exception as exc:
        import traceback
        logger.error("Claude decision failed: %s: %s", type(exc).__name__, exc)
        logger.debug("Claude traceback:\n%s", traceback.format_exc())

        # Smart fallback: use the move scorer to pick the best action
        try:
            from fp.gemini.view import GeminiBattleView
            from fp.gemini.move_scorer import get_best_action
            view = GeminiBattleView.from_battle(battle)
            best_decision = get_best_action(view)
            logger.info("[CLAUDE FALLBACK] Scorer picked: %s", best_decision)
            active_name = battle.user.active.name if battle.user.active else "unknown"
            battle.user.last_selected_move = LastUsedMove(active_name, best_decision.split()[-1], battle.turn)
            return format_gemini_decision(battle, [{"decision": best_decision, "slot": 0}])
        except Exception as fallback_exc:
            logger.error("Scorer fallback also failed: %s", fallback_exc)

        # Last resort: force-switch or move 1
        if battle.request_json and battle.request_json.get("forceSwitch"):
            side_pokemon = battle.request_json.get("side", {}).get("pokemon", [])
            for i, p in enumerate(side_pokemon):
                cond = p.get("condition", "")
                if "fnt" not in cond and not p.get("active", False):
                    return [f"/choose switch {i + 1}", str(battle.rqid)]
        logger.info("[CLAUDE EXHAUSTED] All fallbacks failed. Defaulting to move 1.")
        return ["/choose move 1", str(battle.rqid)]


async def _claude_handle_team_preview(battle, ps_websocket_client):
    """Claude-powered team preview: pick leads using the AI."""
    from fp.claude.decision import find_best_move_claude

    # Temporarily mark as team preview for the view builder
    battle.team_preview = True
    battle.request_json["teamPreview"] = True

    try:
        action_parts = await find_best_move_claude(battle)
        logger.info("Claude team preview: %s", action_parts)

        if action_parts and "team_order" in action_parts[0]:
            order = action_parts[0]["team_order"]
            message = ["/team {}|{}".format(order, battle.rqid)]
        else:
            # Fallback: default order
            size_of_team = len(battle.user.reserve) + 1
            order = "".join(str(i) for i in range(1, size_of_team))
            message = ["/team {}|{}".format(order, battle.rqid)]

        battle.user.last_selected_move = LastUsedMove(
            "teampreview", "claude-lead-{}".format(order[:1]), battle.turn
        )
        await ps_websocket_client.send_message(battle.battle_tag, message)

    except Exception as exc:
        logger.error("Claude team preview failed: %s — using default order", exc)
        size_of_team = len(battle.user.reserve) + 1
        order = "".join(str(i) for i in range(1, size_of_team))
        message = ["/team {}|{}".format(order, battle.rqid)]
        await ps_websocket_client.send_message(battle.battle_tag, message)

    finally:
        battle.team_preview = False


async def get_battle_tag_and_opponent(ps_websocket_client: PSWebsocketClient):
    while True:
        msg = await ps_websocket_client.receive_message()
        split_msg = msg.split("|")
        first_msg = split_msg[0]
        if "battle" in first_msg:
            battle_tag = first_msg.replace(">", "").strip()
            user_name = FoulPlayConfig.username
            opponent_name = (
                split_msg[4].replace(user_name, "").replace("vs.", "").strip()
            )
            logger.info("Initialized {} against: {}".format(battle_tag, opponent_name))
            return battle_tag, opponent_name


async def start_battle_common(
    ps_websocket_client: PSWebsocketClient, pokemon_battle_type
):
    battle_tag, opponent_name = await get_battle_tag_and_opponent(ps_websocket_client)
    if FoulPlayConfig.log_to_file:
        FoulPlayConfig.file_log_handler.do_rollover(
            "{}_{}.log".format(battle_tag, opponent_name)
        )

    battle = Battle(battle_tag)
    battle.opponent.account_name = opponent_name
    battle.pokemon_format = pokemon_battle_type
    battle.generation = pokemon_battle_type[:4]

    # wait until the opponent's identifier is received. This will be `p1` or `p2`.
    #
    # e.g.
    # '>battle-gen9randombattle-44733
    # |player|p1|OpponentName|2|'
    while True:
        msg = await ps_websocket_client.receive_message()
        if "|player|" in msg and battle.opponent.account_name in msg:
            battle.opponent.name = msg.split("|")[2]
            battle.user.name = constants.ID_LOOKUP[battle.opponent.name]
            break

    return battle, msg


async def get_first_request_json(
    ps_websocket_client: PSWebsocketClient, battle: Battle
):
    while True:
        msg = await ps_websocket_client.receive_message()
        msg_split = msg.split("|")
        if msg_split[1].strip() == "request" and msg_split[2].strip():
            user_json = json.loads(msg_split[2].strip("'"))
            battle.request_json = user_json
            battle.user.initialize_first_turn_user_from_json(user_json)
            battle.rqid = user_json[constants.RQID]
            return


async def start_random_battle(
    ps_websocket_client: PSWebsocketClient, pokemon_battle_type
):
    battle, msg = await start_battle_common(ps_websocket_client, pokemon_battle_type)
    battle.battle_type = BattleType.RANDOM_BATTLE
    RandomBattleTeamDatasets.initialize(battle.generation)

    while True:
        if constants.START_STRING in msg:
            battle.started = True

            # hold onto some messages to apply after we get the request JSON
            # omit the bot's switch-in message because we won't need that
            # parsing the request JSON will set the bot's active pkmn
            battle.msg_list = [
                m
                for m in msg.split(constants.START_STRING)[1].strip().split("\n")
                if not (m.startswith("|switch|{}".format(battle.user.name)))
            ]
            break
        msg = await ps_websocket_client.receive_message()

    await get_first_request_json(ps_websocket_client, battle)

    # Attach AI context early — before first move decision
    if _use_gemini():
        await _attach_gemini_context(battle, ps_websocket_client)
    elif _use_claude():
        await _attach_claude_context(battle, ps_websocket_client)

    # apply the messages that were held onto
    process_battle_updates(battle)

    best_move = await async_pick_move(battle)
    await ps_websocket_client.send_message(battle.battle_tag, best_move)

    return battle


async def start_standard_battle(
    ps_websocket_client: PSWebsocketClient, pokemon_battle_type, team_dict
):
    battle, msg = await start_battle_common(ps_websocket_client, pokemon_battle_type)
    battle.user.team_dict = team_dict
    if "battlefactory" in pokemon_battle_type:
        battle.battle_type = BattleType.BATTLE_FACTORY
    else:
        battle.battle_type = BattleType.STANDARD_BATTLE

    if battle.generation in constants.NO_TEAM_PREVIEW_GENS:
        while True:
            if constants.START_STRING in msg:
                battle.started = True

                # hold onto some messages to apply after we get the request JSON
                # omit the bot's switch-in message because we won't need that
                # parsing the request JSON will set the bot's active pkmn
                battle.msg_list = [
                    m
                    for m in msg.split(constants.START_STRING)[1].strip().split("\n")
                    if not (m.startswith("|switch|{}".format(battle.user.name)))
                ]
                break
            msg = await ps_websocket_client.receive_message()

        await get_first_request_json(ps_websocket_client, battle)

        unique_pkmn_names = set(
            [p.name for p in battle.user.reserve] + [battle.user.active.name]
        )
        SmogonSets.initialize(
            FoulPlayConfig.smogon_stats or pokemon_battle_type, unique_pkmn_names
        )
        TeamDatasets.initialize(pokemon_battle_type, unique_pkmn_names)

        # apply the messages that were held onto
        process_battle_updates(battle)

        best_move = await async_pick_move(battle)
        await ps_websocket_client.send_message(battle.battle_tag, best_move)

    else:
        while constants.START_TEAM_PREVIEW not in msg:
            msg = await ps_websocket_client.receive_message()

        preview_string_lines = msg.split(constants.START_TEAM_PREVIEW)[-1].split("\n")

        opponent_pokemon = []
        for line in preview_string_lines:
            if not line:
                continue

            split_line = line.split("|")
            if (
                split_line[1] == constants.TEAM_PREVIEW_POKE
                and split_line[2].strip() == battle.opponent.name
            ):
                opponent_pokemon.append(split_line[3])

        await get_first_request_json(ps_websocket_client, battle)
        battle.initialize_team_preview(opponent_pokemon, pokemon_battle_type)
        battle.during_team_preview()

        unique_pkmn_names = set(
            p.name for p in battle.opponent.reserve + battle.user.reserve
        )

        if battle.battle_type == BattleType.BATTLE_FACTORY:
            battle.battle_type = BattleType.BATTLE_FACTORY
            tier_name = extract_battle_factory_tier_from_msg(msg)
            logger.info("Battle Factory Tier: {}".format(tier_name))
            TeamDatasets.initialize(
                pokemon_battle_type,
                unique_pkmn_names,
                battle_factory_tier_name=tier_name,
            )
        else:
            battle.battle_type = BattleType.STANDARD_BATTLE
            SmogonSets.initialize(
                FoulPlayConfig.smogon_stats or pokemon_battle_type, unique_pkmn_names
            )
            TeamDatasets.initialize(pokemon_battle_type, unique_pkmn_names)

        await handle_team_preview(battle, ps_websocket_client)

    return battle


async def start_battle(ps_websocket_client, pokemon_battle_type, team_dict):
    if "random" in pokemon_battle_type:
        battle = await start_random_battle(ps_websocket_client, pokemon_battle_type)
    else:
        battle = await start_standard_battle(
            ps_websocket_client, pokemon_battle_type, team_dict
        )

    # Attach AI format info for standard battles (random battles attach in start_random_battle)
    if _use_gemini() and battle.format_info is None:
        await _attach_gemini_context(battle, ps_websocket_client)
    elif _use_claude() and battle.format_info is None:
        await _attach_claude_context(battle, ps_websocket_client)

    await ps_websocket_client.send_message(battle.battle_tag, ["hf"])
    await ps_websocket_client.send_message(battle.battle_tag, ["/timer on"])

    return battle


async def _send_tutor_chat(ps_websocket_client, battle_tag, chunks):
    """Send each tutor chunk as a separate Showdown chat message.

    Tutor returns list[str] chunks, but send_message() joins with '|' which
    Showdown interprets as protocol separators. Each chunk must be sent
    individually as its own chat message.
    """
    if isinstance(chunks, str):
        chunks = [chunks]
    for chunk in chunks:
        text = chunk.lstrip("/")
        if text:
            await ps_websocket_client.send_message(battle_tag, [text])
            await asyncio.sleep(0.3)  # small delay to avoid flooding


async def _attach_gemini_context(battle, ps_websocket_client):
    """Attach format_info, verified rules, and optional tutor session to battle."""
    from fp.gemini.format_detection import detect_format_info
    from fp.gemini.format_rules import get_rule_card

    # Detect format
    battle.format_info = detect_format_info(
        battle.pokemon_format, battle.request_json
    )

    # Get hardcoded rule card
    rule_card = get_rule_card(battle.format_info.gen, battle.format_info.format_name)

    # Verify rules + fetch meta context via live Google Search (with timeout)
    try:
        from fp.gemini.format_research import verify_format_rules, fetch_format_meta_context
        from fp.gemini.client import get_client, get_model_name

        client = get_client(
            auth_mode=FoulPlayConfig.gemini_auth_mode,
            api_key_override=FoulPlayConfig.gemini_api_key,
        )
        model_name = get_model_name()

        # Run both fetches concurrently
        rules_task = asyncio.ensure_future(
            verify_format_rules(client, model_name, battle.format_info, rule_card)
        )
        meta_task = asyncio.ensure_future(
            fetch_format_meta_context(client, model_name, battle.format_info)
        )
        battle.format_rules_text, battle.format_meta_context = await asyncio.gather(
            rules_task, meta_task
        )
        logger.info("Format rules verified and meta context fetched")
    except Exception as exc:
        logger.warning("Rule/meta verification failed, using stored card: %s", exc)
        battle.format_rules_text = rule_card
        battle.format_meta_context = ""

    # Tutor session
    if FoulPlayConfig.tutor_mode:
        from fp.gemini.tutor import TutorSession

        battle.tutor_session = TutorSession(FoulPlayConfig.username)
        greeting = await battle.tutor_session.on_battle_start(
            battle.format_info.format_name, battle.format_rules_text
        )
        if greeting:
            await _send_tutor_chat(ps_websocket_client, battle.battle_tag, greeting)


async def _attach_claude_context(battle, ps_websocket_client):
    """Attach format_info, verified rules, and optional tutor session to battle (Claude engine)."""
    from fp.gemini.format_detection import detect_format_info
    from fp.gemini.format_rules import get_rule_card

    # Detect format (shared module)
    battle.format_info = detect_format_info(
        battle.pokemon_format, battle.request_json
    )

    # Get hardcoded rule card
    rule_card = get_rule_card(battle.format_info.gen, battle.format_info.format_name)

    # Verify rules via Claude's web search
    try:
        from fp.claude.format_research import verify_format_rules
        from fp.claude.client import get_async_client, get_model_name

        client = get_async_client(
            auth_mode=FoulPlayConfig.claude_auth_mode,
            api_key_override=FoulPlayConfig.claude_api_key,
        )
        battle.format_rules_text = await verify_format_rules(
            client, get_model_name(), battle.format_info, rule_card
        )
        logger.info("Format rules verified and attached (Claude)")
    except Exception as exc:
        logger.warning("Rule verification failed, using stored card: %s", exc)
        battle.format_rules_text = rule_card

    # Tutor session (Claude)
    if FoulPlayConfig.tutor_mode:
        from fp.claude.tutor import TutorSession

        battle.tutor_session = TutorSession(FoulPlayConfig.username)
        greeting = await battle.tutor_session.on_battle_start(
            battle.format_info.format_name, battle.format_rules_text
        )
        if greeting:
            await _send_tutor_chat(ps_websocket_client, battle.battle_tag, greeting)


async def pokemon_battle(ps_websocket_client, pokemon_battle_type, team_dict):
    battle = await start_battle(ps_websocket_client, pokemon_battle_type, team_dict)
    while True:
        msg = await ps_websocket_client.receive_message()
        if battle_is_finished(battle.battle_tag, msg):
            winner = (
                msg.split(constants.WIN_STRING)[-1].split("\n")[0].strip()
                if constants.WIN_STRING in msg
                else None
            )
            logger.info("Winner: {}".format(winner))
            await ps_websocket_client.send_message(battle.battle_tag, ["gg"])

            # Tutor: post-game review
            if battle.tutor_session:
                review = await battle.tutor_session.on_battle_end(winner, FoulPlayConfig.username)
                if review:
                    await _send_tutor_chat(ps_websocket_client, battle.battle_tag, review)

            if (
                FoulPlayConfig.save_replay == SaveReplay.always
                or (
                    FoulPlayConfig.save_replay == SaveReplay.on_loss
                    and winner != FoulPlayConfig.username
                )
                or (
                    FoulPlayConfig.save_replay == SaveReplay.on_win
                    and winner == FoulPlayConfig.username
                )
            ):
                await ps_websocket_client.save_replay(battle.battle_tag)
            await ps_websocket_client.leave_battle(battle.battle_tag)
            return winner
        else:
            # Tutor: handle inbound chat from opponent
            if battle.tutor_session and constants.CHAT_STRING in msg:
                await _handle_tutor_chat(battle, msg, ps_websocket_client)

            action_required = await async_update_battle(battle, msg)
            if action_required and not battle.wait:
                best_move = await async_pick_move(battle)
                await ps_websocket_client.send_message(battle.battle_tag, best_move)

                # Tutor: post-turn coaching
                if battle.tutor_session and not battle.team_preview:
                    turn_summary = _extract_turn_summary(battle)
                    comment = await battle.tutor_session.on_turn_complete(turn_summary)
                    if comment:
                        await _send_tutor_chat(ps_websocket_client, battle.battle_tag, comment)


async def _handle_tutor_chat(battle, msg, ps_websocket_client):
    """Extract opponent chat messages and let the tutor respond."""
    for line in msg.split("\n"):
        if constants.CHAT_STRING in line:
            parts = line.split("|")
            # Format: |c|~User|message text
            if len(parts) >= 4:
                sender = parts[2].lstrip("~+%@&#").strip()
                text = "|".join(parts[3:]).strip()
                if sender and text:
                    reply = await battle.tutor_session.on_incoming_chat(sender, text)
                    if reply:
                        await _send_tutor_chat(ps_websocket_client, battle.battle_tag, reply)


def _extract_turn_summary(battle) -> str:
    """Build a human-readable turn summary for the Tutor.

    Scoped to the CURRENT TURN only (everything after the last |turn| marker).
    Labels sides correctly:
      - The STUDENT (human) = the OPPONENT of the bot
      - The BOT (us) = battle.user

    Also includes HP context for both sides so the tutor can reason about game state.
    """
    bot_side = battle.user.name  # e.g. "p2"

    def _label(ident: str) -> str:
        """'p2a: Zekrom' → 'Bot's Zekrom' or 'Student's Zekrom'."""
        parts = ident.split(":")
        side = parts[0].strip()[:2]
        name = parts[1].strip() if len(parts) > 1 else "?"
        return f"Bot's {name}" if side == bot_side else f"Student's {name}"

    def _is_bot(ident: str) -> bool:
        return ident.split(":")[0].strip()[:2] == bot_side

    msg_source = getattr(battle, 'gemini_msg_log', None) or battle.msg_list

    # Find the last |turn| marker to scope to current turn only
    turn_start = 0
    for i in range(len(msg_source) - 1, -1, -1):
        parts = msg_source[i].split("|")
        if len(parts) >= 3 and parts[1].strip() == "turn":
            turn_start = i
            break

    # Parse only current turn's messages
    summary_lines = []
    # Track HP changes for context
    bot_hp = {}    # species -> hp string
    student_hp = {}  # species -> hp string

    for line in msg_source[turn_start:]:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        action = parts[1].strip()
        try:
            if action == "turn" and len(parts) >= 3:
                summary_lines.append(f"--- Turn {parts[2].strip()} ---")

            elif action == "move" and len(parts) >= 4:
                who = _label(parts[2])
                move = parts[3].strip()
                summary_lines.append(f"{who} used {move}")

            elif action == "switch" and len(parts) >= 5:
                who = _label(parts[2])
                species = parts[3].strip().split(",")[0]
                hp_str = parts[4].strip().split()[0] if len(parts) >= 5 else "100/100"
                summary_lines.append(f"{who} switched in {species} ({hp_str} HP)")
                # Track HP
                name = parts[2].split(":")[-1].strip() if ":" in parts[2] else species
                if _is_bot(parts[2]):
                    bot_hp[name] = hp_str
                else:
                    student_hp[name] = hp_str

            elif action == "faint" and len(parts) >= 3:
                who = _label(parts[2])
                summary_lines.append(f"{who} fainted!")

            elif action == "-damage" and len(parts) >= 4:
                who = _label(parts[2])
                hp = parts[3].strip().split()[0]
                summary_lines.append(f"{who} took damage → {hp} HP remaining")
                # Track HP
                name = parts[2].split(":")[-1].strip() if ":" in parts[2] else "?"
                if _is_bot(parts[2]):
                    bot_hp[name] = hp
                else:
                    student_hp[name] = hp

            elif action == "-heal" and len(parts) >= 4:
                who = _label(parts[2])
                hp = parts[3].strip().split()[0]
                summary_lines.append(f"{who} recovered → {hp} HP")
                name = parts[2].split(":")[-1].strip() if ":" in parts[2] else "?"
                if _is_bot(parts[2]):
                    bot_hp[name] = hp
                else:
                    student_hp[name] = hp

            elif action == "-supereffective":
                summary_lines.append("  (Super effective!)")

            elif action == "-resisted":
                summary_lines.append("  (Not very effective)")

            elif action == "-immune" and len(parts) >= 3:
                who = _label(parts[2])
                summary_lines.append(f"  {who} was IMMUNE!")

            elif action == "-boost" and len(parts) >= 5:
                who = _label(parts[2])
                stat = parts[3].strip()
                amount = parts[4].strip()
                summary_lines.append(f"{who} {stat} +{amount}")

            elif action == "-unboost" and len(parts) >= 5:
                who = _label(parts[2])
                stat = parts[3].strip()
                amount = parts[4].strip()
                summary_lines.append(f"{who} {stat} -{amount}")

            elif action == "-status" and len(parts) >= 4:
                who = _label(parts[2])
                status = parts[3].strip()
                summary_lines.append(f"{who} was inflicted with {status}")

            elif action == "-ability" and len(parts) >= 4:
                who = _label(parts[2])
                ability = parts[3].strip()
                summary_lines.append(f"{who}'s ability {ability} activated")

        except (IndexError, ValueError):
            continue

    if not summary_lines:
        return f"Turn {battle.turn} (no events recorded)"

    return "\n".join(summary_lines)

