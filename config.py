import argparse
import logging
import os
import sys
from enum import Enum, auto
from logging.handlers import RotatingFileHandler
from typing import Optional


class CustomFormatter(logging.Formatter):
    def format(self, record):
        lvl = "{}".format(record.levelname)
        return "{} {}".format(lvl.ljust(8), record.getMessage())


class CustomRotatingFileHandler(RotatingFileHandler):
    def __init__(self, file_name, **kwargs):
        self.base_dir = "logs"
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        super().__init__("{}/{}".format(self.base_dir, file_name), encoding='utf-8', **kwargs)

    def do_rollover(self, new_file_name):
        new_file_name = new_file_name.replace("/", "_")
        self.baseFilename = "{}/{}".format(self.base_dir, new_file_name)
        self.doRollover()


def init_logging(level, log_to_file):
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    requests_logger = logging.getLogger("urllib3")
    requests_logger.setLevel(logging.INFO)

    # Gets the root logger to set handlers/formatters
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(CustomFormatter())
    logger.addHandler(stdout_handler)
    FoulPlayConfig.stdout_log_handler = stdout_handler

    if log_to_file:
        file_handler = CustomRotatingFileHandler("init.log")
        file_handler.setLevel(logging.DEBUG)  # file logs are always debug
        file_handler.setFormatter(CustomFormatter())
        logger.addHandler(file_handler)
        FoulPlayConfig.file_log_handler = file_handler


class SaveReplay(Enum):
    always = auto()
    never = auto()
    on_loss = auto()
    on_win = auto()


class BotModes(Enum):
    challenge_user = auto()
    accept_challenge = auto()
    search_ladder = auto()


_ENGINE_KEY_ENVS = {
    "claude": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
}


class _FoulPlayConfig:
    websocket_uri: str
    # Optional override for the Showdown login endpoint. Empty string means
    # "use the default play.pokemonshowdown.com". Mirrors (PokéAgent
    # Challenge, private servers) typically need their own action.php URL.
    login_uri: str = ""
    username: str
    password: str | None
    user_id: str
    avatar: str
    bot_mode: BotModes
    pokemon_format: str = ""
    smogon_stats: str = None
    search_time_ms: int
    parallelism: int
    run_count: int
    team_name: str
    team_list: str = None
    user_to_challenge: str
    save_replay: SaveReplay
    room_name: str
    log_level: str
    log_to_file: bool
    stdout_log_handler: logging.StreamHandler
    file_log_handler: Optional[CustomRotatingFileHandler]

    # Gemini integration
    decision_engine: str = "gemini"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.1-pro-preview"
    gemini_tutor_model: str = "gemini-2.0-flash"
    gemini_thinking_budget: int = 1024
    tutor_mode: bool = False

    # Claude integration
    claude_api_key: Optional[str] = None
    claude_model: str = "claude-sonnet-4-6"
    claude_tutor_model: str = "claude-sonnet-4-6"
    claude_thinking_budget: int = 4096

    # DeepSeek integration
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_tutor_model: str = "deepseek-v4-flash"
    deepseek_thinking_budget: int = 4096
    deepseek_reasoning_effort: str = "high"

    def configure(self):
        from pathlib import Path
        env_file = Path(__file__).parent / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))

        def env_int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        def env_truthy(name: str) -> bool:
            return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--websocket-uri",
            default=os.environ.get("PS_WEBSOCKET_URI", "wss://sim3.psim.us/showdown/websocket"),
            help="The PokemonShowdown websocket URI, e.g. wss://sim3.psim.us/showdown/websocket",
        )
        parser.add_argument(
            "--login-uri",
            default=os.environ.get("PS_LOGIN_URI", ""),
            help=(
                "Override the Showdown login URL (action.php). Use this for mirrors "
                "like PokéAgent Challenge that authenticate against their own server. "
                "Leave blank for the default play.pokemonshowdown.com."
            ),
        )
        parser.add_argument("--ps-username", default=os.environ.get("PS_USERNAME"))
        parser.add_argument("--ps-password", default=os.environ.get("PS_PASSWORD", None))
        parser.add_argument("--ps-avatar", default=os.environ.get("PS_AVATAR", None))
        parser.add_argument(
            "--bot-mode", default=os.environ.get("PS_BOT_MODE", "accept_challenge"), choices=[e.name for e in BotModes]
        )
        parser.add_argument(
            "--user-to-challenge",
            default=os.environ.get("PS_USER_TO_CHALLENGE", None),
            help="If bot_mode is `challenge_user`, this is required",
        )
        parser.add_argument(
            "--pokemon-format", default=os.environ.get("PS_FORMAT", "gen9randombattle"), help="e.g. gen9randombattle"
        )
        parser.add_argument(
            "--smogon-stats-format",
            default=os.environ.get("PS_SMOGON_STATS"),
            help="Overwrite which smogon stats are used to infer unknowns. If not set, defaults to the --pokemon-format value.",
        )
        parser.add_argument(
            "--search-time-ms",
            type=int,
            default=env_int("PS_SEARCH_TIME_MS", 300),
            help="Time to search per battle in milliseconds",
        )
        parser.add_argument(
            "--search-parallelism",
            type=int,
            default=env_int("PS_SEARCH_PARALLELISM", 1),
            help="Number of states to search in parallel",
        )
        parser.add_argument(
            "--run-count",
            type=int,
            default=env_int("PS_RUN_COUNT", 1),
            help="Number of PokemonShowdown battles to run",
        )
        parser.add_argument(
            "--team-name",
            default=os.environ.get("PS_TEAM_NAME"),
            help="Which team to use. Can be a filename or a foldername relative to ./teams/teams/. "
            "If a foldername, a random team from that folder will be chosen each battle. "
            "If not set, defaults to the --pokemon-format value.",
        )
        parser.add_argument(
            "--team-list",
            default=os.environ.get("PS_TEAM_LIST"),
            help="A path to a text file containing a list of team names to choose from in order. Takes precedence over --team-name.",
        )
        parser.add_argument(
            "--save-replay",
            default=os.environ.get("PS_SAVE_REPLAY", "never"),
            choices=[e.name for e in SaveReplay],
            help="When to save replays",
        )
        parser.add_argument(
            "--room-name",
            default=os.environ.get("PS_ROOM_NAME"),
            help="If bot_mode is `accept_challenge`, the room to join while waiting",
        )
        parser.add_argument(
            "--log-level",
            default=os.environ.get("PS_LOG_LEVEL", "DEBUG"),
            help="Python logging level",
        )
        parser.add_argument(
            "--log-to-file",
            action="store_true",
            default=env_truthy("PS_LOG_TO_FILE"),
            help="When enabled, DEBUG logs will be written to a file in the logs/ directory",
        )

        # Decision engine + LLM auth (API key only)
        parser.add_argument(
            "--decision-engine",
            default=os.environ.get("DECISION_ENGINE", "gemini"),
            choices=["mcts", "gemini", "claude", "deepseek"],
            help="Which decision engine to use (default: gemini)",
        )
        parser.add_argument(
            "--tutor-mode",
            action="store_true",
            default=env_truthy("TUTOR_MODE"),
            help="Enable chatty tutor mode (post-turn coaching and chat replies)",
        )

        # Gemini
        parser.add_argument(
            "--gemini-api-key",
            default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            help="Gemini API key (also reads GEMINI_API_KEY / GOOGLE_API_KEY env vars)",
        )
        parser.add_argument(
            "--gemini-model",
            default=os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview"),
            help="Gemini model name (default: gemini-3.1-pro-preview)",
        )
        parser.add_argument(
            "--gemini-tutor-model",
            default=os.environ.get("GEMINI_TUTOR_MODEL", "gemini-2.0-flash"),
            help="Gemini model for tutor chat (default: gemini-2.0-flash)",
        )
        parser.add_argument(
            "--gemini-thinking-budget",
            type=int,
            default=env_int("GEMINI_THINKING_BUDGET", 4096),
            help="Thinking token budget for Gemini decisions (default: 4096, 0 to disable)",
        )

        # Claude
        parser.add_argument(
            "--claude-api-key",
            default=os.environ.get("ANTHROPIC_API_KEY"),
            help="Anthropic API key (also reads ANTHROPIC_API_KEY env var)",
        )
        parser.add_argument(
            "--claude-model",
            default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
            help="Claude model name (default: claude-sonnet-4-6)",
        )
        parser.add_argument(
            "--claude-tutor-model",
            default=os.environ.get("CLAUDE_TUTOR_MODEL", "claude-sonnet-4-6"),
            help="Claude model for tutor chat (default: claude-sonnet-4-6)",
        )
        parser.add_argument(
            "--claude-thinking-budget",
            type=int,
            default=env_int("CLAUDE_THINKING_BUDGET", 4096),
            help="Thinking token budget for Claude decisions (default: 4096, 0 to disable)",
        )

        # DeepSeek
        parser.add_argument(
            "--deepseek-api-key",
            default=os.environ.get("DEEPSEEK_API_KEY"),
            help="DeepSeek API key (also reads DEEPSEEK_API_KEY env var)",
        )
        parser.add_argument(
            "--deepseek-model",
            default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            help="DeepSeek model name (default: deepseek-v4-pro)",
        )
        parser.add_argument(
            "--deepseek-tutor-model",
            default=os.environ.get("DEEPSEEK_TUTOR_MODEL", "deepseek-v4-flash"),
            help="DeepSeek model for tutor chat (default: deepseek-v4-flash)",
        )
        parser.add_argument(
            "--deepseek-thinking-budget",
            type=int,
            default=env_int("DEEPSEEK_THINKING_BUDGET", 4096),
            help="Thinking token budget for DeepSeek decisions (default: 4096, 0 to disable)",
        )
        parser.add_argument(
            "--deepseek-reasoning-effort",
            default=os.environ.get("DEEPSEEK_REASONING_EFFORT", "high"),
            choices=["low", "medium", "high", "max", "xhigh"],
            help="DeepSeek reasoning effort level (default: high). low/medium map to high; xhigh maps to max.",
        )

        args = parser.parse_args()

        if not args.ps_username:
            print("\nError: Showdown username is required!")
            print("Set PS_USERNAME in `.env`, pass --ps-username, or open the GUI: `python psymew_gui.py`\n")
            sys.exit(1)

        self.websocket_uri = args.websocket_uri
        self.login_uri = (args.login_uri or "").strip()
        self.username = args.ps_username
        self.password = args.ps_password
        self.avatar = args.ps_avatar
        self.bot_mode = BotModes[args.bot_mode]
        self.pokemon_format = args.pokemon_format
        self.smogon_stats = args.smogon_stats_format
        self.search_time_ms = args.search_time_ms
        self.parallelism = args.search_parallelism
        self.run_count = args.run_count
        self.team_name = args.team_name or self.pokemon_format
        self.team_list = args.team_list
        self.user_to_challenge = args.user_to_challenge
        self.save_replay = SaveReplay[args.save_replay]
        self.room_name = args.room_name
        self.log_level = args.log_level
        self.log_to_file = args.log_to_file

        # Decision engine + shared flags
        self.decision_engine = args.decision_engine
        self.tutor_mode = args.tutor_mode

        # Gemini
        self.gemini_api_key = args.gemini_api_key
        self.gemini_model = args.gemini_model
        self.gemini_tutor_model = args.gemini_tutor_model
        self.gemini_thinking_budget = args.gemini_thinking_budget

        # Claude
        self.claude_api_key = args.claude_api_key
        self.claude_model = args.claude_model
        self.claude_tutor_model = args.claude_tutor_model
        self.claude_thinking_budget = args.claude_thinking_budget

        # DeepSeek
        self.deepseek_api_key = args.deepseek_api_key
        self.deepseek_model = args.deepseek_model
        self.deepseek_tutor_model = args.deepseek_tutor_model
        self.deepseek_thinking_budget = args.deepseek_thinking_budget
        self.deepseek_reasoning_effort = args.deepseek_reasoning_effort

        self.validate_config()

    def requires_team(self) -> bool:
        return not (
            "random" in self.pokemon_format or "battlefactory" in self.pokemon_format
        )

    def validate_config(self):
        if self.bot_mode == BotModes.challenge_user:
            assert (
                self.user_to_challenge is not None
            ), "If bot_mode is `CHALLENGE_USER`, you must declare USER_TO_CHALLENGE"

        logger = logging.getLogger(__name__)
        engine = self.decision_engine
        if engine == "mcts":
            return

        key_envs = _ENGINE_KEY_ENVS.get(engine, ())
        attr_key = {
            "claude": self.claude_api_key,
            "gemini": self.gemini_api_key,
            "deepseek": self.deepseek_api_key,
        }.get(engine)

        has_key = bool((attr_key or "").strip()) or any(
            os.environ.get(name, "").strip() for name in key_envs
        )
        if not has_key:
            env_hint = " / ".join(key_envs) if key_envs else "(no env var)"
            logger.warning(
                "%s API key not detected. Set %s in your .env file or open the GUI: "
                "`python psymew_gui.py`. The bot will fail at runtime without a valid key.",
                engine.capitalize(),
                env_hint,
            )


FoulPlayConfig = _FoulPlayConfig()
