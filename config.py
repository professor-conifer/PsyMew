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


class _FoulPlayConfig:
    websocket_uri: str
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
    gemini_auth_mode: str = "auto"
    gemini_model: str = "gemini-3.1-pro-preview"
    gemini_tutor_model: str = "gemini-2.0-flash"
    gemini_thinking_budget: int = 1024
    tutor_mode: bool = False

    def configure(self):
        from pathlib import Path
        env_file = Path(__file__).parent / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--websocket-uri",
            default=os.environ.get("PS_WEBSOCKET_URI", "wss://sim3.psim.us/showdown/websocket"),
            help="The PokemonShowdown websocket URI, e.g. wss://sim3.psim.us/showdown/websocket",
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
            default=None,
            help="Overwrite which smogon stats are used to infer unknowns. If not set, defaults to the --pokemon-format value.",
        )
        parser.add_argument(
            "--search-time-ms",
            type=int,
            default=100,
            help="Time to search per battle in milliseconds",
        )
        parser.add_argument(
            "--search-parallelism",
            type=int,
            default=1,
            help="Number of states to search in parallel",
        )
        parser.add_argument(
            "--run-count",
            type=int,
            default=1,
            help="Number of PokemonShowdown battles to run",
        )
        parser.add_argument(
            "--team-name",
            default=None,
            help="Which team to use. Can be a filename or a foldername relative to ./teams/teams/. "
            "If a foldername, a random team from that folder will be chosen each battle. "
            "If not set, defaults to the --pokemon-format value.",
        )
        parser.add_argument(
            "--team-list",
            default=None,
            help="A path to a text file containing a list of team names to choose from in order. Takes precedence over --team-name.",
        )
        parser.add_argument(
            "--save-replay",
            default="never",
            choices=[e.name for e in SaveReplay],
            help="When to save replays",
        )
        parser.add_argument(
            "--room-name",
            default=None,
            help="If bot_mode is `accept_challenge`, the room to join while waiting",
        )
        parser.add_argument("--log-level", default="DEBUG", help="Python logging level")
        parser.add_argument(
            "--log-to-file",
            action="store_true",
            help="When enabled, DEBUG logs will be written to a file in the logs/ directory",
        )

        # Gemini integration
        parser.add_argument(
            "--decision-engine",
            default=os.environ.get("DECISION_ENGINE", "gemini"),
            choices=["mcts", "gemini", "claude"],
            help="Which decision engine to use (default: gemini)",
        )
        parser.add_argument(
            "--gemini-api-key",
            default=None,
            help="Gemini API key (also reads GEMINI_API_KEY env var)",
        )
        parser.add_argument(
            "--gemini-auth-mode",
            default=os.environ.get("GEMINI_AUTH_MODE", "auto"),
            choices=["auto", "api_key", "access_token", "adc", "oauth"],
            help="Gemini auth mode. 'oauth' uses Google login (run `python gemini_login.py` to set up). Default 'auto' tries oauth first. (default: auto)",
        )
        parser.add_argument(
            "--gemini-model",
            default="gemini-3.1-pro-preview",
            help="Gemini model name (default: gemini-3.1-pro-preview)",
        )
        parser.add_argument(
            "--gemini-tutor-model",
            default="gemini-2.0-flash",
            help="Gemini model for tutor chat (default: gemini-2.0-flash, no thinking tokens)",
        )
        parser.add_argument(
            "--tutor-mode",
            action="store_true",
            default=os.environ.get("TUTOR_MODE") == "1",
            help="Enable chatty tutor mode (post-turn coaching and chat replies)",
        )

        # Claude integration
        parser.add_argument(
            "--claude-api-key",
            default=os.environ.get("ANTHROPIC_API_KEY"),
            help="Anthropic API key (also reads ANTHROPIC_API_KEY env var)",
        )
        parser.add_argument(
            "--claude-auth-mode",
            default=os.environ.get("CLAUDE_AUTH_MODE", "auto"),
            choices=["auto", "api_key", "oauth"],
            help="Claude auth mode. 'oauth' reuses Claude Code credentials. Default 'auto' tries oauth first. (default: auto)",
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
            "--gemini-thinking-budget",
            type=int,
            default=int(os.environ.get("GEMINI_THINKING_BUDGET", "4096")),
            help="Thinking token budget for Gemini decisions (default: 4096, 0 to disable)",
        )
        parser.add_argument(
            "--claude-thinking-budget",
            type=int,
            default=int(os.environ.get("CLAUDE_THINKING_BUDGET", "4096")),
            help="Thinking token budget for Claude decisions (default: 4096, 0 to disable)",
        )

        args = parser.parse_args()
        
        if not args.ps_username:
            print("\nError: Showdown username is required!")
            print("Please create a `.env` file with `PS_USERNAME=your_username` or use `--ps-username`.\n")
            sys.exit(1)
            
        self.websocket_uri = args.websocket_uri
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

        # Gemini
        self.decision_engine = args.decision_engine
        self.gemini_api_key = args.gemini_api_key
        self.gemini_auth_mode = args.gemini_auth_mode
        self.gemini_model = args.gemini_model
        self.gemini_tutor_model = args.gemini_tutor_model
        self.gemini_thinking_budget = args.gemini_thinking_budget
        self.tutor_mode = args.tutor_mode

        # Claude
        self.claude_api_key = args.claude_api_key
        self.claude_auth_mode = args.claude_auth_mode
        self.claude_model = args.claude_model
        self.claude_tutor_model = args.claude_tutor_model
        self.claude_thinking_budget = args.claude_thinking_budget

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

        if self.decision_engine == "gemini" or self.tutor_mode:
            has_key = self.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            has_token = os.environ.get("GEMINI_ACCESS_TOKEN")
            has_adc = any(
                os.path.isfile(p)
                for p in self._adc_paths()
            )
            has_oauth = any(
                os.path.isfile(str(p))
                for p in self._oauth_paths()
            )
            if not (has_key or has_token or has_adc or has_oauth):
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Gemini credentials not detected. Run `gemini` CLI and 'Login with Google', "
                    "or run `python gemini_login.py` to set up. "
                    "Bot will fail at runtime if no credentials are available."
                )

    @staticmethod
    def _adc_paths() -> list:
        import pathlib
        paths = []
        explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if explicit:
            paths.append(explicit)
        paths.append(str(pathlib.Path.home() / ".config" / "gcloud" / "application_default_credentials.json"))
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            paths.append(str(pathlib.Path(appdata) / "gcloud" / "application_default_credentials.json"))
        return paths

    @staticmethod
    def _oauth_paths() -> list:
        import pathlib
        paths = []
        explicit = os.environ.get("GEMINI_OAUTH_CREDS_FILE", "").strip()
        if explicit:
            paths.append(explicit)
        home = pathlib.Path.home()
        paths.append(str(home / ".gemini" / "oauth_creds.json"))
        userprofile = os.environ.get("USERPROFILE", "").strip()
        if userprofile:
            paths.append(str(pathlib.Path(userprofile) / ".gemini" / "oauth_creds.json"))
        return paths


FoulPlayConfig = _FoulPlayConfig()
