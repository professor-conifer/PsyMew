# PsyMew ![mew](https://play.pokemonshowdown.com/sprites/xyani/mew.gif)

A competitive Pokémon battle bot for [Pokémon Showdown](https://pokemonshowdown.com/) powered by **Anthropic Claude**, **Google Gemini**, or **DeepSeek V4** — with a native Rust MCTS fallback when LLMs are unavailable.

PsyMew reads the full battle state, reasons about type matchups, damage calcs, speed tiers, and win conditions, then picks the best move via structured tool use. It supports every generation (Gen 1–9) and the major rulesets: singles, doubles (WIP), VGC, random battles, OU/UU/Ubers, and more.

> Fork of [Foul Play](https://github.com/pmariglia/foul-play) with a complete AI overhaul to the decision-making engine.

📖 **Full documentation lives on the [Wiki](https://github.com/professor-conifer/PsyMew/wiki).** 💬 [Join the Discord.](https://discord.gg/N34QduHdUP)

![PsyMew configuration GUI](https://i.imgur.com/YC8IuAj.png)

---

## Features

- **Configuration GUI** — one window with prereq detection, API-key testing, server presets, and Start/Stop. New users can clone-and-run without touching a `.env` file.
- **AI-powered decisions** — Claude, Gemini, or DeepSeek reads the full battle context and picks moves via structured function calling.
- **Full battle awareness** — turn-by-turn history, type effectiveness, damage estimates, speed ranges.
- **Strategic reasoning** — matchup analysis, switch prediction, stall breaking, endgame plays.
- **Tutor Mode (WIP)** — the bot coaches you in chat: greets opponents, comments on turns, gives post-game reviews.
- **Smart fallback** — the native MCTS engine kicks in automatically when the LLM is rate-limited or unavailable.

## Decision engines

| Engine | What it is |
|--------|-------------|
| **Claude** *(default in `.env-example`)* | Anthropic Claude with extended-thinking tool use |
| **Gemini** | Google Gemini Pro with function calling |
| **DeepSeek** | DeepSeek V4 reasoning model (OpenAI-compatible API) |
| **MCTS** | Native Rust [poke-engine](https://github.com/pmariglia/poke-engine) Monte Carlo Tree Search — no API key needed |

Pick one in the GUI's **AI** tab, or set `DECISION_ENGINE=` in `.env`.

---

## Quick start

### Prerequisites

| Tool | Why |
|------|-----|
| **Python 3.11+** | Runtime |
| **Rust** *(optional — only for the MCTS engine)* | Compiles `poke-engine` from source |
| **C++ Build Tools** *(Windows only, optional)* | Same — only needed if you compile MCTS locally |

The GUI's **Setup tab** detects all of these and offers one-click installers (via `winget` on Windows). You can skip the manual install steps.

### Run the GUI

```bash
git clone https://github.com/professor-conifer/PsyMew.git
cd PsyMew
python psymew_gui.py
```

The window opens on the **Setup** tab. It scans your machine, lights up green for what's already installed, and gives you install buttons for anything missing. When the lights are green, click **Continue to bot setup →** and walk through Account → Battle → AI → Advanced → Run.

A deeper tour of every tab lives at [`docs/gui-overview.md`](docs/gui-overview.md) and on the [Wiki](https://github.com/professor-conifer/PsyMew/wiki/GUI-Overview).

### Run from the terminal (power-user path)

```bash
cp .env-example .env
# edit .env: set PS_USERNAME, PS_PASSWORD, DECISION_ENGINE, and the matching API key
pip install -r requirements.txt
python start.py
```

API keys are issued at:

- Claude — <https://console.anthropic.com/settings/keys>
- Gemini — <https://aistudio.google.com/apikey>
- DeepSeek — <https://platform.deepseek.com/api_keys>

`python claude_login.py` / `python gemini_login.py` are interactive helpers that test a key and write it into `.env` for you.

---

## Servers & mirrors

PsyMew connects to the official `play.pokemonshowdown.com` server by default. It also works with mirrors — pick a preset in the GUI's **Advanced → Showdown server** dropdown, or set `PS_WEBSOCKET_URI` + `PS_LOGIN_URI` in `.env`. See the [Custom Servers wiki page](https://github.com/professor-conifer/PsyMew/wiki/Custom-Servers) and the dedicated [PokéAgent Challenge wiki page](https://github.com/professor-conifer/PsyMew/wiki/PokeAgent-Challenge) for setup walkthroughs.

---

## Docker

```bash
docker build -t psymew:latest .
docker run --rm --network host --env-file .env psymew:latest
```

API keys are read from `.env` — no credential volumes to mount.

---

## CLI reference

All GUI settings are also `start.py` flags. `python start.py --help` lists them. Example:

```bash
python start.py \
  --ps-username MyBot \
  --ps-password secret \
  --decision-engine claude \
  --claude-api-key sk-ant-… \
  --pokemon-format gen9ou \
  --bot-mode search_ladder
```

---

## How the process model works (quick reference)

Running PsyMew spawns **two** Python processes by design:

1. The **bot** (`showdown.py`) — plays the battle.
2. A **detached loader** that keeps the native Rust battle engine cached in memory across runs.

The GUI's launch closes the bot's console window to stop it; the loader is deliberately left alive so the next Start is faster. Restart the GUI to clean up accumulated loaders. The full explanation is on the [Wiki FAQ](https://github.com/professor-conifer/PsyMew/wiki/FAQ).

---

## License

See [LICENSE](LICENSE) for details.
