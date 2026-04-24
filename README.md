# PsyMew ![mew](https://play.pokemonshowdown.com/sprites/xyani/mew.gif)

A competitive Pokémon battle-bot for [Pokemon Showdown](https://pokemonshowdown.com/) powered by **Google Gemini** or **Anthropic Claude**.

PsyMew uses large language models as its decision engine — it reads the full battle state, reasons about type matchups, damage calcs, speed tiers, and win conditions, then picks the best move via function calling. It supports every format: singles, doubles (WIP), VGC, random battles, and more across all generations. Feel free to contribute to the project!

> Fork of [Foul Play](https://github.com/pmariglia/foul-play) with a complete AI overhaul to the decision making battle engine.

---

## Features

- **AI-powered decisions** — Gemini or Claude reads the full battle context and picks moves via structured tool use
- **Full battle awareness** — Turn-by-turn history, type effectiveness, damage estimates, speed ranges
- **Strategic reasoning** — Matchup analysis, switching prediction, stall breaking, endgame plays
- **Tutor Mode (WIP)** — The bot coaches you in chat: greets opponents, comments on turns, gives post-game reviews
- **Format verification** — Live web search to verify metagame rules at the start of each battle
- **Smart fallback** — MCTS engine kicks in automatically if the AI is rate-limited or unavailable
- **Multi-auth** — Google OAuth, API keys, Claude Code OAuth (Pro/Max), or Application Default Credentials

## Decision Engines

| Engine | Description |
|--------|-------------|
| **Gemini** (default) | Google Gemini 2.5 Pro with function calling |
| **Claude** | Anthropic Claude Sonnet 4 with tool use |
| **MCTS** | [poke-engine](https://github.com/pmariglia/poke-engine) Monte Carlo Tree Search |

Set the engine in your `.env`:
```ini
DECISION_ENGINE=claude   # or claude, mcts
```

---

## Quick Start

### Prerequisites

| Tool | Why | Install |
|------|-----|---------|
| **Python 3.11+** | Runtime | [python.org](https://www.python.org/downloads/) — **check "Add to PATH" on Windows** |
| **Rust** | Builds the MCTS engine | [rustup.rs](https://rustup.rs/) |
| **C++ Build Tools** (Windows) | Native compilation | [VS Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) — select "Desktop development with C++" |

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/PsyMew.git
cd PsyMew

# 2. Install dependencies (takes 1-2 min to compile the engine)
pip install -r requirements.txt

# 3. Create your config
cp .env-example .env
# Edit .env with your Showdown username/password

# 4. Connect to your AI (pick one)

# Option A: Gemini (recommended for Google AI Pro subscribers)
python gemini_login.py

# Option B: Claude (recommended for Claude Pro/Max subscribers)
python claude_login.py

# Option C: API key
# Add GEMINI_API_KEY=... or ANTHROPIC_API_KEY=... to .env

# 5. Run
python start.py
```

Challenge your bot on Showdown and watch it play!

### `.env` Configuration

```ini
PS_USERNAME=your_bot_name
PS_PASSWORD=your_password
PS_BOT_MODE=accept_challenge
PS_FORMAT=gen9randombattle
# TUTOR_MODE=0   // Should be disabled by default unless you are a developer since it does not work properly yet

# AI engine (gemini, claude, or mcts)
DECISION_ENGINE=claude

# Gemini auth (if not using OAuth)
# GEMINI_API_KEY=your_key

# Claude auth (if not using Claude Code OAuth)
# ANTHROPIC_API_KEY=your_key
```

---

## Authentication

### Gemini

| Priority | Mode | How |
|----------|------|-----|
| 1 | OAuth | `python gemini_login.py` — uses your Google AI Pro subscription |
| 2 | API key | `GEMINI_API_KEY` in `.env` |
| 3 | Access token | `GEMINI_ACCESS_TOKEN` env var |
| 4 | ADC | `gcloud` Application Default Credentials |

### Claude

| Priority | Mode | How |
|----------|------|-----|
| 1 | OAuth | Auto-detects [Claude Code](https://code.claude.com) credentials — uses your Pro/Max subscription |
| 2 | API key | `ANTHROPIC_API_KEY` in `.env` |

Run `python claude_login.py` to set up. If you have [Claude Code](https://code.claude.com) installed and logged in, credentials are auto-detected — no extra setup needed.

---

## Docker

```bash
# Build
docker build -t psymew:latest .

# Run with .env file
docker run --rm --network host --env-file .env psymew:latest

# Gemini OAuth (mount credentials)
docker run --rm --network host \
  -v ~/.psymew:/root/.psymew:ro \
  --env-file .env \
  psymew:latest

# Claude OAuth (mount credentials)
docker run --rm --network host \
  -v ~/.claude:/root/.claude:ro \
  --env-file .env \
  psymew:latest
```

---

## MCTS Engine

PsyMew includes [poke-engine](https://github.com/pmariglia/poke-engine) as a fallback MCTS engine. It activates automatically when the AI is unavailable, or you can use it directly:

```ini
DECISION_ENGINE=mcts
```

The engine compiles from source (requires Rust). To rebuild for a different generation:

```bash
pip uninstall -y poke-engine && pip install -v --force-reinstall --no-cache-dir \
  poke-engine --config-settings="build-args=--features poke-engine/gen4 --no-default-features"
```

See the [poke-engine docs](https://poke-engine.readthedocs.io/en/latest/) for details.

---

## CLI Reference

All settings can also be passed as flags. Run `python start.py --help` for the full list.

```bash
python start.py \
  --ps-username MyBot \
  --ps-password secret \
  --decision-engine claude \
  --pokemon-format gen9ou \
  --bot-mode search_ladder
```

---

## License

See [LICENSE](LICENSE) for details.
