# PsyMew GUI overview

`python psymew_gui.py` opens a six-tab window. Each tab maps directly to a section of `.env` plus a couple of action buttons. The GUI is a thin shell over `gui.state.ConfigState`, which round-trips a `.env` file on disk while preserving comments and key order.

## Tabs

### 1. Setup *(landing page)*
Detects host-side prerequisites that pip can't install — Python ≥ 3.11, Rust toolchain, Visual Studio Build Tools (Windows), and every package in `requirements.txt`. Each missing item has a one-click install button (`winget` on Windows; the official downloader page elsewhere). Install output streams into the in-tab log textbox in real time.

### 2. Account
Pokémon Showdown login credentials — username, password, optional avatar. Writes `PS_USERNAME`, `PS_PASSWORD`, `PS_AVATAR`.

### 3. Battle
Bot mode (accept_challenge / challenge_user / search_ladder), format, target user, lobby room, and team selection. The team selector uses a native file dialog. The format dropdown is **server-aware** — switching the server preset in Advanced re-filters the available formats.

### 4. AI
Picks the decision engine (MCTS, Claude, Gemini, DeepSeek). LLM panels include an API-key field with show/hide toggle, model + tutor-model combos, thinking budget, and a `Test connection` button that hits the provider's `models.list` endpoint.

### 5. Advanced
- **Server preset:** Official Showdown, Smogtours, PokéAgent Challenge, Custom. Selecting a preset auto-fills both `PS_WEBSOCKET_URI` and `PS_LOGIN_URI`.
- Logging verbosity, write-to-file, run count, replay-save policy, Smogon-stats override.

### 6. Run
Big status pill (Stopped / Starting / Running / Crashed) + W/L counter. **Save .env** flashes a green toast on success and a persistent red toast on failure. **Start bot** Popens `python -u start.py` with `CREATE_NEW_CONSOLE` so the bot has its own visible terminal — closing that window stops the bot. The Run tab tails `logs/init.log` for the live preview.

## State flow

```
+------------+      +-----------+      +-----------+
| .env file  | <--> | ConfigState| <-> | each tab  |
+------------+      +-----------+      +-----------+
                          |
                          | subscribe(kind: "change" | "save" | "reload")
                          v
                    +---------------+
                    | tab.refresh() |
                    +---------------+
```

`gui.state.ConfigState` is the single source of truth in-memory. Tabs read on init and on `"reload"` / `"save"` events, and write back through `state.set(key, value)` from `trace_add("write", …)` callbacks on their `StringVar`s.

## Process model (critical)

The Run tab launches the bot via:

```python
subprocess.Popen(
    [sys.executable, "-u", "start.py"],
    cwd=repo_root,
    env=os.environ.copy() | {"PS_LOG_TO_FILE": "1"},
    creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
)
```

`start.py` *is the launch path* — the GUI never imports `showdown.py` directly. This is essential because `start.py` Popens a detached native-engine loader (in its own process group) before its `os.execv` into `showdown.py`. The bot PID = `proc.pid`. The loader PID is invisible to the GUI and is deliberately left alone, even at GUI shutdown.
