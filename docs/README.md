# PsyMew documentation

Supplementary docs for PsyMew. Most users should start at the [top-level README](../README.md) or just open the GUI: `python psymew_gui.py`.

## Contents

| File | What's in it |
|---|---|
| `gui-overview.md` | A short tour of the configuration GUI and the role of each tab. |
| `screenshots/` | Reserved for screenshots referenced from the top-level README. |

## Suggested reading order for new contributors

1. Skim the top-level [`README.md`](../README.md).
2. Open the GUI and click through every tab — `gui/tabs/*.py` is small and easy to follow.
3. Look at `start.py` to understand the launcher / loader split (do not modify it).
4. `showdown.py` → `fp/run_battle.py` → `fp/{claude,gemini,deepseek}/decision.py` is the request path for a single battle turn.
