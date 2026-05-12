"""PsyMew configuration GUI entry point.

Run with:
    python psymew_gui.py

The window edits the project's .env file, lets you pick a decision engine
and paste its API key, and starts/stops the bot via start.py.

start.py is the required launch path — the GUI shells out to it and never
bypasses it, because start.py also spawns the native poke-engine loader.
"""

from __future__ import annotations

from pathlib import Path

from gui.app import launch


if __name__ == "__main__":
    launch(Path(__file__).resolve().parent)
