"""Shared theme constants for the PsyMew GUI.

Mew is a Psychic-type Pokémon; her sprite reads pink/magenta, so the
palette leans into that with a deep dark surface and fuchsia accents.
Constants are referenced explicitly by widgets — no global CTk theme
file required, which keeps the palette readable in one place.
"""

from __future__ import annotations

APPEARANCE_MODE = "dark"
COLOR_THEME = "dark-blue"

WINDOW_TITLE = "PsyMew — Configuration"
WINDOW_SIZE = "1020x760"
WINDOW_MIN_SIZE = (900, 640)

FONT_FAMILY = "Segoe UI"
FONT_MONO = "Cascadia Mono"

# --- Background layers ----------------------------------------------------
COLOR_BG = "#0f0f15"
COLOR_SURFACE = "#181823"
COLOR_SURFACE_RAISED = "#23233a"
COLOR_DIVIDER = "#2d2d44"

# --- Brand: Mew-pink with a psychic-lavender support tone ----------------
COLOR_PRIMARY = "#e85ad6"          # PsyMew pink
COLOR_PRIMARY_HOVER = "#c93cbb"
COLOR_PRIMARY_PRESSED = "#a82a9a"
COLOR_SECONDARY = "#a78bfa"        # psychic lavender
COLOR_SECONDARY_HOVER = "#8b5cf6"
COLOR_ACCENT = "#22d3ee"           # cyan — type-energy contrast

# --- Text -----------------------------------------------------------------
COLOR_TEXT = "#f5f5fb"
COLOR_TEXT_MUTED = "#8a8aa3"
COLOR_TEXT_DIM = "#5a5a72"

# --- Status ---------------------------------------------------------------
COLOR_STATUS_STOPPED = "#6e7681"
COLOR_STATUS_STARTING = "#8b5cf6"
COLOR_STATUS_RUNNING = "#22c55e"
COLOR_STATUS_CRASHED = "#ef4444"

COLOR_OK = "#22c55e"
COLOR_WARN = "#eab308"
COLOR_ERROR = "#ef4444"
COLOR_MUTED = COLOR_TEXT_MUTED  # legacy alias used by older code paths

# --- Branding strings -----------------------------------------------------
BRAND_NAME = "PsyMew"
BRAND_TAGLINE = "Pokémon Showdown battle bot — Claude / Gemini / DeepSeek / MCTS"
BRAND_VERSION = "GUI v0.1"

# Mew unicode glyph (◓ poké-ball + sparkles fits even without an image asset)
BRAND_GLYPH = "◓"
