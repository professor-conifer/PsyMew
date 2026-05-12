"""Advanced tab — Showdown server, logging, replays, and tuning."""

from __future__ import annotations

import webbrowser

import customtkinter as ctk

from gui.formats import SERVER_PRESETS, ServerPreset, preset_for_websocket
from gui.state import ConfigState
from gui.theme import (
    COLOR_DIVIDER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_SECONDARY,
    COLOR_SURFACE_RAISED,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
)

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
SAVE_REPLAY_CHOICES = ["never", "always", "on_win", "on_loss"]

_PRESETS_BY_NAME = {p.name: p for p in SERVER_PRESETS}


class AdvancedTab(ctk.CTkScrollableFrame):
    def __init__(self, parent: ctk.CTkBaseClass, cfg_state: ConfigState) -> None:
        super().__init__(parent, fg_color="transparent")
        self.cfg_state = cfg_state

        self.columnconfigure(0, weight=1)
        self._build()
        self.refresh()
        self.cfg_state.subscribe(self._on_state_event)

    # ---- layout --------------------------------------------------------

    def _build(self) -> None:
        ctk.CTkLabel(
            self,
            text="Advanced",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 2))

        ctk.CTkLabel(
            self,
            text=(
                "Pick a Showdown server, tune the bot's behavior, and decide "
                "how chatty the logs get. Defaults are fine for play.pokemonshowdown.com."
            ),
            text_color=COLOR_TEXT_MUTED,
            wraplength=820,
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 14))

        self._build_server_card(row=2)
        self._build_run_card(row=3)
        self._build_logging_card(row=4)
        self._build_replay_card(row=5)
        self._build_tuning_card(row=6)

    def _make_card(self, title: str, subtitle: str, row: int) -> ctk.CTkFrame:
        card = ctk.CTkFrame(self, fg_color=COLOR_SURFACE_RAISED, corner_radius=10)
        card.grid(row=row, column=0, sticky="ew", padx=20, pady=8)
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(10, 0))
        if subtitle:
            ctk.CTkLabel(
                card,
                text=subtitle,
                text_color=COLOR_TEXT_MUTED,
                font=ctk.CTkFont(size=11),
                wraplength=760,
                justify="left",
                anchor="w",
            ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 6))
        return card

    # ---- cards ---------------------------------------------------------

    def _build_server_card(self, row: int) -> None:
        card = self._make_card(
            "Showdown server",
            "Where the bot connects. Pick a preset, or choose Custom to enter "
            "your own URL. Mirrors (PokéAgent, private servers) usually need "
            "their own login endpoint — the preset fills that in for you.",
            row,
        )

        ctk.CTkLabel(card, text="Preset").grid(
            row=2, column=0, sticky="w", padx=14, pady=4
        )
        self.preset_var = ctk.StringVar(value=SERVER_PRESETS[0].name)
        self.preset_menu = ctk.CTkOptionMenu(
            card,
            values=[p.name for p in SERVER_PRESETS],
            variable=self.preset_var,
            fg_color=COLOR_SECONDARY,
            button_color=COLOR_PRIMARY,
            button_hover_color=COLOR_PRIMARY_HOVER,
            command=self._on_preset_change,
        )
        self.preset_menu.grid(row=2, column=1, sticky="ew", padx=14, pady=4)

        ctk.CTkLabel(card, text="WebSocket URL").grid(
            row=3, column=0, sticky="w", padx=14, pady=4
        )
        self.ws_var = ctk.StringVar()
        self.ws_entry = ctk.CTkEntry(
            card,
            textvariable=self.ws_var,
            placeholder_text="wss://example.com/showdown/websocket",
        )
        self.ws_entry.grid(row=3, column=1, sticky="ew", padx=14, pady=4)
        self.ws_var.trace_add(
            "write", lambda *_: self.cfg_state.set("PS_WEBSOCKET_URI", self.ws_var.get())
        )

        ctk.CTkLabel(card, text="Login URL (action.php)").grid(
            row=4, column=0, sticky="w", padx=14, pady=4
        )
        self.login_var = ctk.StringVar()
        self.login_entry = ctk.CTkEntry(
            card,
            textvariable=self.login_var,
            placeholder_text="(leave blank for the default Showdown login server)",
        )
        self.login_entry.grid(row=4, column=1, sticky="ew", padx=14, pady=4)
        self.login_var.trace_add(
            "write", lambda *_: self.cfg_state.set("PS_LOGIN_URI", self.login_var.get())
        )

        # Preset blurb + link buttons (filled in by _on_preset_change)
        self.preset_blurb = ctk.CTkLabel(
            card,
            text="",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            wraplength=760,
            justify="left",
            anchor="w",
        )
        self.preset_blurb.grid(row=5, column=0, columnspan=2, sticky="ew", padx=14, pady=(8, 4))

        self.preset_links = ctk.CTkFrame(card, fg_color="transparent")
        self.preset_links.grid(row=6, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 12))

    def _build_run_card(self, row: int) -> None:
        card = self._make_card(
            "Run length",
            "How many battles to play in a single bot session before exiting.",
            row,
        )
        ctk.CTkLabel(card, text="Battles per session").grid(
            row=2, column=0, sticky="w", padx=14, pady=(4, 12)
        )
        self.run_count_var = ctk.StringVar(value="1")
        ctk.CTkEntry(card, textvariable=self.run_count_var, width=120).grid(
            row=2, column=1, sticky="w", padx=14, pady=(4, 12)
        )
        self.run_count_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set("PS_RUN_COUNT", self.run_count_var.get()),
        )

    def _build_logging_card(self, row: int) -> None:
        card = self._make_card(
            "Logging",
            "DEBUG is loudest; ERROR is quietest. With 'Write to file' on, the "
            "GUI's live log preview reads from logs/init.log.",
            row,
        )

        ctk.CTkLabel(card, text="Verbosity").grid(
            row=2, column=0, sticky="w", padx=14, pady=4
        )
        self.log_level_var = ctk.StringVar(value="DEBUG")
        ctk.CTkOptionMenu(
            card,
            values=LOG_LEVELS,
            variable=self.log_level_var,
            fg_color=COLOR_SECONDARY,
            button_color=COLOR_PRIMARY,
            button_hover_color=COLOR_PRIMARY_HOVER,
            command=lambda v: self.cfg_state.set("PS_LOG_LEVEL", v),
        ).grid(row=2, column=1, sticky="w", padx=14, pady=4)

        self.log_to_file_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            card,
            text="Write a full DEBUG transcript to logs/init.log",
            variable=self.log_to_file_var,
            progress_color=COLOR_PRIMARY,
            button_color=COLOR_TEXT,
            button_hover_color=COLOR_PRIMARY_HOVER,
            command=lambda: self.cfg_state.set(
                "PS_LOG_TO_FILE", "1" if self.log_to_file_var.get() else ""
            ),
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=14, pady=(4, 12))

    def _build_replay_card(self, row: int) -> None:
        card = self._make_card(
            "Replays",
            "Tells Showdown to save a viewable replay link after each battle. "
            "The link is printed in the bot's log and can be opened in your browser.",
            row,
        )
        ctk.CTkLabel(card, text="When to save").grid(
            row=2, column=0, sticky="w", padx=14, pady=(4, 12)
        )
        self.save_replay_var = ctk.StringVar(value="never")
        ctk.CTkOptionMenu(
            card,
            values=SAVE_REPLAY_CHOICES,
            variable=self.save_replay_var,
            fg_color=COLOR_SECONDARY,
            button_color=COLOR_PRIMARY,
            button_hover_color=COLOR_PRIMARY_HOVER,
            command=lambda v: self.cfg_state.set("PS_SAVE_REPLAY", v),
        ).grid(row=2, column=1, sticky="w", padx=14, pady=(4, 12))

    def _build_tuning_card(self, row: int) -> None:
        card = self._make_card(
            "Power-user overrides",
            "Most users leave these blank. Smogon stats override changes which "
            "tier's usage stats are used to infer unknown opponent sets.",
            row,
        )
        ctk.CTkLabel(card, text="Smogon stats format").grid(
            row=2, column=0, sticky="w", padx=14, pady=(4, 12)
        )
        self.smogon_var = ctk.StringVar()
        ctk.CTkEntry(
            card,
            textvariable=self.smogon_var,
            placeholder_text="optional — e.g. gen9ou (overrides PS_FORMAT)",
        ).grid(row=2, column=1, sticky="ew", padx=14, pady=(4, 12))
        self.smogon_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set("PS_SMOGON_STATS", self.smogon_var.get()),
        )

        reset_row = ctk.CTkFrame(card, fg_color="transparent")
        reset_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 12))
        reset_row.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            reset_row,
            text="All advanced fields can be safely left untouched.",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            reset_row,
            text="Reset to defaults",
            width=140,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_DIVIDER,
            text_color=COLOR_TEXT_MUTED,
            command=self._reset_defaults,
        ).grid(row=0, column=1, sticky="e")

    # ---- behavior ------------------------------------------------------

    def _apply_preset(self, preset: ServerPreset) -> None:
        """Populate widgets + state from a preset and enable/disable URL inputs."""
        is_custom = preset.name == "Custom…"

        if not is_custom:
            self.ws_var.set(preset.websocket)
            self.login_var.set(preset.login or "")
            self.ws_entry.configure(state="disabled")
            self.login_entry.configure(state="disabled")
        else:
            self.ws_entry.configure(state="normal")
            self.login_entry.configure(state="normal")

        self.preset_blurb.configure(text=preset.blurb)

        # Rebuild the link-button row to match this preset.
        for child in self.preset_links.winfo_children():
            child.destroy()
        col = 0
        for label, url in [
            ("Open ladder ↗", preset.leaderboard_url),
            ("Watch replays ↗", preset.replays_url),
            ("Docs ↗", preset.docs_url),
        ]:
            if url:
                ctk.CTkButton(
                    self.preset_links,
                    text=label,
                    width=130,
                    height=26,
                    fg_color="transparent",
                    hover_color=COLOR_DIVIDER,
                    border_width=1,
                    border_color=COLOR_SECONDARY,
                    text_color=COLOR_SECONDARY,
                    font=ctk.CTkFont(size=11),
                    command=lambda u=url: webbrowser.open(u),
                ).grid(row=0, column=col, padx=(0, 6), pady=2, sticky="w")
                col += 1

    def _on_preset_change(self, value: str) -> None:
        preset = _PRESETS_BY_NAME.get(value, SERVER_PRESETS[0])
        self._apply_preset(preset)

    def _reset_defaults(self) -> None:
        official = SERVER_PRESETS[0]
        self.preset_var.set(official.name)
        self._apply_preset(official)
        self.run_count_var.set("1")
        self.log_level_var.set("DEBUG")
        self.cfg_state.set("PS_LOG_LEVEL", "DEBUG")
        self.log_to_file_var.set(True)
        self.cfg_state.set("PS_LOG_TO_FILE", "1")
        self.save_replay_var.set("never")
        self.cfg_state.set("PS_SAVE_REPLAY", "never")
        self.smogon_var.set("")

    def refresh(self) -> None:
        saved_ws = self.cfg_state.get("PS_WEBSOCKET_URI")
        saved_login = self.cfg_state.get("PS_LOGIN_URI")

        if saved_ws:
            preset = preset_for_websocket(saved_ws)
            self.preset_var.set(preset.name)
            # If it's a known preset, reflect its login URL (the user's
            # saved value wins if they typed something custom).
            self.ws_var.set(saved_ws)
            self.login_var.set(saved_login or (preset.login or ""))
            is_custom = preset.name == "Custom…"
            self.ws_entry.configure(state="normal" if is_custom else "disabled")
            self.login_entry.configure(state="normal" if is_custom else "disabled")
            self._apply_preset(preset)
            # _apply_preset overwrites entry values for non-custom presets;
            # for custom we want the user's saved values preserved.
            if is_custom:
                self.ws_var.set(saved_ws)
                self.login_var.set(saved_login)
        else:
            official = SERVER_PRESETS[0]
            self.preset_var.set(official.name)
            self._apply_preset(official)

        self.log_level_var.set(self.cfg_state.get("PS_LOG_LEVEL") or "DEBUG")
        saved_log = self.cfg_state.get("PS_LOG_TO_FILE")
        self.log_to_file_var.set(saved_log != "0" and saved_log != "false")
        self.run_count_var.set(self.cfg_state.get("PS_RUN_COUNT") or "1")
        self.save_replay_var.set(self.cfg_state.get("PS_SAVE_REPLAY") or "never")
        self.smogon_var.set(self.cfg_state.get("PS_SMOGON_STATS"))

    def _on_state_event(self, kind: str) -> None:
        if kind in ("reload", "save"):
            self.refresh()
