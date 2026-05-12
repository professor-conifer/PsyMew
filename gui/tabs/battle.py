"""Battle tab — bot mode, format, and team selection."""

from __future__ import annotations

import tkinter.filedialog as filedialog
from pathlib import Path

import customtkinter as ctk

from gui.formats import (
    SHOWDOWN_FORMATS,
    formats_for_server,
    is_teamless_format,
    preset_for_websocket,
)
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


class BattleTab(ctk.CTkScrollableFrame):
    def __init__(
        self,
        parent: ctk.CTkBaseClass,
        cfg_state: ConfigState,
        repo_root: Path,
    ) -> None:
        super().__init__(parent, fg_color="transparent")
        self.cfg_state = cfg_state
        self.repo_root = repo_root
        # Remember which server we're currently showing formats for so we
        # can re-filter when the Advanced tab changes PS_WEBSOCKET_URI.
        self._last_known_server = ""

        self.columnconfigure(1, weight=1)
        self._build()
        self.refresh()
        self.cfg_state.subscribe(self._on_state_event)

    # ---- layout --------------------------------------------------------

    def _build(self) -> None:
        ctk.CTkLabel(
            self,
            text="Battle setup",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=20, pady=(20, 2))

        ctk.CTkLabel(
            self,
            text="What format the bot plays and how it enters battles.",
            text_color=COLOR_TEXT_MUTED,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 14))

        # ---- Card 1: Bot mode -----------------------------------------
        mode_card = self._make_card("How the bot finds battles", row=2)
        mode_card.columnconfigure(0, weight=1)

        self.mode_var = ctk.StringVar(value="accept_challenge")
        mode_frame = ctk.CTkFrame(mode_card, fg_color="transparent")
        mode_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 12))
        for i, (val, label, sub) in enumerate(
            [
                (
                    "accept_challenge",
                    "Accept challenges",
                    "Waits in a chat room until someone challenges the bot.",
                ),
                (
                    "challenge_user",
                    "Challenge a user",
                    "Sends a challenge to a specific username.",
                ),
                (
                    "search_ladder",
                    "Search the ladder",
                    "Hops on the ranked queue and plays whoever it matches with.",
                ),
            ]
        ):
            row = ctk.CTkFrame(mode_frame, fg_color="transparent")
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.columnconfigure(1, weight=1)
            ctk.CTkRadioButton(
                row,
                text=label,
                variable=self.mode_var,
                value=val,
                fg_color=COLOR_PRIMARY,
                hover_color=COLOR_PRIMARY_HOVER,
                command=self._on_mode_change,
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                row, text=sub, text_color=COLOR_TEXT_MUTED, font=ctk.CTkFont(size=11)
            ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        # Conditional fields tied to mode
        ctk.CTkLabel(mode_card, text="User to challenge").grid(
            row=2, column=0, sticky="w", padx=14, pady=(4, 2)
        )
        self.user_var = ctk.StringVar()
        self.user_entry = ctk.CTkEntry(
            mode_card,
            textvariable=self.user_var,
            placeholder_text="Showdown username (only used in Challenge mode)",
        )
        self.user_entry.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 6))
        self.user_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set("PS_USER_TO_CHALLENGE", self.user_var.get()),
        )

        ctk.CTkLabel(mode_card, text="Lobby room").grid(
            row=4, column=0, sticky="w", padx=14, pady=(4, 2)
        )
        self.room_var = ctk.StringVar()
        self.room_entry = ctk.CTkEntry(
            mode_card,
            textvariable=self.room_var,
            placeholder_text="Optional — e.g. 'lobby'. Only used in Accept mode.",
        )
        self.room_entry.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.room_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set("PS_ROOM_NAME", self.room_var.get()),
        )

        # ---- Card 2: Format -------------------------------------------
        fmt_card = self._make_card("Battle format", row=3)
        fmt_card.columnconfigure(0, weight=1)

        self.format_var = ctk.StringVar(value="gen9randombattle")
        self.format_combo = ctk.CTkComboBox(
            fmt_card,
            values=SHOWDOWN_FORMATS,
            variable=self.format_var,
            command=self._on_format_change,
        )
        self.format_combo.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 4))

        # This label is updated by _apply_format_list_for_current_server.
        self.format_hint = ctk.CTkLabel(
            fmt_card,
            text="",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            wraplength=720,
            justify="left",
            anchor="w",
        )
        self.format_hint.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 12))

        # ---- Card 3: Team ---------------------------------------------
        team_card = self._make_card("Team file", row=4)
        team_card.columnconfigure(0, weight=1)

        self.team_status = ctk.CTkLabel(
            team_card,
            text="",
            text_color=COLOR_TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=720,
        )
        self.team_status.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 6))

        row_widgets = ctk.CTkFrame(team_card, fg_color="transparent")
        row_widgets.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 6))
        row_widgets.columnconfigure(0, weight=1)

        self.team_var = ctk.StringVar()
        self.team_entry = ctk.CTkEntry(
            row_widgets,
            textvariable=self.team_var,
            placeholder_text="Click Browse to pick a Showdown export file…",
        )
        self.team_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.team_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set("PS_TEAM_NAME", self.team_var.get()),
        )

        self.team_browse_btn = ctk.CTkButton(
            row_widgets,
            text="Browse file…",
            width=130,
            fg_color=COLOR_SECONDARY,
            hover_color=COLOR_PRIMARY_HOVER,
            command=self._browse_team_file,
        )
        self.team_browse_btn.grid(row=0, column=1)

        self.team_clear_btn = ctk.CTkButton(
            row_widgets,
            text="Clear",
            width=70,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_DIVIDER,
            text_color=COLOR_TEXT_MUTED,
            command=self._clear_team,
        )
        self.team_clear_btn.grid(row=0, column=2, padx=(8, 0))

        # Team list (advanced)
        ctk.CTkLabel(
            team_card,
            text="Team rotation list (optional)",
            text_color=COLOR_TEXT_MUTED,
        ).grid(row=3, column=0, sticky="w", padx=14, pady=(10, 2))

        list_row = ctk.CTkFrame(team_card, fg_color="transparent")
        list_row.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 12))
        list_row.columnconfigure(0, weight=1)

        self.team_list_var = ctk.StringVar()
        self.team_list_entry = ctk.CTkEntry(
            list_row,
            textvariable=self.team_list_var,
            placeholder_text="(optional) path to a text file listing teams to cycle through",
        )
        self.team_list_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.team_list_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set("PS_TEAM_LIST", self.team_list_var.get()),
        )

        ctk.CTkButton(
            list_row,
            text="Browse…",
            width=110,
            fg_color=COLOR_SECONDARY,
            hover_color=COLOR_PRIMARY_HOVER,
            command=self._browse_team_list,
        ).grid(row=0, column=1)

    # ---- helpers -------------------------------------------------------

    def _make_card(self, title: str, row: int) -> ctk.CTkFrame:
        card = ctk.CTkFrame(self, fg_color=COLOR_SURFACE_RAISED, corner_radius=10)
        card.grid(row=row, column=0, columnspan=3, sticky="ew", padx=20, pady=8)
        card.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 6))
        return card

    def _browse_team_file(self) -> None:
        initial = self.repo_root / "teams" / "teams"
        if not initial.is_dir():
            initial = self.repo_root
        chosen = filedialog.askopenfilename(
            title="Pick your Showdown team export",
            initialdir=str(initial),
            filetypes=[
                ("Showdown team export", "*.txt"),
                ("Team file", "*.team"),
                ("All files", "*.*"),
            ],
        )
        if chosen:
            self._set_team_path(chosen)

    def _browse_team_list(self) -> None:
        initial = self.repo_root / "teams"
        if not initial.is_dir():
            initial = self.repo_root
        chosen = filedialog.askopenfilename(
            title="Pick a team rotation list",
            initialdir=str(initial),
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
        )
        if chosen:
            self.team_list_var.set(chosen)

    def _set_team_path(self, path: str) -> None:
        # Store as a path relative to teams/teams/ when possible so the
        # config is portable across machines. Fall back to absolute when
        # the user picked something outside the project tree.
        candidate = Path(path).resolve()
        teams_root = (self.repo_root / "teams" / "teams").resolve()
        try:
            relative = candidate.relative_to(teams_root)
            stored = str(relative).replace("\\", "/")
        except ValueError:
            stored = str(candidate)
        self.team_var.set(stored)

    def _clear_team(self) -> None:
        self.team_var.set("")

    # ---- state events --------------------------------------------------

    def _on_mode_change(self) -> None:
        self.cfg_state.set("PS_BOT_MODE", self.mode_var.get())
        self._update_field_enabled_state()

    def _on_format_change(self, _value: str) -> None:
        self.cfg_state.set("PS_FORMAT", self.format_var.get())
        self._update_field_enabled_state()

    def _update_field_enabled_state(self) -> None:
        mode = self.mode_var.get()
        fmt = self.format_var.get()
        teamless = is_teamless_format(fmt)

        self.user_entry.configure(
            state="normal" if mode == "challenge_user" else "disabled"
        )
        self.room_entry.configure(
            state="normal" if mode == "accept_challenge" else "disabled"
        )

        team_state = "disabled" if teamless else "normal"
        for widget in (self.team_entry, self.team_browse_btn, self.team_clear_btn, self.team_list_entry):
            widget.configure(state=team_state)

        if teamless:
            self.team_status.configure(
                text="No team file needed — this format generates one for you.",
                text_color=COLOR_TEXT_MUTED,
            )
        else:
            current = self.team_var.get().strip()
            if current:
                self.team_status.configure(
                    text=f"Using: {current}",
                    text_color=COLOR_TEXT,
                )
            else:
                self.team_status.configure(
                    text="No team chosen yet — click Browse to pick a .txt team export.",
                    text_color=COLOR_TEXT_MUTED,
                )

    def refresh(self) -> None:
        self.mode_var.set(self.cfg_state.get("PS_BOT_MODE") or "accept_challenge")
        self.format_var.set(self.cfg_state.get("PS_FORMAT") or "gen9randombattle")
        self.user_var.set(self.cfg_state.get("PS_USER_TO_CHALLENGE"))
        self.room_var.set(self.cfg_state.get("PS_ROOM_NAME"))
        self.team_var.set(self.cfg_state.get("PS_TEAM_NAME"))
        self.team_list_var.set(self.cfg_state.get("PS_TEAM_LIST"))
        self._apply_format_list_for_current_server()
        self._update_field_enabled_state()

    def _apply_format_list_for_current_server(self) -> None:
        """Re-filter the format dropdown based on the selected Showdown server."""
        ws = self.cfg_state.get("PS_WEBSOCKET_URI")
        self._last_known_server = ws
        allowed = formats_for_server(ws)
        self.format_combo.configure(values=allowed)

        preset = preset_for_websocket(ws) if ws else None
        if preset and preset.formats is not None:
            # Restricted server — snap the current format into the allowed
            # set if the user previously had something this server doesn't run.
            current = self.format_var.get()
            if current and current not in allowed:
                fallback = allowed[0]
                self.format_var.set(fallback)
                self.cfg_state.set("PS_FORMAT", fallback)
            self.format_hint.configure(
                text=(
                    f"Showing only formats supported on {preset.name}. "
                    f"Switch server in the Advanced tab to see other formats."
                ),
            )
        else:
            self.format_hint.configure(
                text="Type a custom format if yours isn't listed (e.g. gen8anythinggoes)."
            )

    def _on_state_event(self, kind: str) -> None:
        if kind in ("reload", "save"):
            self.refresh()
            return
        # On every change event, see if the server moved under us. If so,
        # re-filter the format dropdown without bulldozing the rest of the
        # tab (which would fight the user mid-typing).
        current_server = self.cfg_state.get("PS_WEBSOCKET_URI")
        if current_server != self._last_known_server:
            self._apply_format_list_for_current_server()
        self._update_field_enabled_state()
