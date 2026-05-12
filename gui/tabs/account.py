"""Account tab — Pokemon Showdown login fields."""

from __future__ import annotations

import webbrowser

import customtkinter as ctk

from gui.state import ConfigState
from gui.theme import (
    COLOR_DIVIDER,
    COLOR_PRIMARY_HOVER,
    COLOR_SECONDARY,
    COLOR_SURFACE_RAISED,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
)


class AccountTab(ctk.CTkScrollableFrame):
    def __init__(self, parent: ctk.CTkBaseClass, cfg_state: ConfigState) -> None:
        super().__init__(parent, fg_color="transparent")
        self.cfg_state = cfg_state

        self.columnconfigure(0, weight=1)
        self._build()
        self.refresh()
        self.cfg_state.subscribe(self._on_state_event)

    def _build(self) -> None:
        ctk.CTkLabel(
            self,
            text="Showdown account",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 2))

        ctk.CTkLabel(
            self,
            text="The credentials the bot logs in with at play.pokemonshowdown.com.",
            text_color=COLOR_TEXT_MUTED,
        ).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 14))

        card = ctk.CTkFrame(self, fg_color=COLOR_SURFACE_RAISED, corner_radius=10)
        card.grid(row=2, column=0, sticky="ew", padx=20, pady=8)
        card.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            card,
            text="Login",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(10, 8))

        # Username
        ctk.CTkLabel(card, text="Username *").grid(
            row=1, column=0, sticky="w", padx=(14, 8), pady=6
        )
        self.username_var = ctk.StringVar()
        ctk.CTkEntry(
            card, textvariable=self.username_var, placeholder_text="ShowdownNick"
        ).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 14), pady=6)
        self.username_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set("PS_USERNAME", self.username_var.get()),
        )

        # Password (with show/hide)
        ctk.CTkLabel(card, text="Password").grid(
            row=2, column=0, sticky="w", padx=(14, 8), pady=6
        )
        self.password_var = ctk.StringVar()
        self.password_entry = ctk.CTkEntry(
            card,
            textvariable=self.password_var,
            show="•",
            placeholder_text="(leave blank for guest login)",
        )
        self.password_entry.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=6)
        self.password_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set("PS_PASSWORD", self.password_var.get()),
        )
        self._password_visible = False
        self.password_toggle = ctk.CTkButton(
            card,
            text="Show",
            width=72,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_DIVIDER,
            text_color=COLOR_TEXT_MUTED,
            command=self._toggle_password,
        )
        self.password_toggle.grid(row=2, column=2, sticky="e", padx=(0, 14), pady=6)

        # Avatar
        ctk.CTkLabel(card, text="Avatar").grid(
            row=3, column=0, sticky="w", padx=(14, 8), pady=6
        )
        self.avatar_var = ctk.StringVar()
        ctk.CTkEntry(
            card,
            textvariable=self.avatar_var,
            placeholder_text="e.g. 'red' or '23' (optional)",
        ).grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=6)
        self.avatar_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set("PS_AVATAR", self.avatar_var.get()),
        )

        ctk.CTkButton(
            card,
            text="Browse avatars",
            width=120,
            fg_color=COLOR_SECONDARY,
            hover_color=COLOR_PRIMARY_HOVER,
            command=lambda: webbrowser.open(
                "https://play.pokemonshowdown.com/sprites/trainers/"
            ),
        ).grid(row=3, column=2, sticky="e", padx=(0, 14), pady=6)

        ctk.CTkLabel(
            card,
            text="No Showdown account yet? Make one at play.pokemonshowdown.com (top-right menu).",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=14, pady=(2, 12))

    def _toggle_password(self) -> None:
        self._password_visible = not self._password_visible
        self.password_entry.configure(show="" if self._password_visible else "•")
        self.password_toggle.configure(text="Hide" if self._password_visible else "Show")

    def refresh(self) -> None:
        self.username_var.set(self.cfg_state.get("PS_USERNAME"))
        self.password_var.set(self.cfg_state.get("PS_PASSWORD"))
        self.avatar_var.set(self.cfg_state.get("PS_AVATAR"))

    def _on_state_event(self, kind: str) -> None:
        if kind in ("reload", "save"):
            self.refresh()
