"""Main PsyMew GUI window — banner + 6 tabs + persistent footer.

Tab order is deliberate: **Setup** is first so a new clone-and-run user
lands on the prereq detection page before they go fiddle with API keys
and find out poke-engine couldn't compile.
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import customtkinter as ctk

from gui.state import ConfigState
from gui.tabs import AccountTab, AdvancedTab, AITab, BattleTab, RunTab, SetupTab
from gui.theme import (
    APPEARANCE_MODE,
    BRAND_GLYPH,
    BRAND_NAME,
    BRAND_TAGLINE,
    BRAND_VERSION,
    COLOR_BG,
    COLOR_DIVIDER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    COLOR_SURFACE_RAISED,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_THEME,
    WINDOW_MIN_SIZE,
    WINDOW_SIZE,
    WINDOW_TITLE,
)


class App(ctk.CTk):
    def __init__(self, repo_root: Path) -> None:
        super().__init__(fg_color=COLOR_BG)
        self.repo_root = repo_root
        self.cfg_state = ConfigState(repo_root / ".env")

        self.title(WINDOW_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(*WINDOW_MIN_SIZE)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        self._build_banner()
        self._build_tabs()
        self._build_footer()

        self.cfg_state.subscribe(self._on_state_event)
        self._refresh_dirty()
        self.tabview.set("Setup")

    def _build_banner(self) -> None:
        banner = ctk.CTkFrame(self, fg_color=COLOR_SURFACE, corner_radius=0)
        banner.grid(row=0, column=0, sticky="ew")
        banner.columnconfigure(1, weight=1)

        # Brand glyph in a fuchsia chip
        glyph_chip = ctk.CTkLabel(
            banner,
            text=BRAND_GLYPH,
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLOR_PRIMARY,
            width=56,
            height=56,
        )
        glyph_chip.grid(row=0, column=0, rowspan=2, padx=(18, 12), pady=12)

        ctk.CTkLabel(
            banner,
            text=BRAND_NAME,
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).grid(row=0, column=1, sticky="sw", pady=(14, 0))

        ctk.CTkLabel(
            banner,
            text=BRAND_TAGLINE,
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_MUTED,
            anchor="w",
        ).grid(row=1, column=1, sticky="nw", pady=(0, 12))

        right_box = ctk.CTkFrame(banner, fg_color="transparent")
        right_box.grid(row=0, column=2, rowspan=2, padx=(0, 18), pady=12, sticky="e")

        ctk.CTkButton(
            right_box,
            text="GitHub",
            width=92,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_DIVIDER,
            text_color=COLOR_TEXT_MUTED,
            command=lambda: webbrowser.open("https://github.com/professor-conifer/PsyMew"),
        ).grid(row=0, column=0, padx=4)

        ctk.CTkButton(
            right_box,
            text="Discord",
            width=92,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_SECONDARY,
            text_color=COLOR_SECONDARY,
            command=lambda: webbrowser.open("https://discord.gg/N34QduHdUP"),
        ).grid(row=0, column=1, padx=4)

        # Thin divider under banner
        ctk.CTkFrame(self, fg_color=COLOR_DIVIDER, height=1).grid(
            row=0, column=0, sticky="ews"
        )

    def _build_tabs(self) -> None:
        self.tabview = ctk.CTkTabview(
            self,
            fg_color=COLOR_BG,
            segmented_button_fg_color=COLOR_SURFACE_RAISED,
            segmented_button_selected_color=COLOR_PRIMARY,
            segmented_button_selected_hover_color=COLOR_PRIMARY_HOVER,
            segmented_button_unselected_hover_color=COLOR_DIVIDER,
            text_color=COLOR_TEXT,
        )
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 4))

        # Setup is the landing tab — first thing new users see, walks them
        # through the host-side prereqs pip can't install.
        for name in ("Setup", "Account", "Battle", "AI", "Advanced", "Run"):
            self.tabview.add(name)

        self.setup_tab = SetupTab(
            self.tabview.tab("Setup"),
            self.repo_root,
            switch_to_account=lambda: self.tabview.set("Account"),
        )
        self.setup_tab.pack(fill="both", expand=True)

        self.account_tab = AccountTab(self.tabview.tab("Account"), self.cfg_state)
        self.account_tab.pack(fill="both", expand=True)

        self.battle_tab = BattleTab(
            self.tabview.tab("Battle"), self.cfg_state, self.repo_root
        )
        self.battle_tab.pack(fill="both", expand=True)

        self.ai_tab = AITab(self.tabview.tab("AI"), self.cfg_state)
        self.ai_tab.pack(fill="both", expand=True)

        self.advanced_tab = AdvancedTab(self.tabview.tab("Advanced"), self.cfg_state)
        self.advanced_tab.pack(fill="both", expand=True)

        self.run_tab = RunTab(
            self.tabview.tab("Run"),
            self.cfg_state,
            self.repo_root,
            save_callback=self._save,
        )
        self.run_tab.pack(fill="both", expand=True)

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color=COLOR_SURFACE, corner_radius=0)
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)

        self.path_label = ctk.CTkLabel(
            footer,
            text=f"  .env: {self.cfg_state.env_path}",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.path_label.grid(row=0, column=0, sticky="w", padx=8, pady=6)

        self.dirty_label = ctk.CTkLabel(
            footer,
            text="",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            anchor="e",
        )
        self.dirty_label.grid(row=0, column=1, sticky="e", padx=12, pady=6)

        ctk.CTkLabel(
            footer,
            text=BRAND_VERSION,
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=10),
        ).grid(row=0, column=2, sticky="e", padx=(0, 14), pady=6)

    def _save(self) -> tuple[bool, str]:
        """Persist the config to .env. Returns (ok, message) for the Run tab toast."""
        try:
            self.cfg_state.save()
        except OSError as exc:
            return False, f"Failed to save .env: {exc}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Unexpected error saving .env: {exc.__class__.__name__}: {exc}"
        return True, f"Saved {self.cfg_state.env_path.name} ({len(self.cfg_state.values)} keys)"

    def _on_state_event(self, kind: str) -> None:
        self._refresh_dirty()

    def _refresh_dirty(self) -> None:
        if self.cfg_state.dirty:
            self.dirty_label.configure(text="● Unsaved changes", text_color=COLOR_PRIMARY)
        else:
            self.dirty_label.configure(text="● Saved", text_color=COLOR_TEXT_MUTED)

    def _on_close(self) -> None:
        # Give the bot a chance to die with us if it's running. The detached
        # loader is in a different process group and is left alone (project
        # invariant: never kill the loader).
        try:
            self.run_tab.on_app_close()
        except Exception:  # noqa: BLE001
            pass
        self.destroy()


def launch(repo_root: Path | None = None) -> None:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    ctk.set_appearance_mode(APPEARANCE_MODE)
    ctk.set_default_color_theme(COLOR_THEME)

    try:
        app = App(repo_root)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to launch PsyMew GUI: {exc}", file=sys.stderr)
        raise
    app.mainloop()
