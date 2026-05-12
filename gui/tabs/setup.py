"""Setup tab — first-run prerequisite checks and installers.

This tab is the GUI's landing page. It scans the host for the bits PsyMew
needs that `pip install -r requirements.txt` *can't* install for you
(Rust, MSVC build tools), reports their status, and offers one-click
install actions where possible.

Each install runs in a worker thread inside `prereqs.TaskRunner`,
streaming subprocess output line-by-line into the in-tab log textbox
so users see what's happening for the long-running ones (winget
installs, especially VS Build Tools, can take 15+ minutes).
"""

from __future__ import annotations

import queue
import webbrowser
from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk

from gui import prereqs
from gui.theme import (
    COLOR_DIVIDER,
    COLOR_ERROR,
    COLOR_OK,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_SECONDARY,
    COLOR_SURFACE_RAISED,
    COLOR_TEXT,
    COLOR_TEXT_DIM,
    COLOR_TEXT_MUTED,
    COLOR_WARN,
    FONT_MONO,
)

# Map prereqs.PrereqStatus.severity -> (glyph, color)
_SEVERITY_STYLE = {
    "ok": ("✓", COLOR_OK),
    "warn": ("●", COLOR_WARN),
    "missing": ("✗", COLOR_ERROR),
    "unknown": ("?", COLOR_TEXT_MUTED),
}

_POLL_INTERVAL_MS = 150
_MAX_LOG_LINES = 2000


class SetupTab(ctk.CTkScrollableFrame):
    def __init__(
        self,
        parent: ctk.CTkBaseClass,
        repo_root: Path,
        switch_to_account: Callable[[], None],
    ) -> None:
        super().__init__(parent, fg_color="transparent")
        self.repo_root = repo_root
        self.switch_to_account = switch_to_account

        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self.runner = prereqs.TaskRunner(
            on_line=lambda line: self._log_queue.put(line),
            on_idle=lambda: self._log_queue.put("__idle__"),
        )

        # Widgets that should disable while a task is running, so the user
        # doesn't fire off two installs at once.
        self._action_buttons: list[ctk.CTkButton] = []

        # status_rows[key] = (status_label, detail_label, blurb_label)
        self._status_rows: dict[str, tuple] = {}

        self.columnconfigure(0, weight=1)
        self._build()
        self._poll_queue()
        self.refresh()

    # ---- layout --------------------------------------------------------

    def _build(self) -> None:
        # Hero
        ctk.CTkLabel(
            self,
            text="First-time setup",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 2))

        ctk.CTkLabel(
            self,
            text=(
                "PsyMew leans on a few tools pip can't install for you — "
                "Rust for the MCTS engine, MSVC build tools on Windows. "
                "This page detects what you have and fills in the rest."
            ),
            text_color=COLOR_TEXT_MUTED,
            wraplength=860,
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 12))

        self._build_toolchain_card(row=2)
        self._build_packages_card(row=3)
        self._build_engine_card(row=4)
        self._build_log_card(row=5)
        self._build_footer(row=6)

    def _make_card(self, title: str, subtitle: str, row: int) -> ctk.CTkFrame:
        card = ctk.CTkFrame(self, fg_color=COLOR_SURFACE_RAISED, corner_radius=10)
        card.grid(row=row, column=0, sticky="ew", padx=20, pady=8)
        card.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))
        if subtitle:
            ctk.CTkLabel(
                card,
                text=subtitle,
                text_color=COLOR_TEXT_MUTED,
                font=ctk.CTkFont(size=11),
                wraplength=820,
                justify="left",
                anchor="w",
            ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
        return card

    def _make_row(
        self,
        parent: ctk.CTkFrame,
        key: str,
        row: int,
        actions: list[tuple[str, Callable[[], None], str]] | None = None,
    ) -> None:
        """Render one prereq row and stash refs so refresh() can update it."""
        outer = ctk.CTkFrame(parent, fg_color="transparent")
        outer.grid(row=row, column=0, sticky="ew", padx=14, pady=4)
        outer.columnconfigure(2, weight=1)

        status = ctk.CTkLabel(
            outer,
            text="…",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=18, weight="bold"),
            width=24,
        )
        status.grid(row=0, column=0, sticky="w")

        name = ctk.CTkLabel(
            outer,
            text=key,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        )
        name.grid(row=0, column=1, sticky="w", padx=(8, 16))

        detail = ctk.CTkLabel(
            outer,
            text="probing…",
            text_color=COLOR_TEXT_MUTED,
            anchor="w",
            font=ctk.CTkFont(size=12),
        )
        detail.grid(row=0, column=2, sticky="ew")

        if actions:
            buttons = ctk.CTkFrame(outer, fg_color="transparent")
            buttons.grid(row=0, column=3, sticky="e")
            for i, (label, cb, style) in enumerate(actions):
                if style == "primary":
                    fg, hover = COLOR_PRIMARY, COLOR_PRIMARY_HOVER
                    text_color = COLOR_TEXT
                    border = 0
                else:
                    fg, hover = "transparent", COLOR_DIVIDER
                    text_color = COLOR_SECONDARY
                    border = 1
                btn = ctk.CTkButton(
                    buttons,
                    text=label,
                    width=130,
                    height=28,
                    fg_color=fg,
                    hover_color=hover,
                    text_color=text_color,
                    border_width=border,
                    border_color=COLOR_SECONDARY,
                    font=ctk.CTkFont(size=11),
                    command=cb,
                )
                btn.grid(row=0, column=i, padx=(6, 0))
                self._action_buttons.append(btn)

        blurb = ctk.CTkLabel(
            outer,
            text="",
            text_color=COLOR_TEXT_DIM,
            font=ctk.CTkFont(size=10),
            anchor="w",
            justify="left",
            wraplength=720,
        )
        blurb.grid(row=1, column=1, columnspan=3, sticky="w", padx=(8, 0), pady=(0, 2))

        self._status_rows[key] = (status, name, detail, blurb)

    def _build_toolchain_card(self, row: int) -> None:
        card = self._make_card(
            "Core toolchain",
            "Programs PsyMew expects on your machine, outside of pip.",
            row,
        )

        self._make_row(
            card, "python", row=2,
            actions=[("python.org ↗", lambda: webbrowser.open("https://www.python.org/downloads/"), "secondary")],
        )
        self._make_row(card, "pip", row=3)
        self._make_row(
            card, "rust", row=4,
            actions=[
                ("Install Rust", self._install_rust, "primary"),
                ("rustup.rs ↗", lambda: webbrowser.open("https://rustup.rs/"), "secondary"),
            ],
        )
        self._make_row(
            card, "cpp", row=5,
            actions=[
                ("Install Build Tools", self._install_cpp, "primary"),
                ("Download ↗", lambda: webbrowser.open(
                    "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
                ), "secondary"),
            ],
        )
        # Pad the bottom
        ctk.CTkLabel(card, text="").grid(row=6, column=0, pady=(0, 6))

    def _build_packages_card(self, row: int) -> None:
        card = self._make_card(
            "Python packages",
            "Everything listed in requirements.txt. Missing ones can be "
            "installed in one click.",
            row,
        )

        self.packages_summary = ctk.CTkLabel(
            card,
            text="probing…",
            text_color=COLOR_TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=820,
        )
        self.packages_summary.grid(row=2, column=0, sticky="ew", padx=14, pady=4)

        self.packages_detail = ctk.CTkLabel(
            card,
            text="",
            text_color=COLOR_TEXT_DIM,
            anchor="w",
            justify="left",
            wraplength=820,
            font=ctk.CTkFont(size=11),
        )
        self.packages_detail.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 4))

        action_row = ctk.CTkFrame(card, fg_color="transparent")
        action_row.grid(row=4, column=0, sticky="w", padx=14, pady=(4, 12))

        btn = ctk.CTkButton(
            action_row,
            text="Install missing packages",
            width=200,
            height=32,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._install_packages,
        )
        btn.grid(row=0, column=0, padx=(0, 6))
        self._action_buttons.append(btn)

        btn2 = ctk.CTkButton(
            action_row,
            text="Open requirements.txt",
            width=170,
            height=32,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_DIVIDER,
            text_color=COLOR_TEXT_MUTED,
            command=self._open_requirements,
        )
        btn2.grid(row=0, column=1, padx=4)

    def _build_engine_card(self, row: int) -> None:
        card = self._make_card(
            "Battle engine",
            "poke-engine compiles from source on first install. Rust + "
            "the C++ tools above must be working for this to succeed.",
            row,
        )
        self._make_row(
            card, "poke_engine", row=2,
            actions=[("Rebuild engine", self._rebuild_engine, "secondary")],
        )
        ctk.CTkLabel(card, text="").grid(row=3, column=0, pady=(0, 6))

    def _build_log_card(self, row: int) -> None:
        card = ctk.CTkFrame(self, fg_color=COLOR_SURFACE_RAISED, corner_radius=10)
        card.grid(row=row, column=0, sticky="ew", padx=20, pady=8)
        card.columnconfigure(0, weight=1)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))
        header.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Install output",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.log_status = ctk.CTkLabel(
            header,
            text="idle",
            text_color=COLOR_TEXT_DIM,
            font=ctk.CTkFont(size=11),
            anchor="e",
        )
        self.log_status.grid(row=0, column=1, sticky="e")

        ctk.CTkButton(
            header,
            text="Clear",
            width=70,
            height=24,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_DIVIDER,
            text_color=COLOR_TEXT_MUTED,
            command=self._clear_log,
        ).grid(row=0, column=2, sticky="e", padx=(6, 0))

        self.log_text = ctk.CTkTextbox(
            card,
            font=ctk.CTkFont(family=FONT_MONO, size=11),
            wrap="word",
            activate_scrollbars=True,
            fg_color="#0c0c14",
            text_color=COLOR_TEXT,
            corner_radius=6,
            height=180,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 12))
        self.log_text.configure(state="disabled")

    def _build_footer(self, row: int) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=row, column=0, sticky="ew", padx=20, pady=(4, 20))
        footer.columnconfigure(1, weight=1)

        ctk.CTkButton(
            footer,
            text="Re-check everything",
            width=170,
            height=34,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_SECONDARY,
            text_color=COLOR_SECONDARY,
            command=self.refresh,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            footer,
            text="Continue to bot setup  →",
            width=210,
            height=38,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.switch_to_account,
        ).grid(row=0, column=2, sticky="e")

    # ---- detection + UI sync ------------------------------------------

    def refresh(self) -> None:
        """Re-run every detection and repaint the rows."""
        statuses: dict[str, prereqs.PrereqStatus] = {
            s.key: s
            for s in (
                prereqs.detect_python(),
                prereqs.detect_pip(),
                prereqs.detect_rust(),
                prereqs.detect_cpp_build_tools(),
                prereqs.detect_poke_engine(),
            )
        }

        labels = {
            "python": "Python interpreter",
            "pip": "pip (package manager)",
            "rust": "Rust toolchain",
            "cpp": "C/C++ build tools",
            "poke_engine": "poke-engine (battle engine)",
        }

        for key, friendly in labels.items():
            s = statuses.get(key)
            if s is None:
                continue
            self._set_row(key, friendly, s)

        # Packages summary
        pkg_statuses = prereqs.detect_python_packages()
        missing = [p for p in pkg_statuses if p.severity != "ok"]
        installed = [p for p in pkg_statuses if p.severity == "ok"]
        total = len(pkg_statuses)
        present = len(installed)
        if missing:
            self.packages_summary.configure(
                text=f"✗ {len(missing)} / {total} packages missing — click below to install them.",
                text_color=COLOR_ERROR,
            )
            missing_names = ", ".join(p.label for p in missing)
            self.packages_detail.configure(
                text=f"Missing: {missing_names}"
            )
        else:
            self.packages_summary.configure(
                text=f"✓ All {present} required packages installed.",
                text_color=COLOR_OK,
            )
            self.packages_detail.configure(text="")

    def _set_row(self, key: str, friendly: str, status: prereqs.PrereqStatus) -> None:
        widgets = self._status_rows.get(key)
        if widgets is None:
            return
        glyph, color = _SEVERITY_STYLE.get(status.severity, ("?", COLOR_TEXT_MUTED))
        status_label, name_label, detail_label, blurb_label = widgets
        status_label.configure(text=glyph, text_color=color)
        name_label.configure(text=friendly)
        detail_label.configure(
            text=status.detail,
            text_color=color if status.severity != "ok" else COLOR_TEXT,
        )
        blurb_label.configure(
            text=status.blurb if status.blurb else "",
        )

    # ---- install actions -----------------------------------------------

    def _install_rust(self) -> None:
        if not prereqs.winget_available():
            self._append_log("[GUI] winget not found — opening rustup.rs instead.")
            webbrowser.open("https://rustup.rs/")
            return
        self._submit(
            prereqs.InstallTask(
                title="Installing Rust (winget)",
                runner=prereqs.install_rust_via_winget,
            )
        )

    def _install_cpp(self) -> None:
        if not prereqs.winget_available():
            self._append_log(
                "[GUI] winget not found — opening the Build Tools download page."
            )
            webbrowser.open(
                "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            )
            return
        self._append_log(
            "[GUI] Installing VS Build Tools 2022 + C++ workload via winget."
        )
        self._append_log(
            "[GUI] Heads up: this can take 10–30 minutes and may pop a UAC prompt."
        )
        self._submit(
            prereqs.InstallTask(
                title="Installing VS Build Tools",
                runner=prereqs.install_vs_build_tools_via_winget,
            )
        )

    def _install_packages(self) -> None:
        repo_root = self.repo_root
        self._submit(
            prereqs.InstallTask(
                title="Installing required Python packages",
                runner=lambda on_line: prereqs.install_missing_packages(on_line, repo_root),
            )
        )

    def _rebuild_engine(self) -> None:
        self._submit(
            prereqs.InstallTask(
                title="Rebuilding poke-engine from source",
                runner=prereqs.rebuild_poke_engine,
            )
        )

    def _open_requirements(self) -> None:
        req = self.repo_root / "requirements.txt"
        if req.is_file():
            webbrowser.open(req.as_uri())

    # ---- task plumbing -------------------------------------------------

    def _submit(self, task: prereqs.InstallTask) -> None:
        if self.runner.busy:
            self._append_log(
                "[GUI] Another task is already running — wait for it to finish."
            )
            return
        for btn in self._action_buttons:
            btn.configure(state="disabled")
        self.log_status.configure(
            text="running…",
            text_color=COLOR_SECONDARY,
        )
        if not self.runner.submit(task):
            for btn in self._action_buttons:
                btn.configure(state="normal")
            self.log_status.configure(text="idle", text_color=COLOR_TEXT_DIM)

    def _poll_queue(self) -> None:
        drained: list[str] = []
        try:
            while True:
                drained.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass

        idle_signal = False
        for item in drained:
            if item == "__idle__":
                idle_signal = True
            else:
                self._append_log(item)

        if idle_signal:
            for btn in self._action_buttons:
                btn.configure(state="normal")
            self.log_status.configure(text="idle", text_color=COLOR_TEXT_DIM)
            # Refresh detection so the row turns green if the install worked.
            self.refresh()

        self.after(_POLL_INTERVAL_MS, self._poll_queue)

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        # Trim to keep memory bounded.
        all_text = self.log_text.get("1.0", "end").splitlines()
        if len(all_text) > _MAX_LOG_LINES:
            keep = all_text[-_MAX_LOG_LINES:]
            self.log_text.delete("1.0", "end")
            self.log_text.insert("end", "\n".join(keep) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
