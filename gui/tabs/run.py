"""Run tab — Start the bot, watch its log, track wins/losses.

The bot launches in its own console window (`start.py` is the launch path,
never modified). To stop the bot, the user closes that console window;
this tab has no Stop button by design.
"""

from __future__ import annotations

import queue
import re
from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk

from gui.process import BotProcess
from gui.state import ConfigState
from gui.theme import (
    COLOR_DIVIDER,
    COLOR_ERROR,
    COLOR_OK,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_STATUS_CRASHED,
    COLOR_STATUS_RUNNING,
    COLOR_STATUS_STARTING,
    COLOR_STATUS_STOPPED,
    COLOR_SURFACE_RAISED,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    FONT_MONO,
)
from gui.validation import errors_only, format_issues_summary, validate

STATUS_COLORS = {
    "stopped": COLOR_STATUS_STOPPED,
    "starting": COLOR_STATUS_STARTING,
    "running": COLOR_STATUS_RUNNING,
    "crashed": COLOR_STATUS_CRASHED,
}
STATUS_LABELS = {
    "stopped": "● Stopped",
    "starting": "● Starting…",
    "running": "● Running",
    "crashed": "● Crashed",
}

# showdown.py emits lines like (note: CustomFormatter prefixes "INFO     "):
#   "INFO     Won with team: random"     "INFO     Lost with team: random"
#   "INFO     W: 3\tL: 2"                "INFO     Winner: bot_name"
# We match anywhere in the line so the level/prefix doesn't matter.
_WIN_TEAM = re.compile(r"\bWon with team\b", re.IGNORECASE)
_LOSS_TEAM = re.compile(r"\bLost with team\b", re.IGNORECASE)
_TALLY = re.compile(r"\bW:\s*(\d+)\s+L:\s*(\d+)", re.IGNORECASE)
_WINNER_LINE = re.compile(r"\bWinner:\s+(\S+)", re.IGNORECASE)

_MAX_LOG_LINES = 2000
_POLL_INTERVAL_MS = 150


class RunTab(ctk.CTkFrame):
    def __init__(
        self,
        parent: ctk.CTkBaseClass,
        cfg_state: ConfigState,
        repo_root: Path,
        save_callback: Callable[[], tuple[bool, str]],
    ) -> None:
        super().__init__(parent, fg_color="transparent")
        self.cfg_state = cfg_state
        self.repo_root = repo_root
        self.save_callback = save_callback

        self._log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._wins = 0
        self._losses = 0
        self._bot_username = ""
        self._current_state = "stopped"
        self._save_toast_after_id: str | None = None

        self.process = BotProcess(
            repo_root=repo_root,
            on_line=lambda line: self._log_queue.put(("line", line)),
            on_state=lambda state: self._log_queue.put(("state", state)),
        )

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._build()
        self.cfg_state.subscribe(self._on_state_event)

        self._poll_queue()
        self._refresh_validation()

    # ---- layout --------------------------------------------------------

    def _build(self) -> None:
        # Big status hero
        hero = ctk.CTkFrame(self, fg_color=COLOR_SURFACE_RAISED, corner_radius=12)
        hero.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        hero.columnconfigure(0, weight=1)
        hero.columnconfigure(1, weight=0)

        self.status_label = ctk.CTkLabel(
            hero,
            text=STATUS_LABELS["stopped"],
            text_color=STATUS_COLORS["stopped"],
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=18, pady=(14, 0))

        self.pid_label = ctk.CTkLabel(
            hero, text="", text_color=COLOR_TEXT_MUTED, font=ctk.CTkFont(size=11)
        )
        self.pid_label.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 12))

        # W/L card on the right
        wl_frame = ctk.CTkFrame(hero, fg_color="transparent")
        wl_frame.grid(row=0, column=1, rowspan=2, sticky="e", padx=18, pady=12)

        self.wins_label = ctk.CTkLabel(
            wl_frame,
            text="0",
            text_color=COLOR_STATUS_RUNNING,
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.wins_label.grid(row=0, column=0, padx=(0, 8))
        ctk.CTkLabel(wl_frame, text="W", text_color=COLOR_TEXT_MUTED).grid(
            row=1, column=0, padx=(0, 8)
        )

        sep = ctk.CTkLabel(wl_frame, text="•", text_color=COLOR_DIVIDER)
        sep.grid(row=0, column=1, padx=(0, 8))

        self.losses_label = ctk.CTkLabel(
            wl_frame,
            text="0",
            text_color=COLOR_STATUS_CRASHED,
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.losses_label.grid(row=0, column=2)
        ctk.CTkLabel(wl_frame, text="L", text_color=COLOR_TEXT_MUTED).grid(
            row=1, column=2
        )

        # Action bar
        action_bar = ctk.CTkFrame(self, fg_color="transparent")
        action_bar.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 6))
        action_bar.columnconfigure(0, weight=1)

        self.validation_label = ctk.CTkLabel(
            action_bar,
            text="",
            text_color=COLOR_TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=540,
        )
        self.validation_label.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.save_btn = ctk.CTkButton(
            action_bar,
            text="Save .env",
            width=120,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_DIVIDER,
            text_color=COLOR_TEXT,
            command=self._on_save,
        )
        self.save_btn.grid(row=0, column=1, padx=4)

        # Save-result toast lives directly below the action bar and
        # auto-clears after a few seconds on success (stays visible on
        # failure so the user can read the error).
        self.save_toast = ctk.CTkLabel(
            self,
            text="",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
            justify="left",
            wraplength=900,
        )
        self.save_toast.grid(row=5, column=0, sticky="ew", padx=20, pady=(2, 0))

        self.start_btn = ctk.CTkButton(
            action_bar,
            text="▶  Start bot",
            width=170,
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            text_color=COLOR_TEXT,
            command=self._on_start,
        )
        self.start_btn.grid(row=0, column=2, padx=(8, 0))

        # Hint about how to stop
        ctk.CTkLabel(
            self,
            text=(
                "Tip: the bot runs in its own console window. To stop it, "
                "close that window — the GUI will pick up the new state."
            ),
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            anchor="w",
            wraplength=900,
            justify="left",
        ).grid(row=3, column=0, sticky="ew", padx=20, pady=(4, 6))

        # Log preview
        log_header = ctk.CTkFrame(self, fg_color="transparent")
        log_header.grid(row=4, column=0, sticky="ew", padx=20, pady=(8, 0))
        log_header.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            log_header,
            text="Live log (tailing logs/init.log)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            log_header,
            text="Clear",
            width=70,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_DIVIDER,
            text_color=COLOR_TEXT_MUTED,
            command=self._clear_log,
        ).grid(row=0, column=1, sticky="e")

        self.log_text = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(family=FONT_MONO, size=11),
            wrap="none",
            activate_scrollbars=True,
            fg_color="#0c0c14",
            text_color=COLOR_TEXT,
            corner_radius=8,
        )
        self.log_text.grid(row=2, column=0, sticky="nsew", padx=20, pady=(4, 10))
        self.log_text.configure(state="disabled")

    # ---- queue plumbing ------------------------------------------------

    def _poll_queue(self) -> None:
        drained: list[tuple[str, str]] = []
        try:
            while True:
                drained.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass

        if drained:
            new_lines = [item[1] for item in drained if item[0] == "line"]
            for line in new_lines:
                self._scan_for_results(line)
            if new_lines:
                self._append_lines(new_lines)
            for kind, payload in drained:
                if kind == "state":
                    self._apply_state(payload)

        self.after(_POLL_INTERVAL_MS, self._poll_queue)

    def _append_lines(self, lines: list[str]) -> None:
        if not lines:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", "\n".join(lines) + "\n")
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

    def _scan_for_results(self, line: str) -> None:
        # Source of truth — showdown.py logs these explicitly.
        if _WIN_TEAM.search(line):
            self._wins += 1
            self._update_battles_label()
            return
        if _LOSS_TEAM.search(line):
            self._losses += 1
            self._update_battles_label()
            return
        # Belt-and-suspenders: showdown.py also prints a running tally
        # right after each battle. Trust this over our own counter if it
        # appears — that way reconnects / partial logs don't drift.
        m = _TALLY.search(line)
        if m:
            try:
                self._wins = int(m.group(1))
                self._losses = int(m.group(2))
                self._update_battles_label()
            except ValueError:
                pass
            return
        # Final-fallback path: fp/run_battle.py logs "Winner: <username>"
        # at battle end. If we know the bot's username we can score from it.
        m = _WINNER_LINE.match(line)
        if m and self._bot_username:
            winner = m.group(1).strip().lower()
            if winner == self._bot_username.lower():
                self._wins += 1
            else:
                self._losses += 1
            self._update_battles_label()

    def _update_battles_label(self) -> None:
        self.wins_label.configure(text=str(self._wins))
        self.losses_label.configure(text=str(self._losses))

    def _apply_state(self, state: str) -> None:
        self._current_state = state
        self.status_label.configure(
            text=STATUS_LABELS.get(state, "● Unknown"),
            text_color=STATUS_COLORS.get(state, COLOR_TEXT_MUTED),
        )
        pid = self.process.pid()
        if state in ("starting", "running") and pid:
            self.pid_label.configure(
                text=f"Bot PID: {pid} — running in its own console window"
            )
        elif state == "crashed":
            self.pid_label.configure(
                text="The bot's console exited with an error. Check the log."
            )
        else:
            self.pid_label.configure(text="")

        running = state in ("starting", "running")
        self.start_btn.configure(state="disabled" if running else "normal")

    # ---- button handlers ----------------------------------------------

    def _on_save(self) -> None:
        ok, msg = self.save_callback()
        self._show_save_toast(ok, msg)

    def _show_save_toast(self, ok: bool, message: str) -> None:
        """Display a transient success / persistent failure indicator near Save."""
        if self._save_toast_after_id is not None:
            try:
                self.after_cancel(self._save_toast_after_id)
            except Exception:  # noqa: BLE001
                pass
            self._save_toast_after_id = None

        if ok:
            self.save_toast.configure(
                text=f"✓ {message}",
                text_color=COLOR_OK,
            )
            # Clear after 3 seconds — success is ephemeral.
            self._save_toast_after_id = self.after(3000, self._clear_save_toast)
        else:
            self.save_toast.configure(
                text=f"✗ {message}",
                text_color=COLOR_ERROR,
            )
            # Failure stays visible until the next save attempt.

    def _clear_save_toast(self) -> None:
        self.save_toast.configure(text="", text_color=COLOR_TEXT_MUTED)
        self._save_toast_after_id = None

    def _on_start(self) -> None:
        issues = errors_only(validate(self.cfg_state.values))
        if issues:
            self._append_lines(
                ["", "[GUI] Cannot start — fix these first:"]
                + [f"  ✗ {i.message}" for i in issues]
            )
            return

        self._wins = 0
        self._losses = 0
        self._update_battles_label()
        self._bot_username = self.cfg_state.get("PS_USERNAME").strip()

        # Snapshot the config to .env so start.py reads our latest values.
        # If the save fails, surface the error to the toast and bail —
        # launching the bot with stale config would be worse than nothing.
        ok, msg = self.save_callback()
        self._show_save_toast(ok, msg)
        if not ok:
            return

        self._append_lines(
            ["", "[GUI] Launching python -u start.py in a new console window…"]
        )
        self.process.start()

    # ---- state subscription -------------------------------------------

    def _refresh_validation(self) -> None:
        issues = validate(self.cfg_state.values)
        if not issues:
            self.validation_label.configure(
                text="✓ Config looks good — click Start when ready.",
                text_color=COLOR_TEXT_MUTED,
            )
        else:
            self.validation_label.configure(text=format_issues_summary(issues))

    def _on_state_event(self, kind: str) -> None:
        self._refresh_validation()

    # ---- lifecycle ----------------------------------------------------

    def on_app_close(self) -> None:
        """Called by App on WM_DELETE_WINDOW so the bot doesn't outlive the GUI."""
        if self.process.is_running():
            self.process.stop()
