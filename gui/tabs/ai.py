"""AI tab — pick an LLM/MCTS engine and configure its API key + model."""

from __future__ import annotations

import threading
import webbrowser
from typing import Callable

import customtkinter as ctk

from gui import api_test
from gui.state import ConfigState
from gui.theme import (
    COLOR_DIVIDER,
    COLOR_ERROR,
    COLOR_OK,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_SECONDARY,
    COLOR_SURFACE_RAISED,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
)

ENGINES = ("mcts", "claude", "gemini", "deepseek")
ENGINE_LABELS = {
    "mcts": "MCTS (Rust search)",
    "claude": "Claude (Anthropic)",
    "gemini": "Gemini (Google)",
    "deepseek": "DeepSeek V4",
}
ENGINE_TAGLINES = {
    "mcts": "Pure tree-search bot. No API key needed.",
    "claude": "Anthropic Claude with extended thinking + tool use.",
    "gemini": "Google Gemini Pro with function calling.",
    "deepseek": "DeepSeek V4 reasoning model (OpenAI-compatible API).",
}
ENGINE_CONSOLE_URLS = {
    "claude": "https://console.anthropic.com/settings/keys",
    "gemini": "https://aistudio.google.com/apikey",
    "deepseek": "https://platform.deepseek.com/api_keys",
}

CLAUDE_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]
GEMINI_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking-exp",
]
DEEPSEEK_MODELS = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-reasoner",
]
DEEPSEEK_REASONING_EFFORT = ["low", "medium", "high", "max", "xhigh"]


class AITab(ctk.CTkScrollableFrame):
    def __init__(self, parent: ctk.CTkBaseClass, cfg_state: ConfigState) -> None:
        super().__init__(parent, fg_color="transparent")
        self.cfg_state = cfg_state

        self.columnconfigure(0, weight=1)
        self._panels: dict[str, ctk.CTkFrame] = {}

        self._build()
        self.refresh()
        self.cfg_state.subscribe(self._on_state_event)

    def _build(self) -> None:
        ctk.CTkLabel(
            self,
            text="Decision engine",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 2))

        ctk.CTkLabel(
            self,
            text="What chooses the bot's moves. LLM engines need an API key.",
            text_color=COLOR_TEXT_MUTED,
        ).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 12))

        # Engine picker
        self.engine_var = ctk.StringVar(value="claude")
        picker = ctk.CTkFrame(self, fg_color=COLOR_SURFACE_RAISED, corner_radius=10)
        picker.grid(row=2, column=0, sticky="ew", padx=20, pady=8)
        picker.columnconfigure((0, 1, 2, 3), weight=1, uniform="engines")

        for i, engine in enumerate(ENGINES):
            tile = ctk.CTkFrame(picker, fg_color="transparent")
            tile.grid(row=0, column=i, sticky="nsew", padx=8, pady=10)
            ctk.CTkRadioButton(
                tile,
                text=ENGINE_LABELS[engine],
                variable=self.engine_var,
                value=engine,
                fg_color=COLOR_PRIMARY,
                hover_color=COLOR_PRIMARY_HOVER,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=self._on_engine_change,
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                tile,
                text=ENGINE_TAGLINES[engine],
                text_color=COLOR_TEXT_MUTED,
                font=ctk.CTkFont(size=10),
                wraplength=180,
                justify="left",
                anchor="w",
            ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Detail panel host
        self.panel_host = ctk.CTkFrame(
            self, fg_color=COLOR_SURFACE_RAISED, corner_radius=10
        )
        self.panel_host.grid(row=3, column=0, sticky="ew", padx=20, pady=(8, 8))
        self.panel_host.columnconfigure(0, weight=1)

        self._build_mcts_panel()
        self._build_llm_panel("claude", "ANTHROPIC_API_KEY", CLAUDE_MODELS, has_reasoning=False)
        self._build_llm_panel("gemini", "GEMINI_API_KEY", GEMINI_MODELS, has_reasoning=False)
        self._build_llm_panel(
            "deepseek", "DEEPSEEK_API_KEY", DEEPSEEK_MODELS, has_reasoning=True
        )

        # Tutor mode
        self.tutor_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(
            self,
            text="Tutor mode — bot chats post-turn coaching and battle commentary",
            variable=self.tutor_var,
            progress_color=COLOR_PRIMARY,
            button_color=COLOR_TEXT,
            button_hover_color=COLOR_PRIMARY_HOVER,
            command=lambda: self.cfg_state.set(
                "TUTOR_MODE", "1" if self.tutor_var.get() else ""
            ),
        ).grid(row=4, column=0, sticky="w", padx=20, pady=(8, 16))

    def _build_mcts_panel(self) -> None:
        panel = ctk.CTkFrame(self.panel_host, fg_color="transparent")
        self._panels["mcts"] = panel
        panel.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            panel,
            text="Pure Rust Monte-Carlo Tree Search — no API key required.",
            text_color=COLOR_TEXT_MUTED,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 8))

        ctk.CTkLabel(panel, text="Search time (ms)").grid(
            row=1, column=0, sticky="w", padx=14, pady=4
        )
        self.search_time_var = ctk.StringVar(value="300")
        ctk.CTkEntry(panel, textvariable=self.search_time_var, width=120).grid(
            row=1, column=1, sticky="w", padx=14, pady=4
        )
        self.search_time_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set("PS_SEARCH_TIME_MS", self.search_time_var.get()),
        )

        ctk.CTkLabel(panel, text="Parallelism").grid(
            row=2, column=0, sticky="w", padx=14, pady=(4, 12)
        )
        self.parallelism_var = ctk.StringVar(value="1")
        ctk.CTkEntry(panel, textvariable=self.parallelism_var, width=120).grid(
            row=2, column=1, sticky="w", padx=14, pady=(4, 12)
        )
        self.parallelism_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set(
                "PS_SEARCH_PARALLELISM", self.parallelism_var.get()
            ),
        )

    def _build_llm_panel(
        self,
        engine: str,
        api_key_var_name: str,
        model_choices: list[str],
        has_reasoning: bool,
    ) -> None:
        panel = ctk.CTkFrame(self.panel_host, fg_color="transparent")
        self._panels[engine] = panel
        panel.columnconfigure(1, weight=1)
        row = 0

        # Header + "Get a key" link
        head = ctk.CTkFrame(panel, fg_color="transparent")
        head.grid(row=row, column=0, columnspan=4, sticky="ew", padx=14, pady=(12, 4))
        head.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            head,
            text=ENGINE_TAGLINES[engine],
            text_color=COLOR_TEXT_MUTED,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        url = ENGINE_CONSOLE_URLS.get(engine)
        if url:
            ctk.CTkButton(
                head,
                text="Get a key ↗",
                width=110,
                fg_color="transparent",
                hover_color=COLOR_DIVIDER,
                border_width=1,
                border_color=COLOR_SECONDARY,
                text_color=COLOR_SECONDARY,
                command=lambda u=url: webbrowser.open(u),
            ).grid(row=0, column=1, sticky="e")
        row += 1

        # API key row
        ctk.CTkLabel(panel, text="API key *").grid(
            row=row, column=0, sticky="w", padx=14, pady=4
        )
        key_var = ctk.StringVar()
        key_entry = ctk.CTkEntry(panel, textvariable=key_var, show="•")
        key_entry.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        key_var.trace_add(
            "write", lambda *_: self.cfg_state.set(api_key_var_name, key_var.get())
        )
        show_btn = ctk.CTkButton(
            panel,
            text="Show",
            width=64,
            fg_color="transparent",
            hover_color=COLOR_DIVIDER,
            border_width=1,
            border_color=COLOR_DIVIDER,
            text_color=COLOR_TEXT_MUTED,
            command=lambda: self._toggle_show(key_entry, show_btn),
        )
        show_btn.grid(row=row, column=2, padx=(0, 6), pady=4)
        test_btn = ctk.CTkButton(
            panel,
            text="Test connection",
            width=140,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
        )
        test_btn.grid(row=row, column=3, padx=(0, 14), pady=4)
        row += 1

        status_label = ctk.CTkLabel(
            panel, text="", text_color=COLOR_TEXT_MUTED, anchor="w"
        )
        status_label.grid(row=row, column=1, columnspan=3, sticky="w", padx=8, pady=(0, 6))
        test_btn.configure(
            command=lambda: self._run_test(engine, key_var.get(), test_btn, status_label)
        )
        row += 1

        # Model
        ctk.CTkLabel(panel, text="Model").grid(
            row=row, column=0, sticky="w", padx=14, pady=4
        )
        model_var = ctk.StringVar(value=model_choices[0])
        ctk.CTkComboBox(panel, values=model_choices, variable=model_var).grid(
            row=row, column=1, columnspan=3, sticky="ew", padx=8, pady=4
        )
        model_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set(f"{engine.upper()}_MODEL", model_var.get()),
        )
        row += 1

        # Tutor model
        ctk.CTkLabel(panel, text="Tutor model").grid(
            row=row, column=0, sticky="w", padx=14, pady=4
        )
        tutor_var = ctk.StringVar(value=model_choices[-1])
        ctk.CTkComboBox(panel, values=model_choices, variable=tutor_var).grid(
            row=row, column=1, columnspan=3, sticky="ew", padx=8, pady=4
        )
        tutor_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set(f"{engine.upper()}_TUTOR_MODEL", tutor_var.get()),
        )
        row += 1

        # Thinking budget
        ctk.CTkLabel(panel, text="Thinking budget").grid(
            row=row, column=0, sticky="w", padx=14, pady=4
        )
        budget_var = ctk.StringVar(value="4096")
        ctk.CTkEntry(panel, textvariable=budget_var, width=120).grid(
            row=row, column=1, sticky="w", padx=8, pady=4
        )
        ctk.CTkLabel(
            panel,
            text="tokens reserved for extended thinking (0 = disabled)",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=11),
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=0, pady=4)
        budget_var.trace_add(
            "write",
            lambda *_: self.cfg_state.set(
                f"{engine.upper()}_THINKING_BUDGET", budget_var.get()
            ),
        )
        row += 1

        if has_reasoning:
            ctk.CTkLabel(panel, text="Reasoning effort").grid(
                row=row, column=0, sticky="w", padx=14, pady=(4, 12)
            )
            effort_var = ctk.StringVar(value="high")
            ctk.CTkOptionMenu(
                panel,
                values=DEEPSEEK_REASONING_EFFORT,
                variable=effort_var,
                width=140,
                fg_color=COLOR_SECONDARY,
                button_color=COLOR_PRIMARY,
                button_hover_color=COLOR_PRIMARY_HOVER,
                command=lambda v: self.cfg_state.set("DEEPSEEK_REASONING_EFFORT", v),
            ).grid(row=row, column=1, sticky="w", padx=8, pady=(4, 12))
            setattr(self, f"_{engine}_effort_var", effort_var)
            row += 1

        # Stash vars for refresh()
        setattr(self, f"_{engine}_key_var", key_var)
        setattr(self, f"_{engine}_model_var", model_var)
        setattr(self, f"_{engine}_tutor_var", tutor_var)
        setattr(self, f"_{engine}_budget_var", budget_var)
        setattr(self, f"_{engine}_api_key_var_name", api_key_var_name)

    def _toggle_show(self, entry: ctk.CTkEntry, btn: ctk.CTkButton) -> None:
        current = entry.cget("show")
        if current == "":
            entry.configure(show="•")
            btn.configure(text="Show")
        else:
            entry.configure(show="")
            btn.configure(text="Hide")

    def _run_test(
        self,
        engine: str,
        api_key: str,
        btn: ctk.CTkButton,
        status_label: ctk.CTkLabel,
    ) -> None:
        btn.configure(state="disabled", text="Testing…")
        status_label.configure(text="Calling provider…", text_color=COLOR_TEXT_MUTED)

        def worker() -> None:
            test_fn: Callable[[str], tuple[bool, str]] = {
                "claude": api_test.test_claude,
                "gemini": api_test.test_gemini,
                "deepseek": api_test.test_deepseek,
            }[engine]
            ok, msg = test_fn(api_key)
            self.after(0, lambda: self._finish_test(btn, status_label, ok, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_test(
        self,
        btn: ctk.CTkButton,
        status_label: ctk.CTkLabel,
        ok: bool,
        msg: str,
    ) -> None:
        btn.configure(state="normal", text="Test connection")
        prefix = "✓" if ok else "✗"
        status_label.configure(
            text=f"{prefix} {msg}",
            text_color=COLOR_OK if ok else COLOR_ERROR,
        )

    def _on_engine_change(self) -> None:
        engine = self.engine_var.get()
        self.cfg_state.set("DECISION_ENGINE", engine)
        self._show_panel(engine)

    def _show_panel(self, engine: str) -> None:
        for panel in self._panels.values():
            panel.grid_remove()
        panel = self._panels.get(engine)
        if panel is not None:
            panel.grid(row=0, column=0, sticky="ew")
            self.panel_host.columnconfigure(0, weight=1)

    def refresh(self) -> None:
        engine = (self.cfg_state.get("DECISION_ENGINE") or "claude").lower()
        if engine not in ENGINES:
            engine = "claude"
        self.engine_var.set(engine)
        self._show_panel(engine)
        self.tutor_var.set(self.cfg_state.get("TUTOR_MODE") == "1")

        for e in ("claude", "gemini", "deepseek"):
            key_name = getattr(self, f"_{e}_api_key_var_name")
            getattr(self, f"_{e}_key_var").set(self.cfg_state.get(key_name))
            saved_model = self.cfg_state.get(f"{e.upper()}_MODEL")
            if saved_model:
                getattr(self, f"_{e}_model_var").set(saved_model)
            saved_tutor = self.cfg_state.get(f"{e.upper()}_TUTOR_MODEL")
            if saved_tutor:
                getattr(self, f"_{e}_tutor_var").set(saved_tutor)
            saved_budget = self.cfg_state.get(f"{e.upper()}_THINKING_BUDGET")
            if saved_budget:
                getattr(self, f"_{e}_budget_var").set(saved_budget)

        if hasattr(self, "_deepseek_effort_var"):
            saved_effort = self.cfg_state.get("DEEPSEEK_REASONING_EFFORT")
            if saved_effort:
                self._deepseek_effort_var.set(saved_effort)

        saved_st = self.cfg_state.get("PS_SEARCH_TIME_MS")
        if saved_st:
            self.search_time_var.set(saved_st)
        saved_par = self.cfg_state.get("PS_SEARCH_PARALLELISM")
        if saved_par:
            self.parallelism_var.set(saved_par)

    def _on_state_event(self, kind: str) -> None:
        if kind in ("reload", "save"):
            self.refresh()
