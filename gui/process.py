"""Bot process lifecycle for the PsyMew GUI.

Launches `python -u start.py` so it gets its own visible console window
(`CREATE_NEW_CONSOLE` on Windows; default behavior elsewhere). The bot's
stdout therefore lands in that console, not in a pipe captured by the GUI
— users can read it directly or close the window to stop the bot, which
is the explicit UX contract.

The GUI still wants a live log + W/L counter, so we tail
`logs/init.log` in a background thread instead. `start.py` honors
`PS_LOG_TO_FILE=1` and writes its full DEBUG transcript there.

Stop() is kept for one purpose: graceful cleanup if the GUI itself
closes while the bot is running. It still kills only the bot PID — the
detached loader survives, per the project invariant.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil


class BotProcess:
    """Single-bot lifecycle manager."""

    def __init__(
        self,
        repo_root: Path,
        on_line: Callable[[str], None],
        on_state: Callable[[str], None],
    ) -> None:
        self.repo_root = repo_root
        self.on_line = on_line
        self.on_state = on_state
        self.proc: subprocess.Popen[str] | None = None
        self._tailer: threading.Thread | None = None
        self._tailer_stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self._monitor_stop = threading.Event()

    # ---- lifecycle -----------------------------------------------------

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def start(self, extra_env: dict[str, str] | None = None) -> None:
        if self.is_running():
            return

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        # Force the log file on so the GUI's log tail has something to read.
        env["PS_LOG_TO_FILE"] = "1"

        log_path = self.repo_root / "logs" / "init.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Remember the file's size at launch so we only tail NEW lines from
        # this session (not the bot's previous run, which would be confusing).
        start_offset = log_path.stat().st_size if log_path.is_file() else 0

        creationflags = 0
        popen_kwargs: dict = {}
        if sys.platform == "win32":
            # New console = users see the bot's own window. CREATE_NEW_PROCESS_GROUP
            # is also set so the bot can receive Ctrl-Break if we ever need to send
            # one; the loader remains independently detached by start.py itself.
            creationflags = (
                subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
            )
            popen_kwargs["creationflags"] = creationflags

        # `-u` makes Python's stdout unbuffered so the new-console window shows
        # output line-by-line instead of in big delayed bursts.
        cmd = [sys.executable, "-u", "start.py"]

        self.on_state("starting")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.repo_root),
            env=env,
            **popen_kwargs,
        )

        # CRITICAL invariant (do not change): self.proc.pid is the bot.
        # start.py Popens a detached native-engine loader BEFORE its
        # os.execv into showdown.py — that loader is in a different
        # process group and we never touch it. Stop() targets only
        # this PID.

        self._start_log_tail(log_path, start_offset)
        self._start_process_monitor()
        self.on_state("running")

    def stop(self, grace: float = 5.0) -> None:
        """Kill the bot only — loader is left alive deliberately."""
        if not self.proc or self.proc.poll() is not None:
            self._shutdown_threads()
            self.on_state("stopped")
            return

        pid = self.proc.pid
        try:
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=grace)
            except psutil.TimeoutExpired:
                p.kill()
        except psutil.NoSuchProcess:
            pass

        self._shutdown_threads()
        self.on_state("stopped")

    # ---- background threads -------------------------------------------

    def _start_log_tail(self, log_path: Path, start_offset: int) -> None:
        self._tailer_stop.clear()

        def tailer() -> None:
            # Wait for the file to appear (the bot may take a beat to create it).
            for _ in range(40):  # ~8s max
                if log_path.is_file():
                    break
                if self._tailer_stop.is_set():
                    return
                time.sleep(0.2)
            else:
                return  # never appeared

            try:
                f = log_path.open("r", encoding="utf-8", errors="replace")
            except OSError:
                return

            try:
                f.seek(start_offset)
                pending = ""
                while not self._tailer_stop.is_set():
                    chunk = f.read()
                    if chunk:
                        pending += chunk
                        while "\n" in pending:
                            line, pending = pending.split("\n", 1)
                            self.on_line(line.rstrip("\r"))
                    else:
                        # Detect rotation: if file shrank, reopen.
                        try:
                            current_size = log_path.stat().st_size
                        except OSError:
                            current_size = -1
                        if current_size != -1 and current_size < f.tell():
                            f.close()
                            try:
                                f = log_path.open("r", encoding="utf-8", errors="replace")
                            except OSError:
                                return
                            pending = ""
                        time.sleep(0.25)
                if pending:
                    self.on_line(pending.rstrip("\r"))
            finally:
                try:
                    f.close()
                except Exception:  # noqa: BLE001
                    pass

        self._tailer = threading.Thread(target=tailer, daemon=True, name="BotLogTail")
        self._tailer.start()

    def _start_process_monitor(self) -> None:
        self._monitor_stop.clear()

        def monitor() -> None:
            assert self.proc is not None
            code = self.proc.wait()
            # Bot has exited (either user closed the console, or it crashed).
            self._tailer_stop.set()
            new_state = "crashed" if code not in (0, None) else "stopped"
            self.on_state(new_state)

        self._monitor = threading.Thread(target=monitor, daemon=True, name="BotMonitor")
        self._monitor.start()

    def _shutdown_threads(self) -> None:
        self._tailer_stop.set()
        self._monitor_stop.set()
