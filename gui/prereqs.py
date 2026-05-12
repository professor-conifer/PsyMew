"""Prerequisite detection + install helpers for the Setup tab.

This module knows how to *find* the toolchain bits PsyMew depends on
(Python version, Rust, MSVC build tools, required Python packages,
the compiled poke-engine native module) and how to *kick off* their
installs in a worker thread so the Setup tab can stream output to the
user.

Detection is deliberately fault-tolerant: anything that can throw is
wrapped, because the whole point of this tab is to *survive* missing
tooling and tell the user about it instead of crashing.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PrereqStatus:
    """One row in the Setup tab.

    `severity` is one of:
      - "ok"        — green check, nothing to do
      - "missing"   — red x, blocks PsyMew from running fully
      - "warn"      — yellow, optional-but-recommended
      - "unknown"   — grey, couldn't determine (e.g. permission error)
    """

    key: str           # short stable id, e.g. "rust", "cpp", "pip:openai"
    label: str         # human-readable name
    severity: str
    detail: str        # version string or short error
    blurb: str = ""    # one-line explanation of why this matters


@dataclass
class InstallTask:
    """A queued install action."""

    title: str
    runner: Callable[[Callable[[str], None]], int]
    # When the install finishes, optional post-action (e.g. open a URL).
    on_done: Callable[[int], None] | None = None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_python() -> PrereqStatus:
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = ".".join(str(x) for x in sys.version_info[:3])
    if (major, minor) >= (3, 11):
        return PrereqStatus(
            "python",
            "Python interpreter",
            "ok",
            f"{version_str} ({sys.executable})",
            "Need 3.11 or newer — you're good.",
        )
    return PrereqStatus(
        "python",
        "Python interpreter",
        "missing",
        f"{version_str} (need 3.11+)",
        "Install a newer Python from python.org and re-run psymew_gui.py with it.",
    )


def detect_pip() -> PrereqStatus:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return PrereqStatus(
                "pip",
                "pip (Python package manager)",
                "ok",
                result.stdout.strip().splitlines()[0],
                "Ships with Python — just confirming it's reachable.",
            )
    except (OSError, subprocess.TimeoutExpired):
        pass
    return PrereqStatus(
        "pip",
        "pip (Python package manager)",
        "missing",
        "pip --version failed",
        "Re-install Python with the 'Add pip' option ticked.",
    )


def detect_rust() -> PrereqStatus:
    rustc = shutil.which("rustc")
    if not rustc:
        return PrereqStatus(
            "rust",
            "Rust toolchain",
            "warn",  # only required if the user runs the MCTS engine
            "Not found on PATH",
            "Only needed for the MCTS engine and to compile poke-engine from source.",
        )
    try:
        result = subprocess.run(
            [rustc, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return PrereqStatus(
                "rust",
                "Rust toolchain",
                "ok",
                result.stdout.strip(),
                "Required to compile the poke-engine MCTS backend.",
            )
    except (OSError, subprocess.TimeoutExpired):
        pass
    return PrereqStatus(
        "rust",
        "Rust toolchain",
        "warn",
        "rustc found but version probe failed",
        "Only needed for the MCTS engine.",
    )


def detect_cpp_build_tools() -> PrereqStatus:
    """Windows-only: detect MSVC build tools via vswhere.

    On non-Windows we return an `ok` status — Linux and macOS use the
    system C compiler which is almost always present or trivially
    installable via the platform's normal channels.
    """
    if platform.system() != "Windows":
        return PrereqStatus(
            "cpp",
            "C/C++ build tools",
            "ok",
            f"{platform.system()} — uses system compiler",
            "On Linux/Mac the system gcc/clang is used; nothing extra needed in most cases.",
        )

    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"

    if not vswhere.is_file():
        return PrereqStatus(
            "cpp",
            "Visual Studio Build Tools",
            "missing",
            "VS Installer not present (no vswhere.exe)",
            "Required to compile poke-engine on Windows. Install the C++ workload.",
        )

    try:
        result = subprocess.run(
            [
                str(vswhere),
                "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-format", "json",
                "-utf8",
            ],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PrereqStatus(
            "cpp",
            "Visual Studio Build Tools",
            "unknown",
            f"vswhere failed: {exc}",
            "Couldn't probe — VS may or may not be installed.",
        )

    if result.returncode != 0:
        return PrereqStatus(
            "cpp",
            "Visual Studio Build Tools",
            "missing",
            f"vswhere returned {result.returncode}",
            "C++ build tools workload likely not installed.",
        )

    try:
        instances = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        instances = []

    if not instances:
        return PrereqStatus(
            "cpp",
            "Visual Studio Build Tools",
            "missing",
            "No VS install with C++ workload",
            "Install Build Tools 2022 and tick 'Desktop development with C++'.",
        )

    inst = instances[0]
    name = inst.get("displayName", "Visual Studio")
    version = inst.get("installationVersion", "")
    return PrereqStatus(
        "cpp",
        "Visual Studio Build Tools",
        "ok",
        f"{name} {version}",
        "Provides the MSVC linker poke-engine needs to compile.",
    )


# ---------------------------------------------------------------------------
# Python package detection
# ---------------------------------------------------------------------------


# (importable_name, friendly_name, requirements_pin) — kept in sync with
# requirements.txt manually because that file is the source of truth.
_REQUIRED_PACKAGES: list[tuple[str, str, str]] = [
    ("requests", "requests", "requests==2.32.4"),
    ("websockets", "websockets", "websockets==14.1"),
    ("dateutil", "python-dateutil", "python-dateutil==2.8.0"),
    ("poke_engine", "poke-engine", "poke-engine==0.0.46"),
    ("google.genai", "google-genai", "google-genai>=1.0"),
    ("anthropic", "anthropic", "anthropic>=0.50"),
    ("openai", "openai", "openai>=1.40"),
    ("customtkinter", "customtkinter", "customtkinter>=5.2.2"),
    ("psutil", "psutil", "psutil>=5.9.8"),
]


def detect_python_packages() -> list[PrereqStatus]:
    statuses: list[PrereqStatus] = []
    for import_name, friendly, _pin in _REQUIRED_PACKAGES:
        statuses.append(_check_package(import_name, friendly))
    return statuses


def _check_package(
    import_name: str,
    friendly: str,
    key: str | None = None,
) -> PrereqStatus:
    # Default key prefixes with "pip:" so the package rows in the Setup
    # tab's package summary don't collide with the standalone rows like
    # "rust" or "poke_engine". Callers building a standalone row must
    # pass an explicit `key` so the Setup tab's lookup finds it.
    resolved_key = key if key is not None else f"pip:{friendly}"
    try:
        mod = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001
        return PrereqStatus(
            resolved_key,
            friendly,
            "missing",
            f"not installed ({exc.__class__.__name__})",
            "",
        )

    version = (
        getattr(mod, "__version__", None)
        or getattr(mod, "VERSION", None)
        or "installed"
    )
    return PrereqStatus(resolved_key, friendly, "ok", str(version), "")


def detect_poke_engine() -> PrereqStatus:
    """Check the compiled native module specifically.

    `poke_engine` is the one Python package that's nigh-guaranteed to
    require Rust + a C linker on the user's machine.  Separating its
    status from the generic pip list makes the relationship between
    Rust/CPP tools and this module legible.

    NOTE: the explicit `key="poke_engine"` is load-bearing — the Setup
    tab's refresh() looks the row up by that exact key.
    """
    return _check_package(
        "poke_engine",
        "poke-engine (MCTS engine)",
        key="poke_engine",
    )


def missing_packages() -> list[tuple[str, str, str]]:
    """Return the requirements.txt pin for each package that isn't importable."""
    out = []
    for import_name, friendly, pin in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except Exception:  # noqa: BLE001
            out.append((import_name, friendly, pin))
    return out


# ---------------------------------------------------------------------------
# Install runners
# ---------------------------------------------------------------------------


def winget_available() -> bool:
    return platform.system() == "Windows" and shutil.which("winget") is not None


def stream_command(
    cmd: list[str],
    on_line: Callable[[str], None],
    cwd: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> int:
    """Run `cmd` in a subprocess, forwarding each output line to `on_line`.

    Captures stdout + stderr together so users see warnings interleaved.
    Returns the subprocess exit code, or -1 on launch failure.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    on_line(f"$ {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        on_line(f"[error] command not found: {cmd[0]}")
        return -1
    except OSError as exc:
        on_line(f"[error] could not launch: {exc}")
        return -1

    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        on_line(line.rstrip("\r\n"))
    code = proc.wait()
    on_line(f"--- exit code {code} ---")
    return code


def install_rust_via_winget(on_line: Callable[[str], None]) -> int:
    """Install Rust via winget (preferred on Windows 10/11)."""
    if not winget_available():
        on_line("[error] winget not on PATH — use the Open rustup.rs button instead.")
        return -1
    return stream_command(
        ["winget", "install", "--id", "Rustlang.Rustup", "-e", "--silent",
         "--accept-source-agreements", "--accept-package-agreements"],
        on_line,
    )


def install_vs_build_tools_via_winget(on_line: Callable[[str], None]) -> int:
    """Install VS Build Tools 2022 with the C++ workload preselected.

    Note: even with --silent, this often takes 10–30 minutes and may pop a
    UAC prompt.  The user should be encouraged to leave the GUI running
    until the exit-code line appears.
    """
    if not winget_available():
        on_line("[error] winget not on PATH — use the download button instead.")
        return -1
    return stream_command(
        [
            "winget", "install", "--id", "Microsoft.VisualStudio.2022.BuildTools", "-e",
            "--silent", "--accept-source-agreements", "--accept-package-agreements",
            "--override",
            (
                "--add Microsoft.VisualStudio.Workload.VCTools "
                "--includeRecommended --quiet --wait --norestart --nocache"
            ),
        ],
        on_line,
    )


def install_missing_packages(
    on_line: Callable[[str], None],
    repo_root: Path,
) -> int:
    """Run `pip install -r requirements.txt` in the active interpreter."""
    req = repo_root / "requirements.txt"
    if not req.is_file():
        on_line(f"[error] {req} not found")
        return -1
    return stream_command(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
        on_line,
    )


def rebuild_poke_engine(on_line: Callable[[str], None]) -> int:
    """Force-reinstall poke-engine — useful when switching generations.

    Mirrors the command in the project's Makefile.
    """
    return stream_command(
        [
            sys.executable, "-m", "pip", "install", "-v", "--force-reinstall",
            "--no-cache-dir", "poke-engine",
            "--config-settings",
            "build-args=--features poke-engine/terastallization --no-default-features",
        ],
        on_line,
    )


# ---------------------------------------------------------------------------
# Threaded task runner
# ---------------------------------------------------------------------------


class TaskRunner:
    """Runs InstallTasks one at a time in a daemon thread."""

    def __init__(self, on_line: Callable[[str], None], on_idle: Callable[[], None]):
        self.on_line = on_line
        self.on_idle = on_idle
        self._lock = threading.Lock()
        self._busy = False
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, task: InstallTask) -> bool:
        """Start `task` if no other task is running. Returns True if accepted."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def worker() -> None:
            try:
                self.on_line("")
                self.on_line(f"=== {task.title} ===")
                code = task.runner(self.on_line)
                if task.on_done is not None:
                    task.on_done(code)
            except Exception as exc:  # noqa: BLE001
                self.on_line(f"[error] task crashed: {exc.__class__.__name__}: {exc}")
            finally:
                with self._lock:
                    self._busy = False
                self.on_idle()

        self._thread = threading.Thread(
            target=worker, daemon=True, name=f"prereq:{task.title}"
        )
        self._thread.start()
        return True
