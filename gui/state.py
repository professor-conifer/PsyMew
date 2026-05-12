"""Central config state shared across GUI tabs.

Holds the current `.env` key/value mapping and notifies subscribers when
values change or are saved. Tabs read from and write to this object;
the app coordinates persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from gui.env_io import read_env, write_env


class ConfigState:
    def __init__(self, env_path: Path) -> None:
        self.env_path = env_path
        self.values: dict[str, str] = (
            read_env(env_path) if env_path.is_file() else {}
        )
        self._dirty = False
        self._listeners: list[Callable[[str], None]] = []

    @property
    def dirty(self) -> bool:
        return self._dirty

    def get(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)

    def set(self, key: str, value: str) -> None:
        current = self.values.get(key, "")
        if current == value:
            return
        if value == "":
            self.values.pop(key, None)
        else:
            self.values[key] = value
        self._dirty = True
        self._emit("change")

    def save(self) -> None:
        write_env(self.env_path, self.values)
        self._dirty = False
        self._emit("save")

    def reload(self) -> None:
        self.values = read_env(self.env_path) if self.env_path.is_file() else {}
        self._dirty = False
        self._emit("reload")

    def subscribe(self, cb: Callable[[str], None]) -> None:
        self._listeners.append(cb)

    def _emit(self, kind: str) -> None:
        for cb in list(self._listeners):
            cb(kind)
