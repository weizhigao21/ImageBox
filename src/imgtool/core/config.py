"""应用配置：读写 config/settings.json，并提供 models / output / presets 目录。"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .log import get_logger

log = get_logger("config")

DEFAULT_SETTINGS: dict[str, Any] = {
    "device": "auto",
    "models_dir": "models",
    "output_dir": "output",
    "presets_dir": "config/presets",
    "log_level": "INFO",
    "check_updates_on_startup": True,
}


def project_root() -> Path:
    """项目根目录。可用环境变量 IMGTOOL_HOME 覆盖。"""
    env = os.environ.get("IMGTOOL_HOME")
    if env:
        return Path(env).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # <root>/src/imgtool/core/config.py -> <root>
    return Path(__file__).resolve().parents[3]


@dataclass
class AppConfig:
    """轻量配置对象。"""

    root: Path = field(default_factory=project_root)
    data: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    path: Path | None = None

    # ---------- 读写 ----------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        root = project_root()
        cfg = cls(root=root)
        cfg.path = Path(path) if path is not None else root / "config" / "settings.json"
        if cfg.path.exists():
            try:
                loaded = json.loads(cfg.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cfg.data.update(loaded)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("配置文件读取失败，使用默认配置：%s", exc)
        return cfg

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else (self.path or self.root / "config" / "settings.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.path = target
        return target

    # ---------- 取值 ----------
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def sub(self, key: str) -> dict[str, Any]:
        """取一个子配置字典（不存在则创建）。"""
        value = self.data.get(key)
        if not isinstance(value, dict):
            value = {}
            self.data[key] = value
        return value

    # ---------- 目录 ----------
    def _dir(self, key: str, default_rel: str, create: bool = True) -> Path:
        raw = str(self.data.get(key) or default_rel)
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self.root / p
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_dir(self) -> Path:
        return self._dir("models_dir", "models")

    @property
    def output_dir(self) -> Path:
        return self._dir("output_dir", "output")

    @property
    def presets_dir(self) -> Path:
        return self._dir("presets_dir", "config/presets")

    @property
    def logs_dir(self) -> Path:
        return self._dir("logs_dir", "output/logs")

    def __repr__(self) -> str:  # pragma: no cover
        return f"AppConfig(root={self.root!r}, path={self.path!r})"
