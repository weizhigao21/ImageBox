"""统一日志配置。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """初始化根日志：控制台 + 可选文件。重复调用只生效一次。"""
    global _CONFIGURED
    root = logging.getLogger("imgtool")
    if _CONFIGURED:
        return

    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    root.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(fh)
        except OSError:
            root.warning("无法写入日志文件：%s", log_file)

    _CONFIGURED = True


def get_logger(name: str = "imgtool") -> logging.Logger:
    if not name.startswith("imgtool"):
        name = f"imgtool.{name}"
    return logging.getLogger(name)
