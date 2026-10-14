"""抠图后端集合。"""

from __future__ import annotations

import importlib.util

from ....core.base import BaseBackend
from .base import MattingBackend
from .birefnet_backend import BiRefNetBackend
from .onnx_backend import OnnxBackend, download_model, list_local_models, resolve_model_file
from .simple_backend import SimpleBackend

BACKENDS: dict[str, type[MattingBackend]] = {
    SimpleBackend.name: SimpleBackend,
    OnnxBackend.name: OnnxBackend,
    BiRefNetBackend.name: BiRefNetBackend,
}


def create_backend(name: str, params: dict | None = None, device: str = "auto") -> MattingBackend:
    cls = BACKENDS.get(name)
    if cls is None:
        raise KeyError(f"未知抠图后端：{name}（可选：{', '.join(BACKENDS)}）")
    return cls(params, device=device)


def backend_status(config: Any = None, quick: bool = False) -> list[tuple[str, str, bool, str]]:
    """[(name, label, 是否可用, 说明)]，用于 UI 展示。"""
    rows = []
    for name, cls in BACKENDS.items():
        if quick and name == "birefnet":
            missing = [pkg for pkg in ("torch", "transformers") if importlib.util.find_spec(pkg) is None]
            ok = not missing
            reason = f"未安装 {', '.join(missing)}" if missing else "依赖已安装，使用时加载模型"
        else:
            ok, reason = cls.available(config)
        rows.append((name, cls.label, ok, reason))
    return rows


__all__ = [
    "MattingBackend",
    "SimpleBackend",
    "OnnxBackend",
    "BiRefNetBackend",
    "BACKENDS",
    "create_backend",
    "backend_status",
    "download_model",
    "list_local_models",
    "resolve_model_file",
    "BaseBackend",
]
