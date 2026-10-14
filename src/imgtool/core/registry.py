"""模块注册表：自动发现 imgtool.modules 下的所有模块。

约定：每个模块包在 ``__init__.py`` 中暴露一个 ``MODULE`` 实例。
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Optional

from .base import BaseModule
from .log import get_logger

log = get_logger("registry")

_cache: Optional[list[BaseModule]] = None


def discover_modules(force: bool = False) -> list[BaseModule]:
    """扫描并返回所有模块实例（带缓存）。"""
    global _cache
    if _cache is not None and not force:
        return list(_cache)

    modules: list[BaseModule] = []
    try:
        import imgtool.modules as pkg  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        log.error("无法导入 imgtool.modules：%s", exc)
        _cache = []
        return []

    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        if not info.ispkg:
            continue
        try:
            mod = importlib.import_module(f"imgtool.modules.{info.name}")
        except Exception as exc:  # 单个模块坏掉不影响其他模块
            log.exception("模块 %s 导入失败：%s", info.name, exc)
            continue
        instance = getattr(mod, "MODULE", None)
        if isinstance(instance, BaseModule):
            modules.append(instance)
        else:
            log.warning("模块 %s 未暴露 MODULE 实例，已跳过", info.name)

    modules.sort(key=lambda m: (getattr(m, "order", 100), m.name))
    _cache = modules
    return list(modules)


def get_module(name: str) -> Optional[BaseModule]:
    for module in discover_modules():
        if module.name == name:
            return module
    return None
