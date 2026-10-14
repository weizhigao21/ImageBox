"""pyupdater：给打包成 exe 的 Windows 应用做「检查更新 → 下载 → 独立进程替换重启」。

分工：
- `pyupdater.check`  查 GitHub Releases、比较版本、下载更新包（纯标准库 + requests）
- `pyupdater.apply`  校验/解压/合并替换/备份回滚/进程协作（纯标准库）
- `pyupdater.updater_main`  独立更新程序的入口（打包成 updater.exe）
- `pyupdater.qt`     可选的 PySide6 现成对话框（宿主是 Qt 程序时用）

设计要点（为什么要独立进程）：Windows 会锁定正在运行的 exe，主程序无法
覆盖自己；因此更新动作交给一个独立的小程序，它被复制到 %TEMP% 运行，
等主程序退出后再替换整个程序目录，最后重新启动主程序。

公共 API 采用**惰性导入**：这样只用到 `apply`（更新程序那一侧）时不会
连带导入 requests，打出来的 updater.exe 更小、依赖更少。
"""
from __future__ import annotations

import importlib

__version__ = "1.0.0"

# 名称 → (模块, 属性)；首次访问时才导入
_LAZY_API: dict[str, tuple[str, str]] = {
    # pyupdater.apply（纯标准库）
    "ApplyError": ("pyupdater.apply", "ApplyError"),
    "extract_payload": ("pyupdater.apply", "extract_payload"),
    "finish_update": ("pyupdater.apply", "finish_update"),
    "inspect_zip": ("pyupdater.apply", "inspect_zip"),
    "launch_updater": ("pyupdater.apply", "launch_updater"),
    "payload_prefix": ("pyupdater.apply", "payload_prefix"),
    "replace_program_files": ("pyupdater.apply", "replace_program_files"),
    "rollback": ("pyupdater.apply", "rollback"),
    "start_app": ("pyupdater.apply", "start_app"),
    "wait_for_process_exit": ("pyupdater.apply", "wait_for_process_exit"),
    # pyupdater.check（需要 requests）
    "ReleaseInfo": ("pyupdater.check", "ReleaseInfo"),
    "UpdateError": ("pyupdater.check", "UpdateError"),
    "cleanup_cache": ("pyupdater.check", "cleanup_cache"),
    "download_asset": ("pyupdater.check", "download_asset"),
    "fetch_latest_release": ("pyupdater.check", "fetch_latest_release"),
    "is_newer": ("pyupdater.check", "is_newer"),
    "parse_release": ("pyupdater.check", "parse_release"),
    "parse_version": ("pyupdater.check", "parse_version"),
    "pick_asset": ("pyupdater.check", "pick_asset"),
    # pyupdater.config（纯标准库）
    "ConfigError": ("pyupdater.config", "ConfigError"),
    "UpdateConfig": ("pyupdater.config", "UpdateConfig"),
}

__all__ = ["__version__", *sorted(_LAZY_API)]


def __getattr__(name: str):
    """PEP 562：按需导入公共 API（避免 `import pyupdater` 就拉进 requests）。"""
    try:
        module_name, attr = _LAZY_API[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_API))
