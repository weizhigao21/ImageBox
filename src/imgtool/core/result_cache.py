"""UI 结果暂存：完整像素写入临时目录，只保留当前图片在内存中。"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from threading import RLock

import numpy as np

from .types import ImageItem


class ResultCache:
    def __init__(self) -> None:
        self._temp = None
        self._entries: dict[str, tuple[ImageItem, Path]] = {}
        self._lock = RLock()

    def store(self, item: ImageItem) -> None:
        """后台线程落盘，避免在 itemDone 回调里写大数组。"""
        if item.rgb is None or item.alpha is None:
            return
        with self._lock:
            if self._temp is None:
                self._temp = tempfile.TemporaryDirectory(prefix="imgtool-results-")
            key = str(item.path)
            self._entries.pop(key, None)
            folder = Path(self._temp.name) / hashlib.sha256(key.encode("utf-8")).hexdigest()
            folder.mkdir(exist_ok=True)
            for name, array in (
                ("rgb", item.rgb), ("alpha", item.alpha),
                ("result_rgb", item.meta.get("result_rgb", item.rgb)),
                ("edit_base_alpha", item.meta.get("edit_base_alpha", item.alpha)),
            ):
                np.save(folder / f"{name}.npy", array, allow_pickle=False)
            self._entries[key] = (item, folder)

    def load(self, item: ImageItem) -> None:
        with self._lock:
            entry = self._entries.get(str(item.path))
            if entry is None or (item.rgb is not None and item.alpha is not None):
                return
            folder = entry[1]
            item.rgb = np.load(folder / "rgb.npy", allow_pickle=False)
            item.alpha = np.load(folder / "alpha.npy", allow_pickle=False)
            item.meta["result_rgb"] = np.load(folder / "result_rgb.npy", allow_pickle=False)
            item.meta["edit_base_alpha"] = np.load(folder / "edit_base_alpha.npy", allow_pickle=False)
            item.meta["preview_alpha"] = item.alpha

    def release(self, item: ImageItem) -> None:
        with self._lock:
            if str(item.path) not in self._entries:
                return
            item.rgb = None
            item.alpha = None
            item.meta.pop("result_rgb", None)
            item.meta.pop("preview_alpha", None)
            item.meta.pop("edit_base_alpha", None)

    def evict_except(self, current: ImageItem) -> None:
        with self._lock:
            for item, _ in self._entries.values():
                if item is not current:
                    self.release(item)

    def remove(self, path: str) -> None:
        with self._lock:
            entry = self._entries.pop(path, None)
            if entry is not None:
                folder = entry[1]
                for name in ("rgb", "alpha", "result_rgb", "edit_base_alpha"):
                    (folder / f"{name}.npy").unlink(missing_ok=True)
                folder.rmdir()

    def close(self) -> None:
        with self._lock:
            self._entries.clear()
            if self._temp is not None:
                self._temp.cleanup()
                self._temp = None
