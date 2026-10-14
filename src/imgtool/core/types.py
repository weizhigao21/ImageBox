"""通用数据类型：图片条目、任务上下文、处理结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class ImageItem:
    """一个待处理的图片。

    rgb   : HxWx3 uint8，RGB 顺序（懒加载）
    alpha : HxW uint8，0~255（抠图等模块的产物）
    meta  : 任意附加信息（尺寸、预览、耗时等）
    """

    path: Path
    rgb: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def size(self) -> tuple[int, int]:
        """(宽, 高)，未知时返回 (0, 0)。"""
        rgb = self.rgb
        if rgb is not None:
            return int(rgb.shape[1]), int(rgb.shape[0])
        wh = self.meta.get("size")
        if isinstance(wh, tuple) and len(wh) == 2:
            return int(wh[0]), int(wh[1])
        return 0, 0


@dataclass
class ModuleResult:
    """单个图片的处理结果。"""

    item: ImageItem
    ok: bool = True
    message: str = ""
    elapsed: float = 0.0
    outputs: dict[str, Path] = field(default_factory=dict)

    @property
    def output_dir(self) -> Optional[Path]:
        for p in self.outputs.values():
            return p.parent
        return None


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


class TaskCancelled(RuntimeError):
    """任务在安全的阶段边界取消。"""


class TaskContext:
    """传给模块的执行上下文：进度回调 + 取消检查 + 配置。"""

    def __init__(
        self,
        config: Any = None,
        on_progress: Optional[ProgressCallback] = None,
        cancel: Optional[CancelCallback] = None,
    ) -> None:
        self.config = config
        self._on_progress = on_progress
        self._cancel = cancel

    def progress(self, done: int, total: int, message: str = "") -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(int(done), int(total), str(message))
        except Exception:  # 回调异常不应中断处理
            pass

    def cancelled(self) -> bool:
        if self._cancel is None:
            return False
        try:
            return bool(self._cancel())
        except Exception:
            return False

    def check_cancelled(self) -> None:
        if self.cancelled():
            raise TaskCancelled("任务已取消")
