"""抠图后端基类。"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ....core.base import BaseBackend


class MattingBackend(BaseBackend):
    """抠图后端：输入 RGB，输出 0~1 的前景透明度。"""

    #: 输出是否为“脏掩码”（需要连通域/填洞清理）
    post_cleanup: bool = False

    @classmethod
    def describe(cls, config: Any = None) -> str:
        ok, reason = cls.available(config)
        return "可用" if ok else f"不可用（{reason}）"

    @abstractmethod
    def predict(self, rgb: np.ndarray, hint: np.ndarray | None = None) -> np.ndarray:
        ...
