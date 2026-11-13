"""模型权重下载线程。

``downloader`` 缺省走抠图的 ONNX 权重下载；其他模块（如超分）传入自己的
``download_weight`` 闭包即可复用同一个线程与进度信号。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal

from ..core.log import get_logger
from ..core.types import TaskCancelled

log = get_logger("download-worker")

Downloader = Callable[[str, Any, Any, Any], Any]


class DownloadWorker(QThread):
    """在后台下载模型权重，避免界面卡住。"""

    progress = Signal(int, int)   # 已下载, 总大小（可能为 0）
    done = Signal(str)            # 保存路径
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        model: str,
        config: Any = None,
        parent=None,
        downloader: Optional[Downloader] = None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.config = config
        self._downloader = downloader

    def cancel(self) -> None:
        self.requestInterruption()

    def run(self) -> None:  # noqa: D102
        try:
            if self._downloader is None:
                from ..modules.matting.backends import download_model

                fn = download_model
            else:
                fn = self._downloader

            path = fn(
                self.model,
                self.config,
                on_progress=lambda done, total: self.progress.emit(int(done), int(total or 0)),
                cancel=self.isInterruptionRequested,
            )
            self.done.emit(str(path))
        except TaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            log.exception("权重下载失败")
            self.failed.emit(str(exc))
