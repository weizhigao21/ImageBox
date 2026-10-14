"""模型权重下载线程。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThread, Signal

from ..core.log import get_logger
from ..core.types import TaskCancelled

log = get_logger("download-worker")


class DownloadWorker(QThread):
    """在后台下载 ONNX 权重，避免界面卡住。"""

    progress = Signal(int, int)   # 已下载, 总大小（可能为 0）
    done = Signal(str)            # 保存路径
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, model: str, config: Any = None, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.config = config

    def cancel(self) -> None:
        self.requestInterruption()

    def run(self) -> None:  # noqa: D102
        try:
            from ..modules.matting.backends import download_model

            path = download_model(
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
