"""更新检查和下载线程，纳入主窗口安全退出管理。"""

import tempfile
import uuid
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..core.updates import UpdateService


class UpdateWorker(QThread):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(object, object)

    def __init__(self, service: UpdateService, cfg, release=None, parent=None):
        super().__init__(parent)
        self.service, self.cfg, self.release = service, cfg, release
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True
        self.requestInterruption()

    def run(self) -> None:
        try:
            if self.cancelled:
                return
            if self.release is None:
                result = self.service.check(self.cfg)
            else:
                destination = Path(tempfile.gettempdir()) / "ImageBox-updates" / uuid.uuid4().hex
                result = self.service.download(
                    self.cfg, self.release, destination,
                    on_progress=lambda d, t: self.progress.emit(d, t),
                    cancel=self.isInterruptionRequested,
                )
            if not self.cancelled:
                self.done.emit(result)
        except Exception as exc:
            if not self.cancelled:
                self.failed.emit(str(exc))
