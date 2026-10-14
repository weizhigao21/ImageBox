"""文件拖放支持。

为什么需要它：Qt 只会把拖放事件派发给「接受拖放」的控件，而窗口里大部分区域
（预览、参数面板）默认不接受，于是把文件拖到界面上会毫无反应。
另外 QTableWidget / QGraphicsView 这类滚动区域的拖放事件其实是发给 ``viewport()`` 的，
只对控件本身 ``setAcceptDrops(True)`` 依然收不到事件。

这里提供一个可复用的中继：挂到任意控件上，它会同时给控件和它的 viewport 打开拖放，
把「拖进来的本地文件/文件夹」转成 ``dropped`` 信号。
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QMimeData, QObject, Signal
from PySide6.QtWidgets import QWidget

# 拖放事件里可能带 mimeData()，但合成事件下类型会丢失，这里统一判空
def local_paths(mime) -> list[str]:
    """从 MIME 数据里取出本地文件/文件夹路径（不是 QMimeData 时返回空）。"""
    if not isinstance(mime, QMimeData):
        return []
    if not mime.hasUrls():
        return []
    return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]


def event_paths(event) -> list[str]:
    """安全地取事件里的本地路径。"""
    try:
        return local_paths(event.mimeData())
    except (AttributeError, RuntimeError):
        return []


class FileDropRelay(QObject):
    """把某个控件的文件拖放转换成信号。"""

    dropped = Signal(list)      # [路径, ...]
    dragActive = Signal(bool)   # 有文件悬停在上面 / 离开

    def __init__(self, widget: QWidget, parent: Optional[QObject] = None) -> None:
        super().__init__(parent or widget)
        self._widget = widget
        widget.setAcceptDrops(True)
        # 滚动区域的拖放事件发给 viewport，必须一起打开并一起监听
        viewport = getattr(widget, "viewport", None)
        if callable(viewport):
            try:
                vp = widget.viewport()
                vp.setAcceptDrops(True)
                vp.installEventFilter(self)
            except (AttributeError, RuntimeError):
                pass
        widget.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        etype = event.type()
        if etype == QEvent.Type.DragLeave:
            self.dragActive.emit(False)
            return False
        if etype not in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            return False

        paths = event_paths(event)
        if not paths:
            return False

        if etype == QEvent.Type.Drop:
            self.dragActive.emit(False)
            self.dropped.emit(paths)
        else:
            self.dragActive.emit(True)
        event.acceptProposedAction()
        return True
