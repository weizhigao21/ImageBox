"""Qt 与 numpy 之间的桥接工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy

from ..core import io_utils

PREVIEW_MAX = 2048


class ElidedLabel(QLabel):
    """保留完整 text()/提示，只在绘制时省略，避免长状态撑宽布局。"""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.setToolTip(text)

    def setText(self, text: str) -> None:  # noqa: N802
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setPen(self.palette().windowText().color())
        painter.setFont(self.font())
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideMiddle, self.contentsRect().width())
        painter.drawText(self.contentsRect(), self.alignment(), text)


def ndarray_to_qimage(rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> QImage:
    """HxWx3 uint8 RGB（可带 alpha）→ QImage（已拷贝，安全脱离 numpy 内存）。"""
    arr = np.ascontiguousarray(rgb)
    if arr.ndim == 2:
        arr = np.dstack([arr, arr, arr])
    h, w = arr.shape[:2]

    if alpha is not None:
        a = np.ascontiguousarray(alpha)
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        if a.ndim == 3:
            a = a[:, :, 0]
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        rgba = np.dstack([arr, a])
        return QImage(
            rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888
        ).copy()

    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def ndarray_to_qpixmap(rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> QPixmap:
    return QPixmap.fromImage(ndarray_to_qimage(rgb, alpha))


def downscale(rgb: np.ndarray, max_size: int = PREVIEW_MAX) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_size) / float(max(h, w)))
    if scale >= 1.0:
        return rgb
    return cv2.resize(
        rgb,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def load_preview(path: str | Path, max_size: int = PREVIEW_MAX) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """读图并缩放到预览尺寸，返回 (RGB, alpha 或 None)。"""
    rgb, alpha = io_utils.read_image_rgba(path)
    if alpha is not None and alpha.shape[:2] != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_size) / float(max(h, w)))
    if scale < 1.0:
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    return rgb, alpha


def checkerboard_pixmap(cell: int = 12, c1: str = "#f5f5f5", c2: str = "#e9e9eb") -> QPixmap:
    """生成棋盘格贴图，用于显示透明区域。"""
    size = cell * 2
    image = QImage(size, size, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.transparent)
    from PySide6.QtGui import QColor, QPainter

    painter = QPainter(image)
    painter.fillRect(0, 0, size, size, QColor(c1))
    painter.fillRect(0, 0, cell, cell, QColor(c2))
    painter.fillRect(cell, cell, cell, cell, QColor(c2))
    painter.end()
    return QPixmap.fromImage(image)
