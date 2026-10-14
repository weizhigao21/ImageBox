"""可缩放、带棋盘格透明底的图片视图。"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QImage, QPainter, QPixmap, QWheelEvent, QColor
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..qt_utils import checkerboard_pixmap


class ImageView(QGraphicsView):
    """显示一张图片，支持滚轮缩放、拖拽平移、双击复位。

    - 按住 **滚轮键（中键）** 拖动 = 平移图片（左键被涂抹占用时也能用）
    - 未开启涂抹时左键拖动也可平移；滚轮缩放；双击复位

    刷新策略：内部始终持有同一张 QPixmap，``update_region()`` 只把变化的那一小块
    画进去并只重绘该区域 —— 涂抹/缩放时不会整图重建，也不会把缩放弹回适应大小。
    """

    def __init__(self, placeholder: str = "无图像", parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self._pixmap: Optional[QPixmap] = None
        self._panning = False
        self._pan_anchor = QPoint()

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._checker = checkerboard_pixmap()
        self.setBackgroundBrush(self._checker)
        self._auto_fit = True

        from PySide6.QtWidgets import QLabel

        self._placeholder = QLabel(placeholder, self)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color:#6b7280; background:transparent;")
        self._placeholder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_preview_background(self, mode: str) -> None:
        if mode == "checker":
            self.setBackgroundBrush(self._checker)
        else:
            self.setBackgroundBrush(QColor("#ffffff" if mode == "white" else "#262629"))
        text_color = "#c0c0c4" if mode == "dark" else "#85858a"
        self._placeholder.setStyleSheet(f"color:{text_color}; background:transparent;")
        self.viewport().update()

    # ------------------------------------------------------------------
    def set_qimage(self, image: Optional[QImage], fit: Optional[bool] = None) -> None:
        """替换整张图。fit 为 None 时只在尺寸变化（或首次显示）时重新适应窗口。"""
        if image is None or image.isNull():
            self.clear_image()
            return
        pixmap = QPixmap.fromImage(image)
        previous = self._pixmap
        size_changed = previous is None or previous.isNull() or previous.size() != pixmap.size()
        self._pixmap = pixmap
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._placeholder.hide()
        if fit is None:
            fit = size_changed
        if fit:
            self.fit_to_view()

    def set_numpy(
        self,
        rgb: Optional[np.ndarray],
        alpha: Optional[np.ndarray] = None,
        fit: Optional[bool] = None,
    ) -> None:
        if rgb is None:
            self.clear_image()
            return
        from ..qt_utils import ndarray_to_qimage

        self.set_qimage(ndarray_to_qimage(rgb, alpha), fit=fit)

    def update_region(
        self,
        rgb_sub: np.ndarray,
        alpha_sub: Optional[np.ndarray],
        x: int,
        y: int,
    ) -> None:
        """只把一小块画进现有图像，并只重绘这一块。"""
        if self._pixmap is None or self._pixmap.isNull():
            return
        h, w = rgb_sub.shape[:2]
        if w <= 0 or h <= 0:
            return
        if x < 0 or y < 0 or x + w > self._pixmap.width() or y + h > self._pixmap.height():
            return
        from ..qt_utils import ndarray_to_qimage

        patch = ndarray_to_qimage(rgb_sub, alpha_sub)
        painter = QPainter(self._pixmap)
        # SourceOver 会让新补丁的透明像素透出旧画面，擦除时形成残影。
        # Source 才能连同 alpha 一起覆盖原区域。
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.drawImage(QPoint(int(x), int(y)), patch)
        painter.end()
        self._item.setPixmap(self._pixmap)
        self._item.update(QRectF(int(x), int(y), w, h))

    def clear_image(self) -> None:
        self._pixmap = None
        self._item.setPixmap(QPixmap())
        self._scene.setSceneRect(QRectF())
        self._placeholder.show()

    def set_placeholder(self, text: str) -> None:
        self._placeholder.setText(text)

    def fit_to_view(self) -> None:
        if self._item.pixmap().isNull():
            return
        # fitInView 在隐藏页切换后的滚动条布局中可能按旧视口尺寸缩放。
        # 用当前 viewport 明确计算比例，并在调整期间阻止 resizeEvent 重入。
        rect = self._item.sceneBoundingRect()
        viewport = self.viewport().size()
        if rect.width() <= 0 or rect.height() <= 0 or viewport.width() <= 6 or viewport.height() <= 6:
            return
        scale = min((viewport.width() - 6) / rect.width(), (viewport.height() - 6) / rect.height())
        self._auto_fit = False
        self.resetTransform()
        self.scale(scale, scale)
        self.centerOn(rect.center())
        self._auto_fit = True

    # ------------------------------------------------------------------
    # 中键（滚轮键）拖动平移：任何模式下都能用，包括左键被涂抹占用时
    def _start_pan(self, event) -> None:
        self._panning = True
        self._pan_anchor = event.position().toPoint()
        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

    def _end_pan(self) -> None:
        self._panning = False
        self._restore_cursor()

    def _restore_cursor(self) -> None:
        self.viewport().setCursor(self._idle_cursor())

    def _idle_cursor(self):
        return Qt.CursorShape.ArrowCursor

    def _pan_to(self, event) -> None:
        point = event.position().toPoint()
        delta = point - self._pan_anchor
        self._pan_anchor = point
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        self._auto_fit = False  # 手动平移过，窗口变化时不再自动弹回

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._item.pixmap().isNull():
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.zoom_by(factor)

    def zoom_by(self, factor: float) -> None:
        """以当前缩放为基础放大/缩小，并保留用户选择的比例。"""
        current = self.transform().m11()
        target = current * factor
        if 0.02 < target < 40:
            self._auto_fit = False
            self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton and not self._item.pixmap().isNull():
            self._start_pan(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            self._pan_to(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            self._end_pan()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit_to_view()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._placeholder.setGeometry(self.rect())
        if self._auto_fit:
            self.fit_to_view()
