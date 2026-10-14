"""支持画笔涂抹的图片视图。

在 ImageView 基础上增加：按住左键拖动即产生笔画（坐标是图片像素坐标），
鼠标上显示笔刷圆圈（按住 Alt 可临时切换擦除/保留）。
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsView

from .image_view import ImageView

ERASE = 1
KEEP = 2

_COLORS = {ERASE: QColor(255, 92, 92), KEEP: QColor(96, 220, 128)}


class PaintImageView(ImageView):
    """可涂抹的预览视图。"""

    strokeBegin = Signal(float, float, int)          # x, y, 值(1=擦除 2=保留)
    strokeMove = Signal(float, float, float, float)  # x0, y0, x1, y1
    strokeEnd = Signal()
    undoRequested = Signal()
    brushSizeStep = Signal(int)                      # -1 变小 / +1 变大

    def __init__(self, placeholder: str = "无图像", parent=None) -> None:
        super().__init__(placeholder, parent)
        self.painting_enabled = False
        self.brush_radius = 24.0
        self.brush_value = ERASE
        self._last_scene: Optional[tuple[float, float]] = None
        self._hover: Optional[QPointF] = None
        self._hover_value = ERASE
        self._edit_overlay = QGraphicsPixmapItem()
        self._edit_overlay.setZValue(1)
        self._scene.addItem(self._edit_overlay)
        self._overlay_pixmap: Optional[QPixmap] = None
        self._edit_overlay.hide()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    def set_painting_enabled(self, enabled: bool) -> None:
        self.painting_enabled = bool(enabled)
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag
        )
        if not self._panning:
            self._restore_cursor()
        if not self.painting_enabled:
            self._clear_stroke_overlay()
        self.viewport().update()

    def _clear_stroke_overlay(self) -> None:
        self._overlay_pixmap = None
        self._edit_overlay.hide()
        self._edit_overlay.setPixmap(QPixmap())

    def _draw_stroke_overlay(self, start: QPointF, end: QPointF, value: int) -> None:
        """只高亮当前正在画的一笔，松开鼠标后立即清除。"""
        source = self._item.pixmap()
        if source.isNull():
            return
        if self._overlay_pixmap is None or self._overlay_pixmap.size() != source.size():
            self._overlay_pixmap = QPixmap(source.size())
            self._overlay_pixmap.fill(Qt.GlobalColor.transparent)
        color = QColor(_COLORS.get(value, _COLORS[ERASE]))
        color.setAlpha(145 if value == ERASE else 130)
        painter = QPainter(self._overlay_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if start == end:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(end, self.brush_radius, self.brush_radius)
        else:
            pen = QPen(color, self.brush_radius * 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(start, end)
        painter.end()
        self._edit_overlay.setPixmap(self._overlay_pixmap)
        self._edit_overlay.show()

    def clear_image(self) -> None:
        super().clear_image()
        self._clear_stroke_overlay()

    def _idle_cursor(self):
        # 涂抹开启时用十字光标；中键拖动时基类会临时换成握拳
        return Qt.CursorShape.CrossCursor if self.painting_enabled else Qt.CursorShape.ArrowCursor

    def _scene_pos(self, event: QMouseEvent) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _effective_value(self, event: QMouseEvent) -> int:
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            return KEEP if self.brush_value == ERASE else ERASE
        return self.brush_value

    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            self.painting_enabled
            and event.button() == Qt.MouseButton.LeftButton
            and not self._item.pixmap().isNull()
        ):
            pos = self._scene_pos(event)
            self._last_scene = (pos.x(), pos.y())
            value = self._effective_value(event)
            self._hover_value = value
            self._draw_stroke_overlay(pos, pos, value)
            self.strokeBegin.emit(pos.x(), pos.y(), value)
            self.viewport().update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:  # 中键平移优先
            super().mouseMoveEvent(event)
            return
        self._hover = self._scene_pos(event)
        self._hover_value = self._effective_value(event)
        if self.painting_enabled and self._last_scene is not None:
            pos = self._hover
            self._draw_stroke_overlay(QPointF(*self._last_scene), pos, self._hover_value)
            self.strokeMove.emit(self._last_scene[0], self._last_scene[1], pos.x(), pos.y())
            self._last_scene = (pos.x(), pos.y())
            self.viewport().update()
            event.accept()
            return
        self.viewport().update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._last_scene is not None:
            self._last_scene = None
            self.strokeEnd.emit()
            self._clear_stroke_overlay()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = None
        self.viewport().update()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undoRequested.emit()
            event.accept()
            return
        if event.text() == "[":
            self.brushSizeStep.emit(-1)
            event.accept()
            return
        if event.text() == "]":
            self.brushSizeStep.emit(1)
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    def drawForeground(self, painter: QPainter, rect) -> None:  # noqa: N802, D102
        super().drawForeground(painter, rect)
        if not self.painting_enabled or self._hover is None or self._panning:
            return
        color = _COLORS.get(self._hover_value, _COLORS[ERASE])
        radius = max(1.0, float(self.brush_radius))
        if self._last_scene is None:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        else:
            fill = QColor(color)
            fill.setAlpha(40)
            painter.setBrush(fill)
        outline = QPen(QColor(0, 0, 0, 220), 3)
        outline.setCosmetic(True)
        painter.setPen(outline)
        painter.drawEllipse(self._hover, radius + 2.0, radius + 2.0)
        inner = QPen(color, 2)
        inner.setCosmetic(True)
        painter.setPen(inner)
        painter.drawEllipse(self._hover, radius, radius)
