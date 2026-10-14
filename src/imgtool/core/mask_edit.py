"""手动掩码编辑（画笔涂抹）。

把用户用笔刷涂出来的「擦除 / 保留」记录成一层整数标记，再叠加到模型输出的 alpha 上：

    layer: 0 = 未涂抹，1 = 擦除（强制透明），2 = 保留（强制不透明）

图层按预览分辨率保存，叠加到原图分辨率时自动缩放（线性插值顺便带来柔和边缘）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np

NONE = 0
ERASE = 1
KEEP = 2


@dataclass
class Stroke:
    """一笔（一条折线）。"""

    points: list[tuple[float, float]] = field(default_factory=list)
    radius: float = 20.0
    value: int = ERASE

    def scaled(self, sx: float, sy: float, sr: float) -> "Stroke":
        return Stroke(
            points=[(x * sx, y * sy) for x, y in self.points],
            radius=max(1.0, self.radius * sr),
            value=self.value,
        )


class MaskEdit:
    """一张图上的手动涂抹记录。"""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.layer = np.zeros(self.shape, np.uint8)
        self._strokes: list[Stroke] = []
        self._current: Optional[Stroke] = None

    # ------------------------------------------------------------ 状态
    @property
    def stroke_count(self) -> int:
        return len(self._strokes) + (1 if self._current is not None else 0)

    @property
    def is_empty(self) -> bool:
        return not self._strokes and self._current is None

    @property
    def dirty_pixels(self) -> int:
        return int(np.count_nonzero(self.layer))

    # ------------------------------------------------------------ 涂抹
    def begin(self, x: float, y: float, radius: float, value: int) -> None:
        """开始一笔（落笔即画一个圆点）。"""
        self.end()
        self._current = Stroke(points=[(float(x), float(y))], radius=max(1.0, float(radius)), value=int(value))
        self._draw_dot(self._current.points[0], self._current.radius, self._current.value)

    def extend(self, x: float, y: float) -> None:
        """延续当前笔。"""
        if self._current is None:
            return
        last = self._current.points[-1]
        point = (float(x), float(y))
        self._current.points.append(point)
        self._draw_line(last, point, self._current.radius, self._current.value)

    def end(self) -> None:
        """结束当前笔，进入可撤销的列表。"""
        if self._current is None:
            return
        stroke = self._current
        self._current = None
        if len(stroke.points) >= 1:
            self._strokes.append(stroke)

    def undo(self) -> bool:
        """撤销最后一笔。"""
        if self._current is not None:
            self._current = None
            self._render()
            return True
        if not self._strokes:
            return False
        self._strokes.pop()
        self._render()
        return True

    def clear(self) -> None:
        self._strokes.clear()
        self._current = None
        self.layer[:] = 0

    # ------------------------------------------------------------ 绘制
    def _draw_dot(self, point: tuple[float, float], radius: float, value: int) -> None:
        cv2.circle(
            self.layer,
            (int(round(point[0])), int(round(point[1]))),
            max(1, int(round(radius))),
            int(value),
            -1,
            lineType=cv2.LINE_8,
        )

    def _draw_line(self, p0, p1, radius: float, value: int) -> None:
        cv2.line(
            self.layer,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            int(value),
            max(2, int(round(radius * 2))),
            lineType=cv2.LINE_8,
        )
        # 圆头，避免折线拐角出现缺口
        self._draw_dot(p1, radius, value)

    def _render(self) -> None:
        self.layer[:] = 0
        for stroke in self._strokes:
            if not stroke.points:
                continue
            self._draw_dot(stroke.points[0], stroke.radius, stroke.value)
            for a, b in zip(stroke.points, stroke.points[1:]):
                self._draw_line(a, b, stroke.radius, stroke.value)

    # ------------------------------------------------------------ 叠加
    def masks(self, shape: Optional[tuple[int, int]] = None) -> tuple[np.ndarray, np.ndarray]:
        """返回 (擦除权重, 保留权重)，都是 0~1 的 float32。"""
        erase = (self.layer == ERASE).astype(np.float32)
        keep = (self.layer == KEEP).astype(np.float32)
        target = tuple(shape) if shape is not None else self.shape
        if target != self.shape:
            size = (int(target[1]), int(target[0]))
            erase = cv2.resize(erase, size, interpolation=cv2.INTER_LINEAR)
            keep = cv2.resize(keep, size, interpolation=cv2.INTER_LINEAR)
        else:
            # 同分辨率时给 1px 柔化，避免锯齿
            erase = cv2.GaussianBlur(erase, (3, 3), 0.8)
            keep = cv2.GaussianBlur(keep, (3, 3), 0.8)
        return erase, keep

    def apply(self, alpha: np.ndarray) -> np.ndarray:
        """把涂抹叠加到 alpha 上，保持输入的类型与量纲。

        注意：始终返回**新数组**，不会把输入原样返回 ——
        调用方常把它当成可写的显示缓冲（复合结果），别名会写坏底图。
        """
        if self.is_empty and not self.layer.any():
            return alpha.copy()
        is_uint8 = alpha.dtype == np.uint8
        base = alpha.astype(np.float32)
        if is_uint8:
            base = base / 255.0
        erase, keep = self.masks(base.shape[:2])
        out = base * (1.0 - erase)
        out = np.maximum(out, keep)
        # 柔化仅作用在笔刷外缘；笔刷实际覆盖的像素必须完全擦除/补回。
        hard = self.layer
        if hard.shape != base.shape[:2]:
            hard = cv2.resize(
                hard, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        out[hard == ERASE] = 0.0
        out[hard == KEEP] = 1.0
        out = np.clip(out, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8) if is_uint8 else out

    def keep_region(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """只算一块的「保留」权重（0~1），用于原图标记的增量刷新。"""
        sub = (self.layer[y0:y1, x0:x1] == KEEP).astype(np.float32)
        return cv2.GaussianBlur(sub, (3, 3), 0.8)

    def apply_region(
        self, alpha: np.ndarray, x0: int, y0: int, x1: int, y1: int
    ) -> np.ndarray:
        """只计算 [y0:y1, x0:x1] 这一块（涂抹时增量刷新用），返回同尺寸子块。"""
        is_uint8 = alpha.dtype == np.uint8
        base = alpha[y0:y1, x0:x1].astype(np.float32)
        if is_uint8:
            base = base / 255.0

        layer = self.layer[y0:y1, x0:x1]
        erase = cv2.GaussianBlur((layer == ERASE).astype(np.float32), (3, 3), 0.8)
        keep = cv2.GaussianBlur((layer == KEEP).astype(np.float32), (3, 3), 0.8)
        out = np.clip(base * (1.0 - erase), 0.0, 1.0)
        out = np.maximum(out, keep)
        out[layer == ERASE] = 0.0
        out[layer == KEEP] = 1.0
        out = np.clip(out, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8) if is_uint8 else out

    def scaled(self, shape: tuple[int, int]) -> "MaskEdit":
        """换到另一个分辨率（例如重新处理用了不同的预览尺寸）。"""
        if tuple(shape) == self.shape:
            return self
        sy = float(shape[0]) / float(self.shape[0])
        sx = float(shape[1]) / float(self.shape[1])
        new = MaskEdit(shape)
        new._strokes = [s.scaled(sx, sy, (sx + sy) / 2.0) for s in self._strokes]
        if self._current is not None:
            new._strokes.append(self._current.scaled(sx, sy, (sx + sy) / 2.0))
        new._render()
        return new

    def foreground_mask(self, shape: Optional[tuple[int, int]] = None) -> np.ndarray:
        """把「保留」笔迹取成 0~1 的前景提示图（用于在原图上标记要抠的主体）。"""
        _, keep = self.masks(shape)
        return keep

    def strokes(self) -> Iterable[Stroke]:
        return tuple(self._strokes)
