"""抠图后处理：掩码清理、羽化、边界收缩/扩张、去色边。"""

from __future__ import annotations

import cv2
import numpy as np


def to_float(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    return np.clip(a.astype(np.float32), 0.0, 1.0)


def cleanup_mask(alpha: np.ndarray, min_area_ratio: float = 0.002) -> np.ndarray:
    """清理“脏掩码”：去噪点、保留主体连通域、填内部空洞。"""
    a = to_float(alpha)
    binary = np.where(a >= 0.5, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num > 1:
        h, w = binary.shape[:2]
        min_area = max(64, int(h * w * float(min_area_ratio)))
        keep = np.zeros_like(binary)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 255
        if keep.any():
            binary = keep

    # 填内部空洞：从角落泛洪背景，剩下的非背景即空洞
    h, w = binary.shape[:2]
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    binary = cv2.bitwise_or(binary, holes)

    return binary.astype(np.float32) / 255.0


def refine_alpha(
    alpha: np.ndarray,
    rgb: np.ndarray | None = None,
    threshold: float = 8.0,
    expand: float = 0.0,
    feather: float = 1.5,
    gamma: float = 1.0,
) -> np.ndarray:
    """按参数细化 alpha。

    threshold : 0~255，低于该值的半透明被视为全透明（提升对比度）
    expand    : 正=外扩（多留边缘），负=内缩（去白边），单位像素
    feather   : 高斯羽化 sigma，0 表示不羽化
    gamma     : >1 更“实”，<1 更“透”
    """
    a = to_float(alpha)
    if rgb is not None and a.shape[:2] != rgb.shape[:2]:
        a = cv2.resize(a, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    t = float(np.clip(threshold, 0.0, 254.0)) / 255.0
    if t > 0:
        a = np.clip((a - t) / max(1e-6, 1.0 - t), 0.0, 1.0)

    if abs(float(expand)) >= 0.5:
        r = int(round(abs(float(expand))))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        b = np.where(a >= 0.5, 255, 0).astype(np.uint8)
        b = cv2.dilate(b, k) if expand > 0 else cv2.erode(b, k)
        a = b.astype(np.float32) / 255.0

    if float(feather) > 0.05:
        sigma = float(feather)
        ksize = int(max(3, round(sigma * 4))) | 1
        a = cv2.GaussianBlur(a, (ksize, ksize), sigma)

    if abs(float(gamma) - 1.0) > 1e-3:
        a = np.power(np.clip(a, 0.0, 1.0), float(gamma))

    return np.clip(a, 0.0, 1.0)


def decontaminate(
    rgb: np.ndarray,
    alpha: np.ndarray,
    strength: float = 1.0,
    radius: int = 8,
) -> np.ndarray:
    """去色边：用背景估计反解前景色，消除边缘残留的背景颜色。

    F = (I - (1-a) * B) / a，其中 B 为背景色估计（由 a≈0 的像素加权平滑得到）。
    """
    if strength <= 0:
        return rgb
    a = to_float(alpha)
    if a.shape[:2] != rgb.shape[:2]:
        a = cv2.resize(a, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    a3 = a[:, :, None]

    k = int(max(3, radius * 2 + 1)) | 1
    weight = (1.0 - a3) ** 2 + 1e-3
    weight3 = np.repeat(weight, 3, axis=2)
    src = rgb.astype(np.float32)
    num = cv2.boxFilter(src * weight3, -1, (k, k), normalize=False)
    den = cv2.boxFilter(weight3, -1, (k, k), normalize=False)
    bg = num / np.maximum(den, 1e-6)

    fg = (src - (1.0 - a3) * bg) / np.maximum(a3, 0.15)
    fg = np.clip(fg, 0.0, 255.0)

    # 只在半透明边缘混合，前景内部颜色保持不变
    edge = np.clip((1.0 - np.abs(a3 * 2.0 - 1.0)) * float(strength), 0.0, 1.0)
    out = src * (1.0 - edge) + fg * edge
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def apply_foreground_hint(
    alpha: np.ndarray,
    hint: np.ndarray,
    expand: float = 12.0,
    feather: float = 6.0,
) -> np.ndarray:
    """把用户在原图上涂抹的范围作为「只在这里面抠」的约束。

    hint : HxW，0~1 或 0/255，非 0 表示用户标记的主体范围
    expand : 先把标记范围外扩这么多像素（手涂的边界通常比真实边缘小）
    feather: 过渡带柔化，避免切出一条生硬的直边
    """
    a = to_float(alpha)
    h = to_float(hint)
    if h.shape[:2] != a.shape[:2]:
        h = cv2.resize(h, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    if float(expand) >= 0.5:
        r = int(round(float(expand)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        h = cv2.dilate(h, k)
    if float(feather) > 0.05:
        h = cv2.GaussianBlur(h, (0, 0), float(feather))
    h = np.clip(h, 0.0, 1.0)
    return np.clip(a * h, 0.0, 1.0)


def apply_hint_subject(
    alpha: np.ndarray,
    hint: np.ndarray,
    expand: float = 8.0,
    feather: float = 3.0,
) -> np.ndarray:
    """只保留与涂抹区域相连的前景成分：涂在主体上任意一处，就抠出整只主体。

    与 :func:`apply_foreground_hint` 的区别：这里不做矩形式的范围裁剪，
    而是「按连通性挑出被涂到的那只主体」，所以结果是主体自身的轮廓。
    """
    a = to_float(alpha)
    h = to_float(hint)
    if h.shape[:2] != a.shape[:2]:
        h = cv2.resize(h, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 模型未把笔迹识别为前景时，也要让用户明确标记的区域真正补回来。
    # 选连通主体仍用于剔除未标记的其它物体；笔迹本身始终是确定前景。
    binary = np.where(a >= 0.5, 255, 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(binary, 8)
    if num <= 1:
        return np.maximum(a, h)

    r = max(2, int(round(float(expand))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    seeds_area = cv2.dilate(np.where(h > 0.2, 255, 0).astype(np.uint8), k)
    seeds = {int(v) for v in np.unique(labels[seeds_area > 0]) if int(v) != 0}
    if not seeds:
        return np.maximum(a, h)

    keep = np.isin(labels, list(seeds)).astype(np.uint8) * 255
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if float(feather) > 0.05:
        keep_f = cv2.GaussianBlur(keep.astype(np.float32) / 255.0, (0, 0), float(feather))
    else:
        keep_f = keep.astype(np.float32) / 255.0
    return np.maximum(
        np.clip(a * np.clip(keep_f, 0.0, 1.0), 0.0, 1.0), h
    )


def outline(rgb: np.ndarray, alpha: np.ndarray, width: int = 3, color=(255, 255, 255)) -> tuple[np.ndarray, np.ndarray]:
    """给主体加描边（返回新的 rgb/alpha，画布不变）。"""
    if width <= 0:
        return rgb, alpha
    a = to_float(alpha)
    b = np.where(a >= 0.5, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    outer = cv2.dilate(b, k)
    ring = cv2.subtract(outer, b)

    out_rgb = rgb.copy()
    out_rgb[ring > 0] = np.array(color, dtype=np.uint8)
    out_alpha = np.where(ring > 0, 255, (a * 255).astype(np.uint8)).astype(np.uint8)
    return out_rgb, out_alpha
