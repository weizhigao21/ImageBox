"""临时实验：在杂乱背景样本上比较几种 GrabCut 初始化策略。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imgtool.core import io_utils  # noqa: E402
from make_sample import make_busy_sample  # noqa: E402


def iou(a, b):
    a = a > 0
    b = b > 0
    return float(np.logical_and(a, b).sum()) / float(max(1, np.logical_or(a, b).sum()))


def border_mask(shape, ratio=0.02):
    h, w = shape
    bw = max(3, int(round(min(h, w) * ratio)))
    m = np.zeros((h, w), np.uint8)
    m[:bw, :] = 255
    m[-bw:, :] = 255
    m[:, :bw] = 255
    m[:, -bw:] = 255
    return m


def grabcut(bgr, mask, mode, rect=(0, 0, 1, 1), iters=4):
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    m = mask.copy()
    cv2.grabCut(bgr, m, rect, bgd, fgd, iters, mode)
    return np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def main() -> int:
    rgb, gt = make_busy_sample()
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    small = cv2.resize(bgr, (w, h))
    border = border_mask((h, w), 0.02)
    margin = max(2, int(round(min(h, w) * 0.02)))
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    rows = []

    # A 当前做法：矩形初始化
    m = np.zeros((h, w), np.uint8)
    out = grabcut(small, m, cv2.GC_INIT_WITH_RECT, rect, 4)
    rows.append(("A rect x4", out))

    # B 中心椭圆先验
    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (int(w * 0.3), int(h * 0.36)), 0, 0, 360, cv2.GC_PR_FGD, -1)
    m[border > 0] = cv2.GC_BGD
    out = grabcut(small, m, cv2.GC_INIT_WITH_MASK, (0, 0, 1, 1), 4)
    rows.append(("B center-ellipse x4", out))

    # C 矩形 + 更多迭代
    m = np.zeros((h, w), np.uint8)
    out = grabcut(small, m, cv2.GC_INIT_WITH_RECT, rect, 10)
    rows.append(("C rect x10", out))

    # D 边缘 k-means 背景距离作为先验
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    samples = lab[border_mask((h, w), 0.06) > 0]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(samples, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    dist = None
    for c in np.asarray(centers).reshape(-1, 3):
        d = np.linalg.norm(lab - c.reshape(1, 1, 3), axis=2)
        dist = d if dist is None else np.minimum(dist, d)
    norm = np.clip(dist / max(1e-3, float(np.percentile(dist, 99))) * 255, 0, 255).astype(np.uint8)
    _, rough = cv2.threshold(cv2.GaussianBlur(norm, (0, 0), 1.5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, k, 1)
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, k, 2)
    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    m[rough > 0] = cv2.GC_PR_FGD
    m[border > 0] = cv2.GC_BGD
    out = grabcut(small, m, cv2.GC_INIT_WITH_MASK, (0, 0, 1, 1), 4)
    rows.append(("D border-kmeans x4", out))

    # E 矩形跑一遍，再用结果做掩码跑第二遍
    m = np.zeros((h, w), np.uint8)
    first = grabcut(small, m, cv2.GC_INIT_WITH_RECT, rect, 4)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    m[cv2.erode(first, k) > 0] = cv2.GC_FGD
    m[cv2.dilate(first, k) == 0] = cv2.GC_BGD
    out = grabcut(small, m, cv2.GC_INIT_WITH_MASK, (0, 0, 1, 1), 4)
    rows.append(("E two-pass x4", out))

    for name, out in rows:
        print(f"{name:<24} IoU={iou(out, gt):.3f}  fg={np.count_nonzero(out) / out.size:.1%}")

    # 存一张对比图
    panels = []
    for name, out in rows:
        vis = rgb.copy()
        edge = cv2.morphologyEx(out, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        vis[edge] = (255, 0, 0)
        cv2.putText(vis, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        panels.append(vis)
    grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:] + [np.zeros_like(panels[0])])])
    dst = Path(__file__).resolve().parent / "_tmp" / "debug_busy.png"
    io_utils.write_image(dst, grid)
    print("已输出", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
