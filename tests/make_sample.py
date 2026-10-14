"""合成测试图片：一个彩色物体 + 渐变背景，用于验证抠图链路。

用法：python tests/make_sample.py [输出路径]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imgtool.core import io_utils  # noqa: E402


def make_sample(width: int = 800, height: int = 600, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """返回 (RGB 图, 真实前景掩码 0/1)。"""
    rng = np.random.default_rng(seed)

    # 渐变背景（带轻微噪点，模拟真实照片）
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    bg = np.zeros((height, width, 3), np.float32)
    bg[:, :, 0] = 60 + 120 * (xx / width)
    bg[:, :, 1] = 90 + 90 * (yy / height)
    bg[:, :, 2] = 150 - 60 * (xx / width)
    bg += rng.normal(0, 4, bg.shape)
    img = np.clip(bg, 0, 255)

    mask = np.zeros((height, width), np.uint8)
    center = (int(width * 0.5), int(height * 0.52))
    axes = (int(width * 0.22), int(height * 0.32))
    cv2.ellipse(mask, center, axes, 12, 0, 360, 255, -1)

    # 物体本体：暖色 + 内部渐变（避免纯色让算法“作弊”）
    obj = np.zeros_like(img)
    obj[:, :, 0] = 235 - 40 * (yy / height)
    obj[:, :, 1] = 140 + 60 * (xx / width)
    obj[:, :, 2] = 70 + 30 * (yy / height)
    obj += rng.normal(0, 6, obj.shape)

    sel = mask > 0
    img[sel] = np.clip(obj[sel], 0, 255)

    # 一点软阴影，考验边缘处理
    shadow = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 18)
    img *= (1.0 - 0.35 * shadow)[:, :, None]

    rgb = np.clip(img, 0, 255).astype(np.uint8)
    return rgb, (mask > 0).astype(np.uint8)


def make_busy_sample(width: int = 720, height: int = 540, seed: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """杂乱背景（多项式拟合不可信）的样本，用于验证矩形初始化兜底路径。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 190, (height, width, 3)).astype(np.float32)

    # 大块色斑 + 条纹，破坏背景的“光滑性”
    for _ in range(24):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (int(rng.integers(20, 120)), int(rng.integers(20, 120)))
        color = tuple(int(c) for c in rng.integers(40, 220, 3))
        cv2.ellipse(img, center, axes, float(rng.integers(0, 180)), 0, 360, color, -1)
    img = cv2.GaussianBlur(img, (0, 0), 3.0)
    img += rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255)

    mask = np.zeros((height, width), np.uint8)
    center = (int(width * 0.5), int(height * 0.5))
    axes = (int(width * 0.24), int(height * 0.32))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    obj = np.zeros_like(img)
    obj[:, :, 0] = 240
    obj[:, :, 1] = 250
    obj[:, :, 2] = 235
    obj += rng.normal(0, 4, obj.shape)
    sel = mask > 0
    img[sel] = np.clip(obj[sel], 0, 255)

    rgb = np.clip(img, 0, 255).astype(np.uint8)
    return rgb, (mask > 0).astype(np.uint8)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "_tmp" / "sample.png"
    rgb, mask = make_sample()
    out.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_image(out, rgb)
    io_utils.write_image(out.with_name(out.stem + "_gt.png"), mask * 255)
    print(f"已生成 {out}  ({rgb.shape[1]}x{rgb.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
