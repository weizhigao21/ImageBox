"""调试：把 原图 / 粗略背景距离 / 最终 alpha / 真值 拼成一张对比图。

用法：python tests/debug_simple.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imgtool.core import io_utils  # noqa: E402
from imgtool.modules.matting.backends.simple_backend import SimpleBackend  # noqa: E402
from make_sample import make_sample  # noqa: E402


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    out = rgb.copy()
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[edge] = color
    return out


def main() -> int:
    rgb, gt = make_sample()
    backend = SimpleBackend({"max_size": 1600})

    small = cv2.resize(rgb, (800, 600), interpolation=cv2.INTER_AREA)
    rough = backend._background_distance_mask(small)
    alpha = backend.predict(rgb)
    final = (alpha > 0.5).astype(np.uint8) * 255

    panels = [
        overlay(rgb, gt * 255, (0, 255, 0)),
        overlay(rgb, rough, (255, 128, 0)),
        overlay(rgb, final, (255, 0, 0)),
        io_utils.alpha_to_rgb(alpha),
    ]
    labels = ["GT(绿)", "rough(橙)", "final(红)", "alpha"]
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    grid = np.vstack([top, bottom])

    out = Path(__file__).resolve().parent / "_tmp" / "debug_simple.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_image(out, grid)
    print("已输出", out)

    inter = np.logical_and(final > 0, gt > 0).sum()
    union = np.logical_or(final > 0, gt > 0).sum()
    print(f"final IoU = {inter / max(1, union):.3f}")
    inter = np.logical_and(rough > 0, gt > 0).sum()
    union = np.logical_or(rough > 0, gt > 0).sum()
    print(f"rough IoU = {inter / max(1, union):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
