"""图像读写工具。

要点：Windows 下 ``cv2.imread`` 无法处理中文路径，
这里统一走 ``np.fromfile + cv2.imdecode`` / ``cv2.imencode + tofile``。
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


# ---------------------------------------------------------------- 读
def read_image(path: str | Path) -> np.ndarray:
    """读取为 HxWx3 uint8 RGB。"""
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"文件为空或不可读：{p}")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法解码图片：{p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_image_rgba(path: str | Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """读取为 (RGB, alpha 或 None)。"""
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"文件为空或不可读：{p}")
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法解码图片：{p}")
    if img.ndim == 3 and img.shape[2] == 4:
        bgra = img
        rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
        return rgb, bgra[:, :, 3].copy()
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return rgb, None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None


def probe_size(path: str | Path) -> tuple[int, int]:
    """快速读尺寸（只读文件头），失败返回 (0, 0)。"""
    p = Path(path)
    try:
        from PIL import Image  # 延迟导入，PIL 只用于读取头部信息

        with Image.open(p) as img:
            return int(img.width), int(img.height)
    except Exception:
        pass
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_8)
        if img is None:
            return 0, 0
        return int(img.shape[1] * 8), int(img.shape[0] * 8)
    except Exception:
        return 0, 0


# ---------------------------------------------------------------- 写
def write_image(path: str | Path, image: np.ndarray) -> Path:
    """按扩展名写出。image 可为 RGB / RGBA / 单通道。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        out = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        out = arr

    ext = p.suffix.lower() or ".png"
    if ext in {".jpg", ".jpeg"} and out.ndim == 3 and out.shape[2] == 4:
        out = out[:, :, :3]  # JPEG 不支持透明
    ok, buf = cv2.imencode(ext, out)
    if not ok:
        raise ValueError(f"无法编码为 {ext}：{p}")
    buf.tofile(str(p))
    return p


def write_jpg(path: str | Path, image: np.ndarray, quality: int = 92) -> Path:
    """按指定质量写 JPG。image 为 RGB / RGBA（透明通道会被丢弃）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError(f"无法编码为 jpg：{p}")
    buf.tofile(str(p))
    return p


# ---------------------------------------------------------------- 合成/转换
def parse_color(value: str | Sequence[int], default: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    """'#ffffff' / '255,255,255' -> (r, g, b)。"""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return int(value[0]), int(value[1]), int(value[2])
    text = str(value or "").strip().lstrip("#")
    try:
        if len(text) == 6:
            return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
        if len(text) == 3:
            return tuple(int(c * 2, 16) for c in text)  # type: ignore[return-value]
        parts = [int(x) for x in text.split(",")[:3]]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]  # type: ignore[return-value]
    except ValueError:
        pass
    return default


def color_to_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(c))) for c in color])


def to_rgba(rgb: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    """拼成 RGBA uint8。"""
    if alpha is None:
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    if alpha.shape[:2] != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    a = alpha if alpha.dtype == np.uint8 else np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([rgb, a])


def flatten(rgb: np.ndarray, alpha: Optional[np.ndarray], bg: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """把带透明的图压到纯色背景上，返回 RGB uint8。"""
    rgba = to_rgba(rgb, alpha).astype(np.float32)
    a = (rgba[:, :, 3:4] / 255.0)
    bg_arr = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
    out = rgba[:, :, :3] * a + bg_arr * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def alpha_to_rgb(alpha: np.ndarray) -> np.ndarray:
    """遮罩转灰度 RGB，便于预览。"""
    a = alpha
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([a, a, a])


# ---------------------------------------------------------------- 目录/命名
def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def list_images(paths: Iterable[str | Path], recursive: bool = True) -> list[Path]:
    """把文件/目录混合列表展开成图片文件列表（去重、稳定排序）。"""
    found: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = str(p).lower()
        if key not in seen and is_image(p):
            seen.add(key)
            found.append(p)

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for child in sorted(it):
                if child.is_file():
                    add(child)
        elif p.is_file():
            add(p)
    return found


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(base: str | Path, module_name: str) -> Path:
    """output/<模块>/<时间戳>/"""
    d = Path(base) / module_name / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_output_path(
    out_dir: str | Path,
    src: str | Path,
    name_template: str = "{stem}_cutout",
    ext: str = ".png",
    overwrite: bool = False,
) -> Path:
    """按模板生成输出路径；不覆盖时自动加 _1 / _2 后缀。"""
    src_path = Path(src)
    stem = src_path.stem
    try:
        name = name_template.format(stem=stem, name=src_path.name, ext=ext.lstrip("."))
    except (KeyError, IndexError):
        name = f"{stem}_cutout"
    if not name:
        name = f"{stem}_cutout"
    ext = ext if ext.startswith(".") else f".{ext}"
    if not name.lower().endswith(ext.lower()):
        name = f"{name}{ext}"

    out_dir = Path(out_dir)
    target = out_dir / name
    if overwrite or not target.exists():
        return target
    i = 1
    while True:
        candidate = out_dir / f"{target.stem}_{i}{target.suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def open_in_explorer(path: str | Path) -> bool:
    """在文件管理器中打开目录/定位文件。"""
    p = Path(path)
    if not p.exists():
        return False
    try:
        if sys.platform.startswith("win"):
            if p.is_file():
                subprocess.Popen(["explorer", "/select,", str(p)])
            else:
                os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p if p.is_dir() else p.parent)])
        else:
            subprocess.Popen(["xdg-open", str(p if p.is_dir() else p.parent)])
        return True
    except Exception:
        return False
