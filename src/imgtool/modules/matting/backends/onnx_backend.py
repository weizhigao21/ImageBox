"""ONNX 抠图后端（推荐主路径）。

支持 u2net / isnet-general-use / rmbg-1.4 等常见抠图模型，
权重放在 ``models/`` 目录即可自动发现；也可从 UI/CLI 触发下载。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ....core.log import get_logger
from .base import MattingBackend
from .downloads import download_file

log = get_logger("onnx")

# 模型登记表：file / 输入边长 / 均值方差 / 下载地址
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "isnet-general-use": {
        "label": "ISNet 通用（推荐，质量好）",
        "file": "isnet-general-use.onnx",
        "size": 1024,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
    },
    "u2net": {
        "label": "U2Net（经典，速度快）",
        "file": "u2net.onnx",
        "size": 320,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
    },
    "u2netp": {
        "label": "U2Net-lite（最轻量）",
        "file": "u2netp.onnx",
        "size": 320,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
    },
    "rmbg-1.4": {
        "label": "RMBG-1.4（商品/物体，推荐）",
        "file": "rmbg-1.4.onnx",
        "size": 1024,
        "mean": (0.5, 0.5, 0.5),
        "std": (1.0, 1.0, 1.0),
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx",
    },
}

PREFERENCE = ["rmbg-1.4", "isnet-general-use", "u2net", "u2netp"]


def models_dir(config: Any = None) -> Path:
    if config is not None and hasattr(config, "models_dir"):
        return Path(config.models_dir)
    return Path("models")


def list_local_models(config: Any = None) -> list[str]:
    """返回 models/ 下已存在的 .onnx 文件名（含登记名与未知模型）。"""
    d = models_dir(config)
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob("*.onnx") if p.is_file())


def resolve_model_file(model: str, config: Any = None) -> Optional[Path]:
    """把模型名解析成实际权重路径；找不到返回 None。"""
    d = models_dir(config)
    if model in MODEL_REGISTRY:
        target = d / MODEL_REGISTRY[model]["file"]
        if target.exists():
            return target
    # 允许直接给文件名
    candidate = d / model
    if candidate.exists():
        return candidate
    # auto：按偏好顺序挑一个可用的
    if model in {"auto", "", None}:
        files = list_local_models(config)
        for name in PREFERENCE:
            fname = MODEL_REGISTRY[name]["file"]
            if fname in files:
                return d / fname
        if files:
            return d / files[0]
    return None


class OnnxBackend(MattingBackend):
    name = "onnx"
    label = "ONNX 模型"
    post_cleanup = False

    def __init__(self, params: dict | None = None, device: str = "auto") -> None:
        super().__init__(params, device)
        self.config = self.params.get("_config")
        self.model = str(self.params.get("onnx_model") or "auto")
        self.session = None
        self.input_name = "input"
        self.input_size = (320, 320)
        self.mean = np.array(MODEL_REGISTRY["u2net"]["mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(MODEL_REGISTRY["u2net"]["std"], dtype=np.float32).reshape(1, 1, 3)
        self.model_path: Optional[Path] = None

    # ---------- 可用性 ----------
    @classmethod
    def available(cls, config: Any = None) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401, PLC0415
        except Exception as exc:
            return False, f"未安装 onnxruntime（{exc}）"
        if not list_local_models(config):
            return False, f"models 目录下没有 .onnx 权重：{models_dir(config)}"
        return True, ""

    # ---------- 加载 ----------
    def load(self) -> None:
        if self._loaded and self.session is not None:
            return
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"未安装 onnxruntime：{exc}") from exc

        path = resolve_model_file(self.model, self.config)
        if path is None:
            raise FileNotFoundError(
                f"找不到 ONNX 权重（模型={self.model}），请把 .onnx 放到 {models_dir(self.config)}"
            )
        self.model_path = path

        from ....core.device import onnx_providers  # 避免循环导入

        providers = onnx_providers(self.device)
        log.info("加载 ONNX 模型 %s，providers=%s", path.name, providers)
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.active_providers = self.session.get_providers()

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = list(inp.shape)
        size = None
        if len(shape) == 4 and all(isinstance(s, int) for s in shape[2:]):
            size = (int(shape[2]), int(shape[3]))
        meta = None
        for key, cfg in MODEL_REGISTRY.items():
            if cfg["file"] == path.name:
                meta = cfg
                break
        if size is None or min(size) <= 0:
            size = (int((meta or {}).get("size", 1024)),) * 2
        self.input_size = (size[0], size[1])
        mean = (meta or {}).get("mean", MODEL_REGISTRY["u2net"]["mean"])
        std = (meta or {}).get("std", MODEL_REGISTRY["u2net"]["std"])
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self._loaded = True

    # ---------- 推理 ----------
    def predict(self, rgb: np.ndarray, hint: np.ndarray | None = None) -> np.ndarray:
        # hint 由模块层统一施加约束，这里忽略
        self.load()
        assert self.session is not None
        h, w = rgb.shape[:2]
        ih, iw = self.input_size

        interp = cv2.INTER_AREA if (ih < h or iw < w) else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (iw, ih), interpolation=interp)

        x = resized.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))[None, ...]
        x = np.ascontiguousarray(x, dtype=np.float32)

        outputs = self.session.run(None, {self.input_name: x})
        pred = np.asarray(outputs[0])
        pred = np.squeeze(pred)
        if pred.ndim == 3:  # 例如 u2net 的多尺度输出
            pred = pred[0]
        if pred.ndim != 2:
            raise RuntimeError(f"模型输出维度异常：{np.asarray(outputs[0]).shape}")

        lo, hi = float(pred.min()), float(pred.max())
        if hi - lo < 1e-6:
            pred = np.zeros_like(pred, dtype=np.float32)
        else:
            pred = (pred - lo) / (hi - lo)

        alpha = cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0.0, 1.0)

    def close(self) -> None:
        self.session = None
        super().close()


def download_model(model: str, config: Any = None, on_progress=None, cancel=None) -> Path:
    """下载指定模型的权重到 models/。"""
    if model not in MODEL_REGISTRY:
        raise KeyError(f"未知模型：{model}")
    cfg = MODEL_REGISTRY[model]
    dest = models_dir(config) / cfg["file"]
    url = cfg.get("url")
    if not url:
        raise ValueError(f"模型 {model} 未登记下载地址，请手动放置权重")
    return download_file(url, dest, on_progress=on_progress, cancel=cancel)
