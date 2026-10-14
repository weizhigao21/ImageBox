"""BiRefNet 后端（可选，质量最好但吃显存）。

依赖 transformers + torch，首次使用需要联网下载权重。
任何依赖/显存问题都会抛出 BackendUnavailable，由上层给出提示并建议改用 onnx / simple。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ....core.base import BackendUnavailable
from ....core.log import get_logger
from .base import MattingBackend

log = get_logger("birefnet")

DEFAULT_REPO = "ZhengPeng7/BiRefNet"


class BiRefNetBackend(MattingBackend):
    name = "birefnet"
    label = "BiRefNet（实验，高质量）"
    post_cleanup = False

    def __init__(self, params: dict | None = None, device: str = "auto") -> None:
        super().__init__(params, device)
        self.repo = str(self.params.get("birefnet_repo") or DEFAULT_REPO)
        self.size = int(self.params.get("birefnet_size") or 1024)
        self.model = None
        self.torch = None

    @classmethod
    def available(cls, config: Any = None) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401, PLC0415
        except Exception as exc:
            return False, f"未安装 torch（{exc}）"
        try:
            import transformers  # noqa: F401, PLC0415
        except Exception as exc:
            return False, f"未安装 transformers（{exc}）"
        return True, "首次运行需联网下载权重"

    def load(self) -> None:
        if self._loaded and self.model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForImageSegmentation  # noqa: PLC0415
        except Exception as exc:
            raise BackendUnavailable(f"BiRefNet 依赖缺失：{exc}") from exc

        from ....core.device import resolve_device  # noqa: PLC0415

        device = resolve_device(self.device)
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                self.repo, trust_remote_code=True
            )
        except Exception as exc:
            raise BackendUnavailable(
                f"加载 BiRefNet 失败（{self.repo}）：{exc}；可改用 ONNX 或简易后端"
            ) from exc

        model.eval()
        if device == "cuda":
            try:
                model = model.to("cuda")
                # 半精度显存更友好；如果算子不支持再退回 fp32
                try:
                    model = model.half()
                    self._half = True
                except Exception:
                    self._half = False
            except Exception as exc:
                log.warning("BiRefNet 迁移到 GPU 失败，改用 CPU：%s", exc)
                device = "cpu"
                self._half = False
        else:
            self._half = False

        self.torch = torch
        self.device = device
        self.model = model
        self._loaded = True

    def predict(self, rgb: np.ndarray, hint: np.ndarray | None = None) -> np.ndarray:
        # hint 由模块层统一施加约束，这里忽略
        self.load()
        assert self.model is not None and self.torch is not None
        torch = self.torch

        h, w = rgb.shape[:2]
        size = self.size
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std
        tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))[None, ...].copy())
        if getattr(self, "_half", False):
            tensor = tensor.half()
        tensor = tensor.to(self.device)

        with torch.no_grad():
            try:
                outputs = self.model(tensor)
            except Exception as exc:
                raise BackendUnavailable(f"BiRefNet 推理失败：{exc}") from exc

        if isinstance(outputs, (list, tuple)):
            logits = outputs[-1]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        pred = torch.sigmoid(logits[-1][0]).float().cpu().numpy()
        if pred.ndim == 3:
            pred = pred[0]

        alpha = cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0.0, 1.0)

    def close(self) -> None:
        self.model = None
        try:
            if self.torch is not None and self.device == "cuda":
                self.torch.cuda.empty_cache()
        except Exception:
            pass
        super().close()
