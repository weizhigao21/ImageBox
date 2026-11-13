"""超分后端：Real-ESRGAN 的 ONNX / Torch 两种加载方式 + 分块推理。

ONNX（推荐）：动态 H/W 的 fp32 导出，靠 onnxruntime，装完依赖即可用；
Torch：直接加载官方 GitHub Release 的 .pth 权重，按 state dict 自动推断网络结构。

大图一次性 4x 前向非常吃显存，统一走「分块 + 重叠填充」推理：
每个输入块向外扩 ``tile_pad`` 像素取上下文，前向后只取中心 4x 区域拼回，
与官方 realesrgan 推理脚本行为一致。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ...core.log import get_logger

log = get_logger("superres")

# 模型登记表：官方 .pth 与验证过的 ONNX 导出地址
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "x4plus": {
        "label": "RealESRGAN x4plus（通用照片）",
        "scale": 4,
        "torch_file": "RealESRGAN_x4plus.pth",
        "torch_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "onnx_file": "realesrgan-x4plus.onnx",
        "onnx_url": "https://huggingface.co/jonathanst29/tinier-upscale-models/resolve/main/realesrgan-x4plus.onnx",
        "arch": {"num_block": 23, "num_feat": 64, "num_grow_ch": 32},
    },
    "x4plus-anime": {
        "label": "RealESRGAN x4plus 动漫（6B，轻量）",
        "scale": 4,
        "torch_file": "RealESRGAN_x4plus_anime_6B.pth",
        "torch_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "arch": {"num_block": 6, "num_feat": 64, "num_grow_ch": 32},
    },
}

BACKEND_CHOICES = [
    ("onnx", "ONNX（推荐，轻量）"),
    ("torch", "Torch（官方 .pth）"),
]


def models_dir(config: Any = None) -> Path:
    if config is not None and hasattr(config, "models_dir"):
        return Path(config.models_dir)
    return Path("models")


def list_local_files(config: Any = None, suffix: str = ".onnx") -> list[str]:
    d = models_dir(config)
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob(f"*{suffix}") if p.is_file())


def resolve_weight(model: str, kind: str, config: Any = None) -> Optional[Path]:
    """把「模型名或本地文件名」解析成实际权重路径；找不到返回 None。"""
    d = models_dir(config)
    if model in MODEL_REGISTRY:
        fname = MODEL_REGISTRY[model].get(f"{kind}_file")
        if fname:
            candidate = d / fname
            if candidate.exists():
                return candidate
        return None
    # 允许直接给文件名（.onnx / .pth）
    if str(model).lower().endswith((".onnx", ".pth")):
        candidate = d / model
        if candidate.exists():
            return candidate
    return None


def download_weight(
    model: str,
    kind: str,
    config: Any = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> Path:
    """下载指定模型的权重到 models/（kind: onnx | torch）。"""
    if model not in MODEL_REGISTRY:
        raise KeyError(f"未知模型：{model}")
    cfg = MODEL_REGISTRY[model]
    url = cfg.get(f"{kind}_url")
    fname = cfg.get(f"{kind}_file")
    if not url or not fname:
        raise ValueError(f"模型 {model} 没有登记 {kind} 版下载地址，请手动放置权重")
    from ..matting.backends.downloads import download_file

    return download_file(url, models_dir(config) / fname, on_progress=on_progress, cancel=cancel)


# ---------------------------------------------------------------- 分块推理
def tiled_forward(
    rgb: np.ndarray,
    forward: Callable[[np.ndarray], np.ndarray],
    scale: int,
    tile: int = 256,
    tile_pad: int = 10,
    on_progress: Optional[Callable[[int, int], None]] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    """把 HxWx3 float32（0~1）RGB 分块送入 forward，拼出 (4H)x(4W)x3。

    forward 输入/输出均为 HxWx3 float32（0~1），输出边长须为输入的 scale 倍。
    tile<=0 或整图不超过 tile 时整图一次前向。
    """
    h, w = rgb.shape[:2]
    if tile <= 0 or (h <= tile and w <= tile):
        out = forward(rgb)
        return np.clip(out, 0.0, 1.0)

    out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    tiles = [(y, x) for y in range(0, h, tile) for x in range(0, w, tile)]
    total = len(tiles)
    for idx, (y, x) in enumerate(tiles):
        if check_cancelled is not None:
            check_cancelled()
        y_end = min(y + tile, h)
        x_end = min(x + tile, w)
        # 上下文窗口：向四周扩 tile_pad，但不出图边界
        y0 = max(0, y - tile_pad)
        x0 = max(0, x - tile_pad)
        y1 = min(h, y_end + tile_pad)
        x1 = min(w, x_end + tile_pad)
        patch = forward(rgb[y0:y1, x0:x1])
        # 只保留中心区域（对应无 pad 的原块），避免相邻块接缝
        oy = (y - y0) * scale
        ox = (x - x0) * scale
        oh = (y_end - y) * scale
        ow = (x_end - x) * scale
        out[y * scale : y * scale + oh, x * scale : x * scale + ow] = patch[
            oy : oy + oh, ox : ox + ow
        ]
        if on_progress is not None:
            on_progress(idx + 1, total)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------- 后端基类
class SuperresBackend(ABC):
    """超分后端：输入 HxWx3 uint8 RGB，输出 (4H)x(4W)x3 uint8 RGB。"""

    name = "base"
    label = "base"

    def __init__(self, params: dict[str, Any] | None = None, device: str = "auto") -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.device = device
        self._loaded = False

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def _forward(self, rgb_f32: np.ndarray) -> np.ndarray: ...

    def predict(
        self,
        rgb: np.ndarray,
        tile: int = 256,
        tile_pad: int = 10,
        on_progress: Optional[Callable[[int, int], None]] = None,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> np.ndarray:
        self.load()
        f32 = rgb.astype(np.float32) / 255.0
        out = tiled_forward(f32, self._forward, self.scale, tile, tile_pad, on_progress, check_cancelled)
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)

    scale: int = 4

    def close(self) -> None:
        self._loaded = False


# ---------------------------------------------------------------- ONNX 后端
class OnnxSuperresBackend(SuperresBackend):
    name = "onnx"
    label = "ONNX Real-ESRGAN"

    def __init__(self, params: dict[str, Any] | None = None, device: str = "auto") -> None:
        super().__init__(params, device)
        self.config = self.params.get("_config")
        self.model = str(self.params.get("model") or "x4plus")
        self.weight_path: Optional[Path] = None
        self.session = None
        self.input_name = "input"
        self.static_size: Optional[tuple[int, int]] = None  # 固定输入尺寸的模型需缩放适配

    @classmethod
    def available(cls, config: Any = None) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401, PLC0415
        except Exception as exc:
            return False, f"未安装 onnxruntime（{exc}）"
        return True, ""

    def load(self) -> None:
        if self._loaded and self.session is not None:
            return
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"未安装 onnxruntime：{exc}") from exc

        path = resolve_weight(self.model, "onnx", self.config)
        if path is None:
            raise FileNotFoundError(
                f"找不到 ONNX 超分权重（模型={self.model}），请点「下载」或把 .onnx 放到 {models_dir(self.config)}"
            )
        self.weight_path = path

        from ...core.device import onnx_providers

        providers = onnx_providers(self.device)
        log.info("加载 ONNX 超分模型 %s，providers=%s", path.name, providers)
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.active_providers = self.session.get_providers()

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = list(inp.shape)
        if len(shape) == 4 and all(isinstance(s, int) and s > 0 for s in shape[2:]):
            self.static_size = (int(shape[2]), int(shape[3]))
        self._loaded = True

    def _forward(self, rgb_f32: np.ndarray) -> np.ndarray:
        assert self.session is not None
        h, w = rgb_f32.shape[:2]
        x = np.transpose(rgb_f32, (2, 0, 1))[None, ...]
        x = np.ascontiguousarray(x, dtype=np.float32)

        if self.static_size is not None:
            # 固定输入尺寸的导出：缩放块到该尺寸，输出再缩回目标尺寸
            ih, iw = self.static_size
            import cv2

            x = cv2.resize(rgb_f32, (iw, ih), interpolation=cv2.INTER_AREA)
            x = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None, ...], dtype=np.float32)

        pred = np.asarray(self.session.run(None, {self.input_name: x})[0])
        pred = np.squeeze(pred)
        if pred.ndim != 3:
            raise RuntimeError(f"超分模型输出维度异常：{np.asarray(pred).shape}")
        out = np.transpose(pred, (1, 2, 0))

        if self.static_size is not None:
            import cv2

            out = cv2.resize(out, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC)
        return out

    def close(self) -> None:
        self.session = None
        super().close()


# ---------------------------------------------------------------- Torch 后端
def _build_rrdb(num_block: int, num_feat: int, num_grow_ch: int, scale: int):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, feat: int, grow: int) -> None:
            super().__init__()
            self.convs = nn.ModuleList(
                [nn.Conv2d(feat + i * grow, grow, 3, 1, 1) for i in range(5)]
            )
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = x
            outs = []
            for conv in self.convs:
                out = self.lrelu(conv(feat))
                outs.append(out)
                feat = torch.cat([feat, out], dim=1)
            return outs[-1] * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, feat: int, grow: int) -> None:
            super().__init__()
            self.rdb1 = ResidualDenseBlock(feat, grow)
            self.rdb2 = ResidualDenseBlock(feat, grow)
            self.rdb3 = ResidualDenseBlock(feat, grow)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet()


def _infer_arch_from_state_dict(sd: dict) -> dict[str, int]:
    """从官方 .pth 的键名推断 RRDB 结构参数，兼容任意 block 数的本地权重。"""
    num_feat = 64
    for key in ("conv_first.weight", "conv_body.weight"):
        if key in sd:
            num_feat = int(sd[key].shape[0])
            break
    blocks = [int(k.split(".")[1]) for k in sd if k.startswith("body.") and k.endswith(".rdb1.conv1.weight")]
    num_block = max(blocks) + 1 if blocks else 23
    return {"num_block": num_block, "num_feat": num_feat, "num_grow_ch": 32}


class TorchSuperresBackend(SuperresBackend):
    name = "torch"
    label = "Torch Real-ESRGAN"

    def __init__(self, params: dict[str, Any] | None = None, device: str = "auto") -> None:
        super().__init__(params, device)
        self.config = self.params.get("_config")
        self.model = str(self.params.get("model") or "x4plus")
        self.half = bool(self.params.get("half", True))
        self.weight_path: Optional[Path] = None
        self.net = None

    @classmethod
    def available(cls, config: Any = None) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401, PLC0415
        except Exception as exc:
            return False, f"未安装 torch（{exc}）"
        return True, ""

    def load(self) -> None:
        if self._loaded and self.net is not None:
            return
        try:
            import torch  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"未安装 torch：{exc}") from exc

        path = resolve_weight(self.model, "torch", self.config)
        if path is None:
            raise FileNotFoundError(
                f"找不到超分权重（模型={self.model}），请点「下载」或把 .pth 放到 {models_dir(self.config)}"
            )
        self.weight_path = path

        from ...core.device import resolve_device

        dev = resolve_device(self.device)
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        arch = MODEL_REGISTRY.get(self.model, {}).get("arch") or _infer_arch_from_state_dict(sd)
        net = _build_rrdb(
            arch["num_block"], arch["num_feat"], arch.get("num_grow_ch", 32), scale=4
        )
        missing, unexpected = net.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if "params_ema" not in k]
        if real_missing:
            raise RuntimeError(f"权重与网络结构不匹配，缺少：{real_missing[:5]}")
        net.eval()
        if dev == "cuda" and self.half:
            net.half()
        net.to(dev)
        self.net = net
        self.device = dev
        self._loaded = True

    def _forward(self, rgb_f32: np.ndarray) -> np.ndarray:
        import torch  # noqa: PLC0415

        assert self.net is not None
        x = torch.from_numpy(np.transpose(rgb_f32, (2, 0, 1))[None, ...])
        x = x.to(self.device, dtype=next(self.net.parameters()).dtype)
        with torch.no_grad():
            pred = self.net(x)
        out = pred.squeeze(0).float().cpu().numpy()
        if out.ndim != 3:
            raise RuntimeError(f"超分模型输出维度异常：{out.shape}")
        return np.transpose(out, (1, 2, 0))

    def close(self) -> None:
        self.net = None
        super().close()


def create_backend(kind: str, params: dict[str, Any] | None = None, device: str = "auto"):
    if kind == "onnx":
        return OnnxSuperresBackend(params, device)
    if kind == "torch":
        return TorchSuperresBackend(params, device)
    raise KeyError(f"未知超分后端：{kind}")
