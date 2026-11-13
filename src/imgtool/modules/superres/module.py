"""超分模块：Real-ESRGAN 放大 4 倍。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...core import io_utils
from ...core.base import BackendUnavailable, BaseModule, ParamSpec
from ...core.log import get_logger
from ...core.types import ImageItem, ModuleResult, TaskContext
from .backend import (
    BACKEND_CHOICES,
    MODEL_REGISTRY,
    SuperresBackend,
    create_backend,
    list_local_files,
    resolve_weight,
)

log = get_logger("superres")

# 单张结果超过该像素数时不再往内存塞 result_rgb（预览退回原图），避免爆内存
PREVIEW_PIXEL_CAP = 16_000_000


class SuperresModule(BaseModule):
    name = "superres"
    title = "超分"
    icon = "⤢"
    description = "用 Real-ESRGAN 把图片放大 4 倍并补充细节（ONNX / Torch）。"
    order = 20

    def __init__(self) -> None:
        self._backend: Optional[SuperresBackend] = None
        self._backend_key: Optional[tuple] = None
        self.config: Any = None

    # ------------------------------------------------------------ 参数
    def _model_choices(self) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        for key, cfg in MODEL_REGISTRY.items():
            choices.append((key, cfg["label"]))
        for fname in list_local_files(self.config, ".onnx"):
            if fname not in {cfg.get("onnx_file") for cfg in MODEL_REGISTRY.values()}:
                choices.append((fname, f"本地 ONNX 权重：{fname}"))
        for fname in list_local_files(self.config, ".pth"):
            if fname not in {cfg.get("torch_file") for cfg in MODEL_REGISTRY.values()}:
                choices.append((fname, f"本地 Torch 权重：{fname}"))
        return choices

    def _default_backend(self) -> str:
        # 本地已有 ONNX 权重 → onnx；否则装了 torch → torch；都退 onnx（提示下载）
        if resolve_weight("x4plus", "onnx", self.config) is not None:
            return "onnx"
        try:
            import torch  # noqa: F401, PLC0415

            return "torch"
        except Exception:
            return "onnx"

    def param_spec(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                "backend", "超分后端", "choice", self._default_backend(),
                choices=BACKEND_CHOICES, group="基础",
                tooltip="ONNX 依赖 onnxruntime；Torch 直接加载官方 .pth（需要安装 torch）",
            ),
            ParamSpec(
                "model", "模型权重", "choice", "x4plus",
                choices=self._model_choices(), group="基础",
                tooltip="把 .onnx / .pth 权重放进 models 目录即可出现在这里",
            ),
            ParamSpec(
                "device", "运算设备", "choice", "auto",
                choices=[("auto", "自动"), ("cuda", "GPU (CUDA)"), ("cpu", "CPU")], group="基础",
            ),
            ParamSpec(
                "tile", "分块尺寸", "int", 256, minimum=0, maximum=2048, step=64,
                suffix=" px", group="基础",
                tooltip="大图分块推理防显存溢出；0 表示整图一次推理（更快但可能爆显存）",
            ),
            ParamSpec(
                "tile_pad", "分块重叠", "int", 10, minimum=0, maximum=64, step=2,
                suffix=" px", group="基础", depends_on=("tile", "nonzero"),
                tooltip="每块向四周多算这么多像素再裁中心，重叠越大接缝越不易察觉",
            ),
            ParamSpec(
                "half", "GPU 半精度 (fp16)", "bool", True, group="基础",
                depends_on=("backend", ["torch"]),
                tooltip="Torch 后端在 CUDA 上用半精度，省显存更快；结果略有差异",
            ),
            ParamSpec(
                "output_format", "输出格式", "choice", "png",
                choices=[("png", "PNG 无损"), ("jpg", "JPG 有损")], group="输出",
            ),
            ParamSpec(
                "jpg_quality", "JPG 质量", "int", 92, minimum=60, maximum=100, step=1,
                group="输出", depends_on=("output_format", ["jpg"]),
            ),
            ParamSpec(
                "out_dir", "输出目录", "path", "", group="输出",
                tooltip="自动保存到配置文件，重启后恢复；留空则使用配置的 output_dir/superres/<时间戳>/",
            ),
            ParamSpec(
                "name_template", "文件名模板", "str", "{stem}_x4", group="输出",
                tooltip="可用变量：{stem} {name} {ext}",
            ),
            ParamSpec("overwrite", "覆盖同名文件", "bool", False, group="输出"),
        ]

    # ------------------------------------------------------------ 后端
    def get_backend(self, params: dict[str, Any]) -> SuperresBackend:
        kind = str(params.get("backend") or "onnx")
        model = str(params.get("model") or "x4plus")
        device = str(params.get("device") or "auto")
        weight_path = resolve_weight(model, "onnx" if kind == "onnx" else "torch", self.config)
        stamp = None
        if weight_path is not None:
            stat = weight_path.stat()
            stamp = (str(weight_path.resolve()), stat.st_mtime_ns, stat.st_size)
        key = (kind, stamp or model, device)
        if self._backend is not None and self._backend_key == key:
            return self._backend
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
        self._backend = create_backend(
            kind,
            {"model": model, "half": params.get("half", True), "_config": self.config},
            device=device,
        )
        self._backend_key = key
        return self._backend

    # ------------------------------------------------------------ 处理
    def process(self, item: ImageItem, params: dict[str, Any], ctx: TaskContext) -> ModuleResult:
        ctx.check_cancelled()
        rgb = io_utils.read_image(item.path)
        item.rgb = rgb
        h, w = rgb.shape[:2]
        item.meta["size"] = (int(w), int(h))

        kind = str(params.get("backend") or "onnx")
        try:
            backend = self.get_backend(params)
        except (BackendUnavailable, FileNotFoundError, RuntimeError) as exc:
            return ModuleResult(item=item, ok=False, message=str(exc))

        ctx.check_cancelled()
        try:
            upscaled = backend.predict(
                rgb,
                tile=int(params.get("tile") or 0),
                tile_pad=int(params.get("tile_pad") or 10),
                check_cancelled=ctx.check_cancelled,
            )
        except Exception as exc:
            log.exception("超分推理失败（%s/%s）", kind, params.get("model"))
            return ModuleResult(item=item, ok=False, message=f"超分失败：{exc}")

        up_h, up_w = upscaled.shape[:2]

        # 预览结果（超大图不塞内存，预览退回原图）
        if up_w * up_h <= PREVIEW_PIXEL_CAP:
            item.meta["result_rgb"] = upscaled
        item.meta["result_size"] = (int(up_w), int(up_h))
        item.meta["upscale_scale"] = int(backend.scale)

        # 输出
        ctx.check_cancelled()
        out_dir = Path(params.get("_out_dir") or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt = str(params.get("output_format") or "png")
        ext = ".jpg" if fmt in {"jpg", "jpeg"} else ".png"
        target = io_utils.build_output_path(
            out_dir, item.path, str(params.get("name_template") or "{stem}_x4"), ext,
            bool(params.get("overwrite", False)),
        )
        try:
            if ext == ".jpg":
                io_utils.write_jpg(target, upscaled, quality=int(params.get("jpg_quality") or 92))
            else:
                io_utils.write_image(target, upscaled)
        except Exception as exc:
            log.exception("写出失败：%s", target)
            return ModuleResult(item=item, ok=False, message=f"写出失败：{exc}")

        outputs = {"main": target}
        item.meta["outputs"] = outputs
        item.meta["run_dir"] = out_dir
        item.meta["process_params"] = dict(params)
        from ...core.device import backend_device_label

        if kind == "onnx":
            active = getattr(backend, "active_providers", [])
            item.meta["device_label"] = "GPU · CUDA" if "CUDAExecutionProvider" in active else "CPU"
        else:
            item.meta["device_label"] = "GPU · CUDA" if backend.device == "cuda" else "CPU"
        return ModuleResult(
            item=item,
            ok=True,
            outputs=outputs,
            message=(
                f"{w}×{h} → {up_w}×{up_h} · {item.meta['device_label']} → {target.name}"
            ),
        )

    # ------------------------------------------------------------ 预览
    def preview_kinds(self) -> list[str]:
        return ["原图", "结果"]


MODULE = SuperresModule()
