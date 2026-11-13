"""抠图模块：调度后端 → 后处理 → 输出。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...core import io_utils
from ...core.base import BackendUnavailable, BaseModule, ParamSpec
from ...core.log import get_logger
from ...core.types import ImageItem, ModuleResult, TaskContext
from . import postprocess
from .backends import BACKENDS, create_backend, list_local_models
from .backends.onnx_backend import MODEL_REGISTRY, resolve_model_file

log = get_logger("matting")

BACKEND_CHOICES = [
    ("simple", "简易 GrabCut（无需权重）"),
    ("onnx", "ONNX 模型（推荐）"),
    ("birefnet", "BiRefNet（实验）"),
]

OUTPUT_CHOICES = [
    ("rgba", "透明 PNG（RGBA）"),
    ("color", "纯色背景图"),
    ("mask", "仅遮罩"),
    ("all", "全部输出"),
]


class MattingModule(BaseModule):
    name = "matting"
    title = "抠图"
    icon = "✂"
    description = "把主体从背景中分离，输出透明 PNG / 纯色底图 / 遮罩。"
    order = 10
    supports_manual_mask = True
    supports_fg_hint = True

    def __init__(self) -> None:
        self._backend = None
        self._backend_key: Optional[tuple] = None
        self.config: Any = None
        # 只缓存最近一张的原始推理，避免每次调整羽化/输出都重跑模型。
        self._prediction_cache = None

    # ------------------------------------------------------------ 参数
    def _onnx_choices(self) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = [("auto", "自动（models 首选）")]
        known_files = set()
        for key, cfg in MODEL_REGISTRY.items():
            known_files.add(cfg["file"])
            choices.append((key, cfg["label"]))
        for fname in list_local_models(self.config):
            if fname not in known_files:
                choices.append((fname, f"本地权重：{fname}"))
        return choices

    def param_spec(self) -> list[ParamSpec]:
        # 本地已有权重时默认用模型；无权重的安装仍可直接用简易模式。
        model_available = bool(list_local_models(self.config))
        settings = self.config.sub("matting") if self.config is not None else {}
        selected_model = str(settings.get("onnx_model") or "auto")
        if resolve_model_file(selected_model, self.config) is None:
            selected_model = "auto"
        default_backend = str(settings.get("backend") or ("onnx" if model_available else "simple"))
        if default_backend == "onnx" and not model_available:
            default_backend = "simple"
        return [
            ParamSpec(
                "backend", "抠图后端", "choice", default_backend,
                choices=BACKEND_CHOICES, group="基础",
                tooltip="简易后端无需下载权重；ONNX 需要 models 目录下有 .onnx 文件",
            ),
            ParamSpec(
                "onnx_model", "ONNX 权重", "choice", selected_model,
                choices=self._onnx_choices(), group="基础",
                depends_on=("backend", ["onnx"]),
                tooltip="把 .onnx 权重放进 models 目录即可出现在这里",
            ),
            ParamSpec(
                "device", "运算设备", "choice", "auto",
                choices=[("auto", "自动"), ("cuda", "GPU (CUDA)"), ("cpu", "CPU")], group="基础",
            ),
            ParamSpec(
                "max_size", "处理长边", "int", 1600, minimum=320, maximum=8000, step=64,
                suffix=" px", group="基础",
                tooltip="先缩放到该长边再处理，越大越精细也越慢",
                depends_on=("backend", ["simple"]),
            ),
            ParamSpec(
                "hint_mode", "标记含义", "choice", "subject",
                choices=[
                    ("subject", "标记主体 / 补回漏抠部位"),
                    ("region", "严格只抠涂抹到的范围"),
                ],
                group="基础",
                tooltip="主体模式选中已识别的主体并补回笔迹；简易后端还会用笔迹重新分割。范围模式只保留笔迹附近",
            ),
            ParamSpec(
                "fg_hint_expand", "涂抹范围外扩", "int", 12, minimum=0, maximum=300, step=2,
                suffix=" px", group="基础",
                tooltip="在原图上涂了「要抠的主体」后，把该范围外扩这么多像素再抠，"
                        "避免手涂边界把主体切掉；没涂抹时此项无效",
            ),
            ParamSpec(
                "threshold", "Alpha 阈值", "float", 8.0, minimum=0.0, maximum=254.0, step=1.0,
                decimals=1, group="边缘",
                tooltip="低于该值的半透明当作全透明，可去掉灰蒙蒙的边",
            ),
            ParamSpec(
                "feather", "羽化半径", "float", 1.5, minimum=0.0, maximum=30.0, step=0.5,
                decimals=1, suffix=" px", group="边缘",
            ),
            ParamSpec(
                "expand", "边缘外扩/内缩", "float", 0.0, minimum=-30.0, maximum=30.0, step=1.0,
                decimals=1, suffix=" px", group="边缘",
                tooltip="正数多留边缘，负数收紧（可减少白边）",
            ),
            ParamSpec(
                "gamma", "Alpha 对比度", "float", 1.0, minimum=0.2, maximum=3.0, step=0.1,
                decimals=2, group="边缘", tooltip=">1 更实，<1 更透",
            ),
            ParamSpec(
                "decontaminate", "去色边（反解前景色）", "bool", True, group="边缘",
                tooltip="消除边缘残留的背景颜色，透明 PNG 更干净",
            ),
            ParamSpec(
                "outline_width", "描边宽度", "int", 0, minimum=0, maximum=40, step=1,
                suffix=" px", group="边缘", tooltip="0 表示不描边",
            ),
            ParamSpec(
                "outline_color", "描边颜色", "color", "#FFFFFF", group="边缘",
                depends_on=("outline_width", "nonzero"),
            ),
            ParamSpec(
                "output_mode", "输出内容", "choice", "rgba",
                choices=OUTPUT_CHOICES, group="输出",
            ),
            ParamSpec(
                "bg_color", "背景色", "color", "#FFFFFF", group="输出",
                depends_on=("output_mode", ["color", "all"]),
            ),
            ParamSpec(
                "out_dir", "输出目录", "path", "", group="输出",
                tooltip="自动保存到配置文件，重启后恢复；留空则使用配置的 output_dir/matting/<时间戳>/",
            ),
            ParamSpec(
                "name_template", "文件名模板", "str", "{stem}_cutout", group="输出",
                tooltip="可用变量：{stem} {name} {ext}",
            ),
            ParamSpec("overwrite", "覆盖同名文件", "bool", False, group="输出"),
        ]

    # ------------------------------------------------------------ 后端
    def _backend_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "max_size": params.get("max_size", 1600),
            "onnx_model": params.get("onnx_model", "auto"),
            "birefnet_repo": params.get("birefnet_repo"),
            "birefnet_size": params.get("birefnet_size"),
            "_config": self.config,
        }

    def get_backend(self, params: dict[str, Any]):
        name = str(params.get("backend") or "simple")
        device = str(params.get("device") or "auto")
        weight_stamp = None
        if name == "onnx":
            weight_path = resolve_model_file(str(params.get("onnx_model") or "auto"), self.config)
            if weight_path is not None:
                stat = weight_path.stat()
                weight_stamp = (str(weight_path.resolve()), stat.st_mtime_ns, stat.st_size)
        key = (
            name, (weight_stamp or str(params.get("onnx_model") or "auto")) if name == "onnx" else None, device,
            int(params.get("max_size") or 0) if name == "simple" else None,
            str(params.get("birefnet_repo")) if name == "birefnet" else None,
            params.get("birefnet_size") if name == "birefnet" else None,
            str(getattr(self.config, "models_dir", "models")) if name == "onnx" else None,
        )
        if self._backend is not None and self._backend_key == key:
            return self._backend
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
        self._backend = create_backend(name, self._backend_params(params), device=device)
        self._backend_key = key
        return self._backend

    # ------------------------------------------------------------ 处理
    def process(self, item: ImageItem, params: dict[str, Any], ctx: TaskContext) -> ModuleResult:
        out_dir = Path(params.get("_out_dir") or ".")

        # 1) 读图
        ctx.check_cancelled()
        rgb = io_utils.read_image(item.path)
        item.rgb = rgb
        item.meta["size"] = (int(rgb.shape[1]), int(rgb.shape[0]))

        # 用户在原图上涂抹的「要抠的主体」范围（原图坐标，0/255）
        hint = params.get("_fg_mask")
        if hint is None:
            hint = item.meta.get("fg_hint")
        if hint is not None and not np.any(hint):
            hint = None

        # 2) 推理
        backend_name = str(params.get("backend") or "simple")
        try:
            backend = self.get_backend(params)
            # 「严格范围」只裁剪自动分割结果；把整条笔迹强制当作前景，
            # 会把画在背景上的部分也拉进 GrabCut，产生笔刷形状的色块。
            guided_hint = hint if str(params.get("hint_mode") or "subject") == "subject" else None
            stat = item.path.stat()
            cache_key = (str(item.path.resolve()), stat.st_mtime_ns, stat.st_size, self._backend_key)
            cached = self._prediction_cache
            if guided_hint is None and cached is not None and cached[0] == cache_key:
                alpha = cached[1].copy()
                item.meta["inference_cached"] = True
            else:
                alpha = backend.predict(rgb, hint=guided_hint)
                item.meta["inference_cached"] = False
                if guided_hint is None:
                    self._prediction_cache = (cache_key, alpha.copy())
        except BackendUnavailable as exc:
            return ModuleResult(item=item, ok=False, message=str(exc))
        except FileNotFoundError as exc:
            return ModuleResult(item=item, ok=False, message=str(exc))
        except Exception as exc:
            log.exception("后端 %s 推理失败", backend_name)
            return ModuleResult(item=item, ok=False, message=f"后端 {backend_name} 失败：{exc}")

        # 3) 后处理
        ctx.check_cancelled()
        if getattr(backend, "post_cleanup", False):
            alpha = postprocess.cleanup_mask(alpha)
        alpha = postprocess.refine_alpha(
            alpha,
            rgb,
            threshold=float(params.get("threshold", 8.0)),
            expand=float(params.get("expand", 0.0)),
            feather=float(params.get("feather", 1.5)),
            gamma=float(params.get("gamma", 1.0)),
        )

        hint_used = hint is not None
        if hint_used:
            expand = float(params.get("fg_hint_expand", 12))
            hint_mode = str(params.get("hint_mode") or "subject")
            if hint_mode == "region":
                # 严格模式：涂抹范围之外一律不要
                alpha = postprocess.apply_foreground_hint(
                    alpha, hint, expand=expand, feather=max(2.0, expand / 2.0)
                )
            else:
                # 默认模式：涂抹只用来指明「是哪只主体」，轮廓仍由算法决定
                alpha = postprocess.apply_hint_subject(
                    alpha, hint, expand=max(4.0, expand / 2.0), feather=3.0
                )

        result_rgb = rgb
        if bool(params.get("decontaminate", True)):
            result_rgb = postprocess.decontaminate(result_rgb, alpha)

        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        outline_width = int(params.get("outline_width") or 0)
        if outline_width > 0:
            color = io_utils.parse_color(params.get("outline_color", "#FFFFFF"))
            result_rgb, alpha_u8 = postprocess.outline(
                result_rgb, alpha_u8, width=outline_width, color=color
            )

        item.alpha = alpha_u8
        item.meta["edit_base_alpha"] = alpha_u8
        item.meta["result_rgb"] = result_rgb
        item.meta["preview_alpha"] = alpha_u8
        item.meta["fg_hint_used"] = hint_used

        # 4) 输出
        ctx.check_cancelled()
        outputs = self.write_outputs(item, params, result_rgb, alpha_u8)
        item.meta["outputs"] = outputs
        item.meta["result_revision"] = int(item.meta.get("result_revision", 0)) + 1
        item.meta["run_dir"] = out_dir
        item.meta["process_params"] = dict(params)
        from ...core.device import backend_device_label
        item.meta["device_label"] = backend_device_label(backend_name, str(params.get("device") or "auto"))
        if backend_name == "onnx":
            active = getattr(backend, "active_providers", [])
            item.meta["device_label"] = "GPU · CUDA" if "CUDAExecutionProvider" in active else "CPU"
        elif backend_name == "birefnet":
            item.meta["device_label"] = "GPU · CUDA" if backend.device == "cuda" else "CPU"
        names = "、".join(f"{k}:{v.name}" for k, v in outputs.items())
        return ModuleResult(
            item=item,
            ok=True,
            outputs=outputs,
            message=(
                f"{backend_name}{'（按涂抹范围）' if hint_used else ''} 完成 → {names}"
                if names
                else f"{backend_name} 完成"
            ),
        )

    # ------------------------------------------------------------ 输出
    def _target_path(
        self,
        item: ImageItem,
        out_dir: Path,
        kind: str,
        template: str,
        ext: str,
        overwrite: bool,
    ) -> Path:
        """生成输出路径；同一张图已输出过则原地覆盖，避免涂抹后产生 _1 副本。"""
        existing = (item.meta.get("outputs") or {}).get(kind)
        if existing is not None and Path(existing).parent == Path(out_dir):
            return Path(existing)
        return io_utils.build_output_path(out_dir, item.path, template, ext, overwrite)

    def write_outputs(
        self, item: ImageItem, params: dict[str, Any], rgb: np.ndarray, alpha_u8: np.ndarray
    ) -> dict[str, Path]:
        """按「输出内容」参数写文件（正常输出与手动涂抹修正共用）。"""
        out_dir = Path(params.get("_out_dir") or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = str(params.get("output_mode") or "rgba")
        template = str(params.get("name_template") or "{stem}_cutout")
        overwrite = bool(params.get("overwrite", False))
        bg = io_utils.parse_color(params.get("bg_color", "#FFFFFF"))

        outputs: dict[str, Path] = {}
        if mode in {"rgba", "all"}:
            p = self._target_path(item, out_dir, "rgba", template, ".png", overwrite)
            io_utils.write_image(p, io_utils.to_rgba(rgb, alpha_u8))
            outputs["rgba"] = p
        if mode in {"color", "all"}:
            p = self._target_path(item, out_dir, "color", template, ".jpg", overwrite)
            io_utils.write_image(p, io_utils.flatten(rgb, alpha_u8, bg))
            outputs["color"] = p
        if mode in {"mask", "all"}:
            p = self._target_path(item, out_dir, "mask", f"{template}_mask", ".png", overwrite)
            io_utils.write_image(p, alpha_u8)
            outputs["mask"] = p
        return outputs

    # ------------------------------------------------------------ 手动涂抹
    def save_manual_mask(
        self, item: ImageItem, params: dict[str, Any], alpha: np.ndarray
    ) -> dict[str, Path]:
        """用涂抹修正后的 alpha 重新输出。"""
        rgb = item.meta.get("result_rgb")
        if rgb is None:
            rgb = item.rgb if item.rgb is not None else io_utils.read_image(item.path)
            item.meta["result_rgb"] = rgb
        alpha_u8 = (
            alpha if alpha.dtype == np.uint8 else np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        )
        outputs = self.write_outputs(item, params, rgb, alpha_u8)
        item.alpha = alpha_u8
        item.meta["preview_alpha"] = alpha_u8
        item.meta["outputs"] = outputs
        return outputs


MODULE = MattingModule()
