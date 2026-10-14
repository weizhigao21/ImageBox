"""模块与后端的抽象接口。

新增一个处理能力 = 新建 modules/<name>/ 目录并暴露 ``MODULE`` 实例。
UI 的参数面板由 :meth:`BaseModule.param_spec` 自动生成，无需改动 UI 代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .types import ImageItem, ModuleResult, TaskContext


class BackendUnavailable(RuntimeError):
    """后端不可用（缺依赖 / 缺权重 / 设备不支持）。"""


@dataclass
class ParamSpec:
    """一个可自动生成 UI 的参数描述。"""

    key: str
    label: str
    kind: str = "str"  # choice | int | float | bool | str | path | color
    default: Any = None
    choices: Optional[Sequence[tuple[str, str]]] = None  # [(value, 显示名)]
    minimum: float = 0.0
    maximum: float = 100.0
    step: float = 1.0
    decimals: int = 2
    suffix: str = ""
    tooltip: str = ""
    group: str = "基础"
    #: (另一个参数的 key, 生效条件)。条件可以是取值列表，或 "nonzero"/"truthy"
    depends_on: Optional[tuple[str, Any]] = None

    def coerce(self, value: Any) -> Any:
        """转换并校验外部参数；错误必须在开始处理前明确报告。"""
        try:
            if self.kind in {"int", "float"}:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("需要有限数值")
                if self.kind == "int" and not number.is_integer():
                    raise ValueError("需要整数")
                if not self.minimum <= number <= self.maximum:
                    raise ValueError(f"范围为 {self.minimum:g}–{self.maximum:g}")
                return int(number) if self.kind == "int" else number
            if self.kind == "bool":
                if isinstance(value, str):
                    text = value.strip().lower()
                    if text not in {"1", "true", "yes", "on", "是", "0", "false", "no", "off", "否"}:
                        raise ValueError("需要 true 或 false")
                    return text in {"1", "true", "yes", "on", "是"}
                return bool(value)
            if self.kind == "choice":
                text = str(value)
                choices = {str(v) for v, _ in self.choices or []}
                if choices and text not in choices:
                    raise ValueError(f"可选值：{', '.join(sorted(choices))}")
                return text
            if self.kind in {"str", "path", "color"}:
                return "" if value is None else str(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{self.label}（{self.key}）：{exc}") from exc
        return value


class BaseBackend(ABC):
    """处理后端基类（如抠图的各种模型）。"""

    name: str = "base"
    label: str = "base"

    def __init__(self, params: dict[str, Any] | None = None, device: str = "auto") -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.device = device
        self._loaded = False

    @classmethod
    def available(cls, config: Any = None) -> tuple[bool, str]:
        """返回 (是否可用, 原因)。"""
        return True, ""

    def load(self) -> None:
        """加载模型/资源。可重复调用，实现里应保持幂等。"""
        self._loaded = True

    @abstractmethod
    def predict(self, rgb: np.ndarray, hint: Optional[np.ndarray] = None) -> np.ndarray:
        """输入 HxWx3 uint8 RGB，返回 HxW float32 前景透明度，范围 0~1。

        hint: 可选的 HxW 前景提示（0~1 或 0/255），表示「只在这片区域里找主体」。
              后端可以据此初始化（如 GrabCut），不支持时忽略即可。
        """

    def close(self) -> None:
        self._loaded = False


class BaseModule(ABC):
    """一个图片处理模块。"""

    name: str = "base"
    title: str = "模块"
    icon: str = "◆"
    description: str = ""
    placeholder: bool = False  # True 表示仅占位、尚未实现
    order: int = 100  # UI 中的排序权重，越小越靠前
    #: 是否支持「画笔涂抹」手动修正（模块需实现 save_manual_mask）
    supports_manual_mask: bool = False
    #: 是否支持在原图上涂抹「我要抠出来的部分」（处理前提示前景范围）
    supports_fg_hint: bool = False
    #: 由 UI/CLI 注入的 AppConfig，模块可用它找 models 目录等
    config: Any = None

    # ---------- 参数 ----------
    def param_spec(self) -> list[ParamSpec]:
        return []

    def defaults(self) -> dict[str, Any]:
        return {spec.key: spec.default for spec in self.param_spec()}

    def param_spec_choices(self, key: str) -> list[tuple[Any, str]]:
        """取某个 choice 参数的最新候选项（可用于动态刷新下拉框）。"""
        for spec in self.param_spec():
            if spec.key == key:
                return list(spec.choices or [])
        return []

    def coerce_params(self, values: dict[str, Any]) -> dict[str, Any]:
        """按 param_spec 校正类型，补齐缺失项。"""
        specs = {spec.key: spec for spec in self.param_spec()}
        out = self.defaults()
        for key, value in (values or {}).items():
            if key in specs:
                out[key] = specs[key].coerce(value)
            else:
                out[key] = value
        return out

    # ---------- 处理 ----------
    @abstractmethod
    def process(self, item: ImageItem, params: dict[str, Any], ctx: TaskContext) -> ModuleResult:
        """处理一张图片。params 中会带上本次运行的输出目录 ``_out_dir``。"""

    def save_manual_mask(self, item: ImageItem, params: dict[str, Any], alpha: np.ndarray) -> dict[str, Path]:
        """用「手动涂抹后」的 alpha 重新输出（仅 supports_manual_mask 的模块需要实现）。"""
        raise NotImplementedError(f"{self.title} 不支持手动修正")

    # ---------- 预览 ----------
    def preview_kinds(self) -> list[str]:
        return ["原图", "结果", "遮罩"]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"


class PlaceholderModule(BaseModule):
    """占位模块：UI 中可见，但提示“开发中”。"""

    placeholder = True

    def __init__(
        self,
        name: str,
        title: str,
        icon: str = "◇",
        description: str = "",
        order: int = 100,
    ) -> None:
        self.name = name
        self.title = title
        self.icon = icon
        self.description = description or "该模块正在规划中。"
        self.order = order

    def process(self, item: ImageItem, params: dict[str, Any], ctx: TaskContext) -> ModuleResult:
        return ModuleResult(item=item, ok=False, message=f"{self.title} 模块尚未实现")
