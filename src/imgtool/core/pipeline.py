"""批量执行引擎：统一负责输出目录、进度、取消与异常兜底。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import io_utils
from .base import BaseModule
from .log import get_logger
from .types import ImageItem, ModuleResult, TaskCancelled, TaskContext

log = get_logger("pipeline")


class Pipeline:
    """把一个模块 + 一组参数跑在一批图片上。"""

    def __init__(
        self,
        module: BaseModule,
        params: Optional[dict[str, Any]] = None,
        config: Any = None,
    ) -> None:
        self.module = module
        self.config = config
        self.params = module.coerce_params(params or {})
        if config is not None:
            self.params.setdefault("device", config.get("device", "auto"))

    # ---------- 输出目录 ----------
    def resolve_run_dir(self) -> Path:
        """本次运行的输出目录：优先用参数里的 out_dir，否则 output/<模块>/<时间戳>。"""
        raw = str(self.params.get("out_dir") or "").strip()
        if raw:
            d = Path(raw).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            return d
        base = Path(self.config.output_dir) if self.config is not None else Path("output")
        return io_utils.make_run_dir(base, self.module.name)

    # ---------- 执行 ----------
    def run(
        self,
        items: Sequence[ImageItem] | Iterable[ImageItem],
        ctx: Optional[TaskContext] = None,
        run_dir: Optional[Path] = None,
        on_item_done: Any = None,
    ) -> list[ModuleResult]:
        items = list(items)
        total = len(items)
        ctx = ctx or TaskContext(config=self.config)
        out_dir = Path(run_dir) if run_dir is not None else self.resolve_run_dir()

        params = dict(self.params)
        params["_out_dir"] = str(out_dir)
        params["_run_dir"] = str(out_dir)

        results: list[ModuleResult] = []
        for index, item in enumerate(items):
            if ctx.cancelled():
                log.info("任务已取消（%d/%d）", index, total)
                break
            ctx.progress(index, total, f"处理中：{item.name}")
            started = time.perf_counter()
            try:
                result = self.module.process(item, params, ctx)
            except TaskCancelled:
                log.info("任务已取消：%s", item.name)
                break
            except Exception as exc:  # 单张失败不影响整批
                log.exception("处理失败：%s", item.path)
                result = ModuleResult(item=item, ok=False, message=str(exc))
            result.elapsed = time.perf_counter() - started
            results.append(result)
            if on_item_done is not None:
                on_item_done(result)
            ctx.progress(index + 1, total, f"{'完成' if result.ok else '失败'}：{item.name}")

        return results


def run_once(
    module: BaseModule,
    items: Sequence[ImageItem] | Iterable[ImageItem],
    params: Optional[dict[str, Any]] = None,
    config: Any = None,
    on_progress: Any = None,
    cancel: Any = None,
) -> list[ModuleResult]:
    """便捷函数：一次调用跑完一批。"""
    pipe = Pipeline(module, params, config)
    ctx = TaskContext(config=config, on_progress=on_progress, cancel=cancel)
    return pipe.run(items, ctx)
