"""后台执行线程：把 Pipeline 跑在 QThread 里，避免界面卡死。"""

from __future__ import annotations

from typing import Any, Sequence

from PySide6.QtCore import QThread, Signal

from ..core.base import BaseModule
from ..core.log import get_logger
from ..core.pipeline import Pipeline
from ..core.types import ImageItem, ModuleResult, TaskContext

log = get_logger("worker")


class ModuleWorker(QThread):
    """在一批图片上运行某个模块。"""

    progress = Signal(int, int, str)   # done, total, message
    itemDone = Signal(object)          # ModuleResult
    finishedAll = Signal(list)         # list[ModuleResult]
    failed = Signal(str)

    def __init__(
        self,
        module: BaseModule,
        items: Sequence[ImageItem],
        params: dict[str, Any],
        config: Any = None,
        parent=None,
        result_cache=None,
    ) -> None:
        super().__init__(parent)
        self.module = module
        self.items = list(items)
        self.params = dict(params)
        self.config = config
        self._cancel = False
        self.run_dir = None
        self.result_cache = result_cache

    def _item_done(self, result: ModuleResult) -> None:
        if result.ok and self.result_cache is not None:
            try:
                self.result_cache.store(result.item)
                self.result_cache.release(result.item)
            except (OSError, ValueError):
                log.warning("结果暂存失败，保留内存中的结果：%s", result.item.path, exc_info=True)
        self.itemDone.emit(result)

    def cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:  # noqa: D102
        results: list[ModuleResult] = []
        try:
            pipe = Pipeline(self.module, self.params, self.config)
            # 整批共用一个输出目录（时间戳只取一次）
            run_dir = pipe.resolve_run_dir()
            self.run_dir = run_dir
            ctx = TaskContext(
                config=self.config,
                on_progress=lambda done, total, msg: self.progress.emit(done, total, msg),
                cancel=lambda: self._cancel,
            )
            results = pipe.run(
                self.items, ctx, run_dir=run_dir, on_item_done=self._item_done
            )
        except Exception as exc:  # pragma: no cover
            log.exception("后台任务异常")
            self.failed.emit(str(exc))
        finally:
            self.finishedAll.emit(results)
