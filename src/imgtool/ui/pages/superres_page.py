"""超分页：在通用模块页上增加「权重」状态条与下载。"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.base import BaseModule
from ...core.device import backend_device_label
from ...modules.superres.backend import (
    MODEL_REGISTRY,
    download_weight,
    models_dir,
    resolve_weight,
)
from ..download_worker import DownloadWorker
from ..qt_utils import ElidedLabel
from ..widgets.param_panel import ParamPanel
from .module_page import ModulePage


class SuperresPage(ModulePage):
    """超分工作台。"""

    def __init__(self, module: BaseModule, config: Any = None, parent=None) -> None:
        super().__init__(module, config, parent)
        self._download_worker: DownloadWorker | None = None
        self._build_backend_bar()
        self.refresh_backend_status()

    # ------------------------------------------------------------------
    def _build_backend_bar(self) -> None:
        bar = QWidget()
        column = QVBoxLayout(bar)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        self.weight_label = ElidedLabel("权重检查中…")
        self.weight_label.setProperty("role", "status")
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        status_row.addWidget(self.weight_label, 1)
        btn_check = QPushButton("刷新")
        btn_check.setProperty("kind", "text")
        btn_check.setToolTip("刷新权重状态")
        btn_check.clicked.connect(self.refresh_backend_status)
        status_row.addWidget(btn_check)
        column.addLayout(status_row)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.download_combo = QComboBox()
        self.download_combo.setMinimumWidth(0)
        self.download_combo.setToolTip("选择要下载的权重")
        self._fill_download_choices()
        self.btn_download = QPushButton("下载")
        self.btn_download.setFixedWidth(72)
        self.btn_download.clicked.connect(self.download_weights)
        row.addWidget(self.download_combo, 1)
        row.addWidget(self.btn_download)
        column.addLayout(row)
        self.inspector_layout.insertWidget(1, bar)

    def _fill_download_choices(self) -> None:
        self.download_combo.clear()
        for key, cfg in MODEL_REGISTRY.items():
            if cfg.get("onnx_url"):
                self.download_combo.addItem(f"{cfg['label']} · ONNX", (key, "onnx"))
            if cfg.get("torch_url"):
                self.download_combo.addItem(f"{cfg['label']} · Torch", (key, "torch"))

    # ------------------------------------------------------------------
    def refresh_backend_status(self) -> None:
        params = self.panel.values() if hasattr(self, "panel") else {}
        kind = str(params.get("backend") or "onnx")
        model = str(params.get("model") or "x4plus")
        path = resolve_weight(model, "onnx" if kind == "onnx" else "torch", self.config)
        if kind == "onnx":
            dev = "GPU · CUDA" if "CUDAExecutionProvider" in _onnx_providers() else "CPU"
        else:
            dev = backend_device_label("", "auto")
        if path is not None:
            self.weight_label.setText(f"{kind.upper()} · {path.name} 就绪 · {dev}")
        else:
            self.weight_label.setText(
                f"未找到 {model} 的 {'ONNX' if kind == 'onnx' else 'Torch'} 权重 · 可点「下载」或手动放入 {models_dir(self.config)}"
            )
        self.btn_download.setToolTip(
            f"下载到 {models_dir(self.config)}" + (f"\n已有：{path.name}" if path else "")
        )
        self._refresh_model_choices()

    def _refresh_model_choices(self) -> None:
        if not isinstance(self.panel, ParamPanel):
            return
        spec = next((s for s in self.module.param_spec() if s.key == "model"), None)
        if spec is None:
            return
        self.panel.set_choices("model", self.module.param_spec_choices("model"))

    # ------------------------------------------------------------------
    def download_weights(self) -> None:
        if getattr(self.window(), "_closing", False):
            return
        if self._download_worker is not None and self._download_worker.isRunning():
            self._download_worker.cancel()
            self.btn_download.setEnabled(False)
            self.weight_label.setText("正在取消下载…")
            return
        data = self.download_combo.currentData()
        if not data:
            return
        model, kind = data
        self.btn_download.setEnabled(True)
        self.btn_download.setText("取消")
        self.btn_download.setToolTip("取消当前下载")
        self.weight_label.setText(f"正在下载 {model}（{kind.upper()}）…")
        self.download_combo.setEnabled(False)

        def downloader(m: str, cfg: Any, on_progress=None, cancel=None):
            return download_weight(m, kind, cfg, on_progress=on_progress, cancel=cancel)

        worker = DownloadWorker(str(model), self.config, self, downloader=downloader)
        worker.progress.connect(self._on_download_progress)
        worker.done.connect(self._on_download_done)
        worker.failed.connect(self._on_download_failed)
        worker.cancelled.connect(self._on_download_cancelled)
        self._download_worker = worker
        worker.start()

    def _on_download_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.weight_label.setText(
                f"下载中 · {done / total:.0%} · {done / 1048576:.1f} / {total / 1048576:.1f} MB"
            )
        else:
            self.weight_label.setText(f"下载中 · {done / 1048576:.1f} MB")

    def _on_download_done(self, path: str) -> None:
        self.btn_download.setEnabled(True)
        self.btn_download.setText("下载")
        self.download_combo.setEnabled(True)
        if getattr(self.window(), "_closing", False):
            return
        self.refresh_backend_status()
        data = self.download_combo.currentData()
        if data:
            model, kind = data
            self.panel.set_values({"backend": kind, "model": str(model)})
        self.status.setText(f"权重已保存：{path}")
        self.btn_download.setToolTip(f"权重已保存到：{path}")

    def _on_download_failed(self, message: str) -> None:
        self.btn_download.setEnabled(True)
        self.btn_download.setText("下载")
        self.download_combo.setEnabled(True)
        if getattr(self.window(), "_closing", False):
            return
        self.refresh_backend_status()
        QMessageBox.warning(
            self,
            "下载失败",
            f"{message}\n\n也可以手动下载权重放到：\n{models_dir(self.config)}",
        )

    def _on_download_cancelled(self) -> None:
        if getattr(self.window(), "_closing", False):
            return
        self.btn_download.setText("下载")
        self.btn_download.setEnabled(True)
        self.download_combo.setEnabled(True)
        self.refresh_backend_status()
        self.status.setText("下载已取消")


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []
