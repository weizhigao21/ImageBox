"""抠图页：在通用模块页上增加「后端/权重」状态条。"""

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
from ...modules.matting.backends import backend_status
from ...modules.matting.backends.onnx_backend import MODEL_REGISTRY, list_local_models, models_dir
from ..download_worker import DownloadWorker
from ..widgets.param_panel import ParamPanel
from ..qt_utils import ElidedLabel
from ...core.device import backend_device_label
from .module_page import ModulePage

SHORT_NAMES = {"simple": "简易", "onnx": "ONNX", "birefnet": "BiRefNet"}


class MattingPage(ModulePage):
    """抠图工作台。"""

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

        self.backend_label = ElidedLabel("后端检查中…")
        self.backend_label.setProperty("role", "status")
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        status_row.addWidget(self.backend_label, 1)
        column.addLayout(status_row)
        row = QHBoxLayout()
        row.setSpacing(6)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(0)
        self.model_combo.setToolTip("选择要下载的 ONNX 权重")
        for key, cfg in MODEL_REGISTRY.items():
            self.model_combo.addItem(key.upper(), key)
        recommended = self.model_combo.findData("rmbg-1.4")
        if recommended >= 0:
            self.model_combo.setCurrentIndex(recommended)
        self.btn_download = QPushButton("下载")
        self.btn_download.setFixedWidth(72)
        self.btn_download.clicked.connect(self.download_weights)
        btn_check = QPushButton("刷新")
        btn_check.setProperty("kind", "text")
        btn_check.setToolTip("刷新模型状态")
        btn_check.clicked.connect(self.refresh_backend_status)
        status_row.addWidget(btn_check)

        row.addWidget(self.model_combo, 1)
        row.addWidget(self.btn_download)
        column.addLayout(row)
        self.inspector_layout.insertWidget(1, bar)

    # ------------------------------------------------------------------
    def refresh_backend_status(self) -> None:
        rows = backend_status(self.config, quick=True)
        parts: list[str] = []
        tips: list[str] = []
        for name, label, ok, reason in rows:
            short = SHORT_NAMES.get(name, label)
            if name == "onnx" and ok:
                parts.append(f"ONNX · {backend_device_label('onnx')}")
            elif name == "simple":
                parts.append("简易可用")
            tips.append(f"{label}：{'可用' if ok else '不可用'}{f'（{reason}）' if reason else ''}")
        self.backend_label.setText("后端：" + "　".join(parts))
        self.backend_label.setToolTip("\n".join(tips))

        local = list_local_models(self.config)
        if not local:
            self.backend_label.setText(
                "简易后端可用 · 可下载模型提升效果"
            )
        self.btn_download.setToolTip(
            f"下载到 {models_dir(self.config)}" + (f"\n已有：{', '.join(local)}" if local else "")
        )
        self._refresh_model_choices()

    def _refresh_model_choices(self) -> None:
        if not isinstance(self.panel, ParamPanel):
            return
        spec = next((s for s in self.module.param_spec() if s.key == "onnx_model"), None)
        if spec is None:
            return
        self.panel.set_choices("onnx_model", self.module.param_spec_choices("onnx_model"))

    # ------------------------------------------------------------------
    def download_weights(self) -> None:
        if getattr(self.window(), "_closing", False):
            return
        if self._download_worker is not None and self._download_worker.isRunning():
            self._download_worker.cancel()
            self.btn_download.setEnabled(False)
            self.backend_label.setText("正在取消下载…")
            return
        model = self.model_combo.currentData()
        if not model:
            return
        self.btn_download.setEnabled(True)
        self.btn_download.setText("取消")
        self.btn_download.setToolTip("取消当前下载")
        self.backend_label.setText(f"正在下载 {str(model).upper()}…")
        self.model_combo.setEnabled(False)

        worker = DownloadWorker(str(model), self.config, self)
        worker.progress.connect(self._on_download_progress)
        worker.done.connect(self._on_download_done)
        worker.failed.connect(self._on_download_failed)
        worker.cancelled.connect(self._on_download_cancelled)
        self._download_worker = worker
        worker.start()

    def _on_download_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.backend_label.setText(f"下载中 · {done / total:.0%} · {done / 1048576:.1f} / {total / 1048576:.1f} MB")
        else:
            self.backend_label.setText(f"下载中 · {done / 1048576:.1f} MB")

    def _on_download_done(self, path: str) -> None:
        self.btn_download.setEnabled(True)
        self.btn_download.setText("下载")
        self.model_combo.setEnabled(True)
        if getattr(self.window(), "_closing", False):
            return
        self.refresh_backend_status()
        model = self.model_combo.currentData()
        if model:
            self.panel.set_values({"backend": "onnx", "onnx_model": str(model)})
        self.status.setText(f"权重已保存并切换到 ONNX：{path}")
        self.btn_download.setToolTip(f"权重已保存到：{path}")

    def _on_download_failed(self, message: str) -> None:
        self.btn_download.setEnabled(True)
        self.btn_download.setText("下载")
        self.model_combo.setEnabled(True)
        if getattr(self.window(), "_closing", False):
            return
        self.refresh_backend_status()
        QMessageBox.warning(
            self,
            "下载失败",
            f"{message}\n\n也可以手动下载 .onnx 放到：\n{models_dir(self.config)}",
        )

    def _on_download_cancelled(self) -> None:
        if getattr(self.window(), "_closing", False):
            return
        self.btn_download.setText("下载")
        self.btn_download.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.refresh_backend_status()
        self.status.setText("下载已取消")
