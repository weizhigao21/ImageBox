"""通用模块页：左侧参数面板 + 右侧预览/文件列表/进度。

任何模块都能直接用这一页，无需为 UI 写代码：
参数来自 ``module.param_spec()``，预览标签来自 ``module.preview_kinds()``。
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core import io_utils
from ...core.base import BaseModule
from ...core.log import get_logger
from ...core.mask_edit import MaskEdit
from ...core.result_cache import ResultCache
from ...core.types import ImageItem, ModuleResult
from ..dnd import FileDropRelay
from ..qt_utils import PREVIEW_MAX, ElidedLabel, downscale, load_preview, ndarray_to_qimage
from ..widgets import FileTable, ImageView, ParamPanel
from ..widgets.paint_view import ERASE, KEEP, PaintImageView
from ..workers import ModuleWorker

log = get_logger("page")


@dataclass
class EditSession:
    """一张图的手动涂抹会话（都在预览分辨率上）。"""

    base_rgb: np.ndarray
    base_alpha: np.ndarray
    edit: MaskEdit
    composite: Optional[np.ndarray] = None  # 当前显示的 alpha（增量刷新用）
    dirty: bool = False
    revision: Optional[int] = None


def _union_rect(
    rects: list[Optional[tuple[int, int, int, int]]], shape: tuple[int, int]
) -> Optional[tuple[int, int, int, int]]:
    """合并脏矩形并裁到图像范围内。"""
    valid = [r for r in rects if r is not None]
    if not valid:
        return None
    x0 = min(r[0] for r in valid)
    y0 = min(r[1] for r in valid)
    x1 = max(r[2] for r in valid)
    y1 = max(r[3] for r in valid)
    h, w = shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return x0, y0, x1, y1


def _segment_rect(
    x0: float, y0: float, x1: float, y1: float, radius: float, shape: tuple[int, int]
) -> Optional[tuple[int, int, int, int]]:
    """一段笔画影响的矩形（含笔刷半径与羽化余量）。"""
    pad = float(radius) + 6.0
    h, w = shape[:2]
    rx0 = max(0, int(min(x0, x1) - pad))
    ry0 = max(0, int(min(y0, y1) - pad))
    rx1 = min(w, int(max(x0, x1) + pad) + 1)
    ry1 = min(h, int(max(y0, y1) + pad) + 1)
    if rx1 - rx0 < 1 or ry1 - ry0 < 1:
        return None
    return rx0, ry0, rx1, ry1


class ModulePage(QWidget):
    """一个模块的工作台。"""

    dragActive = Signal(bool)  # 有文件悬停在页面上

    def __init__(self, module: BaseModule, config: Any = None, parent=None) -> None:
        super().__init__(parent)
        self.module = module
        self.config = config
        module.config = config

        self.items: list[ImageItem] = []
        self.results: dict[str, ModuleResult] = {}
        self.sessions: dict[str, EditSession] = {}
        self.hints: dict[str, MaskEdit] = {}
        self._hint_display: dict[str, np.ndarray] = {}
        self._hint_mask_cache: dict[str, np.ndarray] = {}
        self._preview_cache = OrderedDict()
        self._preview_cache_limit = 6
        self.result_cache = ResultCache()
        self._worker: Optional[ModuleWorker] = None
        self._run_dir: Optional[Path] = None
        self.paint_view: Optional[PaintImageView] = None
        self.hint_view: Optional[PaintImageView] = None
        self.hint_bar: Optional[QWidget] = None
        self.edit_bar: Optional[QWidget] = None
        self.brush_radius = 24.0
        self.brush_value = ERASE
        self._busy = False

        self._build()
        self._setup_drop_targets()
        self._load_preset_list()

    # ================================================================ 拖放
    def _setup_drop_targets(self) -> None:
        """页面本身、预览区、文件列表都能接收拖进来的图片。

        （窗口级别的兜底在 MainWindow 上，覆盖其余空白区域。）
        """
        self._drop_relays: list[FileDropRelay] = []
        targets = [self, self.table, *self.views.values()]
        for target in targets:
            relay = FileDropRelay(target, self)
            relay.dropped.connect(self.add_paths)
            relay.dragActive.connect(lambda active: self.dragActive.emit(active))
            self._drop_relays.append(relay)

    # ================================================================ 构建
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(22, 16, 22, 10)
        heading = QHBoxLayout()
        title = QLabel(self.module.title)
        self.page_title = title
        title.setProperty("role", "title")
        self.runtime_badge = ElidedLabel("")
        self.runtime_badge.setProperty("role", "status")
        self.runtime_badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        heading.addWidget(title)
        self.page_description = description = ElidedLabel(self.module.description)
        description.setProperty("role", "muted")
        heading.setSpacing(14)
        heading.addWidget(description, 1)
        header_layout.addLayout(heading)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        outer.addWidget(splitter, 1)

        # ---------------- 左：参数 ----------------
        left = QWidget()
        left.setObjectName("inspector")
        left.setMinimumWidth(320)
        left.setMaximumWidth(352)
        inspector_shell = QVBoxLayout(left)
        inspector_shell.setContentsMargins(0, 0, 0, 0)
        inspector_shell.setSpacing(0)
        switcher = QWidget()
        switcher_layout = QHBoxLayout(switcher)
        switcher_layout.setContentsMargins(18, 14, 18, 8)
        switcher_layout.setSpacing(6)
        self.btn_params = QPushButton("参数")
        self.btn_tasks = QPushButton("任务")
        self.inspector_buttons = QButtonGroup(self)
        self.inspector_buttons.setExclusive(True)
        for index, button in enumerate((self.btn_params, self.btn_tasks)):
            button.setCheckable(True)
            button.setProperty("kind", "sideTab")
            self.inspector_buttons.addButton(button, index)
            switcher_layout.addWidget(button, 1)
        self.btn_params.setChecked(True)
        self.btn_params.clicked.connect(lambda: self._show_inspector_page(0))
        self.btn_tasks.clicked.connect(lambda: self._show_inspector_page(1))
        inspector_shell.addWidget(switcher)
        self.inspector_stack = QStackedWidget()
        inspector_shell.addWidget(self.inspector_stack, 1)
        self.inspector_scroll = QScrollArea()
        self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inspector_body = QWidget()
        self.inspector_layout = left_layout = QVBoxLayout(inspector_body)
        left_layout.setContentsMargins(18, 14, 18, 14)
        left_layout.setSpacing(10)
        section = QLabel("处理参数")
        section.setProperty("role", "section")
        inspector_heading = QHBoxLayout()
        inspector_heading.addWidget(section)
        inspector_heading.addWidget(self.runtime_badge, 1)
        left_layout.addLayout(inspector_heading)

        self.panel = ParamPanel(self.module)
        self.panel.setEnabled(not self.module.placeholder)
        self.panel.valueChanged.connect(self._refresh_runtime_badge)

        left_layout.addWidget(self.panel)

        preset_box = QVBoxLayout()
        preset_box.setSpacing(6)
        self.preset_label = preset_label = QLabel("参数预设")
        preset_label.setProperty("role", "muted")
        preset_box.addWidget(preset_label)
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(0)
        btn_save_preset = QPushButton("保存预设")
        btn_load_preset = QPushButton("载入")
        btn_reset = QPushButton("重置")
        btn_save_preset.clicked.connect(self._save_preset)
        btn_load_preset.clicked.connect(self._load_preset)
        btn_reset.clicked.connect(lambda: self.panel.set_values(self.module.defaults()))
        preset_box.addWidget(self.preset_combo)
        preset_row.addWidget(btn_load_preset)
        preset_row.addWidget(btn_save_preset)
        preset_row.addWidget(btn_reset)
        preset_box.addLayout(preset_row)
        left_layout.addLayout(preset_box)
        left_layout.addStretch(1)
        self.inspector_scroll.setWidget(inspector_body)
        self.inspector_stack.addWidget(self.inspector_scroll)

        # ---------------- 中央：完整画布 ----------------
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(header)
        splitter.addWidget(content)

        self.tabs = QTabWidget()
        self.views: dict[str, ImageView] = {}
        for kind in self.module.preview_kinds():
            view: ImageView
            if kind == "结果" and self.module.supports_manual_mask:
                view = PaintImageView("处理后的图片会显示在这里\n选择图片后，点击「处理当前」")
                self.paint_view = view
                self._connect_paint_view(view, "edit")
            elif kind == "原图" and self.module.supports_fg_hint:
                view = PaintImageView("拖入图片，开始抠图\n也可在右侧「任务」中添加图片或文件夹")
                self.hint_view = view
                self._connect_paint_view(view, "hint")
            else:
                view = ImageView(f"{kind}（暂无）")
            self.views[kind] = view
            self.tabs.addTab(view, kind)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(22, 8, 22, 12)
        preview_layout.setSpacing(8)
        preview_tools = QHBoxLayout()
        self.preview_title = ElidedLabel("图片预览")
        preview_tools.addWidget(self.preview_title, 1)
        self.bg_combo = QComboBox()
        for text, value in (("透明棋盘", "checker"), ("白色背景", "white"), ("深色背景", "dark")):
            self.bg_combo.addItem(text, value)
        self.bg_combo.setFixedWidth(116)
        self.bg_combo.setToolTip("只切换预览底色，不改变导出背景")
        self.bg_combo.currentIndexChanged.connect(self._change_preview_background)
        fit_button = QPushButton("适应窗口")
        fit_button.clicked.connect(lambda: self.tabs.currentWidget().fit_to_view())
        preview_tools.addWidget(self.bg_combo)
        preview_tools.addWidget(fit_button)
        preview_layout.addLayout(preview_tools)
        preview_layout.addWidget(self.tabs, 1)

        self.brush_stack = QStackedWidget()
        if self.hint_view is not None:
            self.hint_bar = self._build_hint_bar()
            self.brush_stack.addWidget(self.hint_bar)
        if self.paint_view is not None:
            self.edit_bar = self._build_brush_bar()
            self.brush_stack.addWidget(self.edit_bar)
        if self.brush_stack.count():
            preview_layout.addWidget(self.brush_stack)
        content_layout.addWidget(preview_panel, 1)

        self.task_page = QWidget()
        task_layout = QVBoxLayout(self.task_page)
        task_layout.setContentsMargins(18, 12, 18, 14)
        task_layout.setSpacing(8)

        self.file_count = QLabel("图片队列 · 0")
        self.file_count.setProperty("role", "section")
        task_layout.addWidget(self.file_count)
        self.btn_add = QPushButton("添加图片")
        self.btn_add_dir = QPushButton("添加文件夹")
        self.btn_remove = QPushButton("移除选中")
        self.btn_clear = QPushButton("清空")
        self.btn_open = QPushButton("输出目录")
        self.btn_add.clicked.connect(self.add_files)
        self.btn_add_dir.clicked.connect(self.add_folder)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_items)
        self.btn_open.clicked.connect(self.open_output_dir)
        import_row = QHBoxLayout()
        import_row.addWidget(self.btn_add, 1)
        import_row.addWidget(self.btn_add_dir, 1)
        manage_row = QHBoxLayout()
        manage_row.addWidget(self.btn_remove, 1)
        manage_row.addWidget(self.btn_clear, 1)
        task_layout.addLayout(import_row)
        task_layout.addLayout(manage_row)
        task_hint = QLabel("选择任务查看图片 · Ctrl / Shift 多选")
        task_hint.setProperty("role", "muted")
        task_layout.addWidget(task_hint)

        self.table = FileTable()
        self.table.set_compact(True)
        self.table.currentRowChanged.connect(self._on_row_changed)
        task_layout.addWidget(self.table, 1)
        self.inspector_stack.addWidget(self.task_page)

        run_bar = QWidget()
        run_bar.setObjectName("runBar")
        run_bar.setFixedHeight(78)
        run_layout = QVBoxLayout(run_bar)
        run_layout.setContentsMargins(22, 10, 22, 8)
        run_layout.setSpacing(8)
        run_row = QHBoxLayout()
        self.btn_run_current = QPushButton("处理当前")
        self.btn_run_current.setObjectName("primary")
        self.btn_run_all = QPushButton("处理全部")
        self.btn_run_selected = QPushButton("处理选中")
        self.btn_run_selected.clicked.connect(self.process_selected)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_run_current.clicked.connect(self.process_current)
        self.btn_run_all.clicked.connect(self.process_all)
        self.btn_cancel.clicked.connect(self.cancel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.status = ElidedLabel("添加图片，开始处理")
        self.status.setProperty("role", "status")
        run_row.addWidget(self.btn_run_current)
        run_row.addWidget(self.btn_run_all)
        run_row.addWidget(self.btn_run_selected)
        run_row.addWidget(self.btn_cancel)
        run_row.addWidget(self.status, 1)
        run_row.addWidget(self.btn_open)
        run_layout.addLayout(run_row)
        run_layout.addWidget(self.progress)
        splitter.addWidget(left)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([820, 352])
        outer.addWidget(run_bar)
        self._refresh_runtime_badge()
        self._refresh_actions()
        self._update_brush_bar()

    def _show_inspector_page(self, index: int) -> None:
        self.inspector_stack.setCurrentIndex(index)
        (self.btn_params if index == 0 else self.btn_tasks).setChecked(True)

    def _change_preview_background(self) -> None:
        for view in self.views.values():
            view.set_preview_background(self.bg_combo.currentData())

    def _refresh_runtime_badge(self) -> None:
        from ...core.device import backend_device_label
        params = self.panel.values()
        backend = str(params.get("backend") or "simple")
        device = backend_device_label(backend, str(params.get("device") or "auto"))
        model = str(params.get("onnx_model") or "auto") if backend == "onnx" else ""
        self.runtime_badge.setText(" · ".join(p for p in (backend.upper(), model, device) if p))

    # ================================================================ 文件
    def add_paths(self, paths: list[str]) -> int:
        if self._busy or getattr(self.window(), "_closing", False):
            self.status.setText("正在处理图片；完成后可继续添加")
            return 0
        files = io_utils.list_images(paths, recursive=True)
        known = {str(item.path) for item in self.items}
        added = 0
        for path in files:
            if str(path) in known:
                continue
            item = ImageItem(path)
            self.items.append(item)
            row = self.table.append_item(item)
            w, h = io_utils.probe_size(path)
            if w and h:
                self.table.set_size(path, f"{w}×{h}")
            known.add(str(path))
            added += 1
            if row == 0 and self.table.currentRow() < 0:
                self.table.selectRow(0)
        if added:
            self.status.setText(f"已添加 {added} 张，共 {len(self.items)} 张")
            self._show_inspector_page(1)
        self.file_count.setText(f"图片队列 · {len(self.items)}")
        self._refresh_actions()
        return added

    def add_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(io_utils.IMAGE_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", f"图片 ({patterns});;所有文件 (*)")
        if files:
            self.add_paths(files)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            self.add_paths([folder])

    def remove_selected(self) -> None:
        paths = {str(p) for p in self.table.selected_paths()}
        if not paths:
            return
        self.items = [item for item in self.items if str(item.path) not in paths]
        for path in paths:
            self.result_cache.remove(path)
            self.results.pop(path, None)
            self.sessions.pop(path, None)
            self.hints.pop(path, None)
            self._hint_display.pop(path, None)
            self._hint_mask_cache.pop(path, None)
            self._preview_cache.pop(path, None)
        self._refresh_table()

    def clear_items(self) -> None:
        self.result_cache.close()
        self.items.clear()
        self.results.clear()
        self.sessions.clear()
        self.hints.clear()
        self._hint_display.clear()
        self._hint_mask_cache.clear()
        self._preview_cache.clear()
        self._refresh_table()
        for view in self.views.values():
            view.clear_image()
        self._update_brush_bar()
        self.status.setText("已清空")

    def _refresh_table(self) -> None:
        self.file_count.setText(f"图片队列 · {len(self.items)}")
        current = self.table.current_path()
        self.table.set_items(self.items)
        for item in self.items:
            w, h = io_utils.probe_size(item.path)
            if w and h:
                self.table.set_size(item.path, f"{w}×{h}")
            result = self.results.get(str(item.path))
            if result is not None:
                self.table.set_status(item.path, "完成" if result.ok else "失败", result.elapsed)
        if current is not None:
            row = self.table.row_for(current)
            if row is not None:
                self.table.selectRow(row)
            elif self.items:
                self.table.selectRow(0)
        elif self.items:
            self.table.selectRow(0)
        self._refresh_actions()

    # ================================================================ 画笔
    def _build_hint_bar(self) -> QWidget:
        """原图页工具条：涂抹「我要抠出来的部分」。"""
        bar = QWidget()
        column = QVBoxLayout(bar)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(6)
        details = QHBoxLayout()
        details.setSpacing(6)

        self.hint_check = QCheckBox("画笔标记主体")
        self.hint_check.setToolTip(
            "在原图上标记主体或漏抠部位，再点「按标记抠图」；"
            "需要直接擦除或补回时，请用结果页的画笔涂抹"
        )
        self.hint_check.toggled.connect(self._sync_paint_mode)

        size_label = QLabel("笔刷")
        self.hint_slider = QSlider(Qt.Orientation.Horizontal)
        self.hint_slider.setRange(4, 300)
        self.hint_slider.setValue(int(self.brush_radius))
        self.hint_slider.setFixedWidth(110)
        self.hint_slider.valueChanged.connect(self._set_brush_radius)
        self.hint_size_label = QLabel(f"{int(self.brush_radius)} px")
        self.hint_size_label.setFixedWidth(48)

        self.hint_undo = QPushButton("撤销")
        self.hint_clear = QPushButton("清除标记")
        self.btn_process_hint = QPushButton("按标记抠图")
        self.btn_process_hint.setObjectName("primary")
        self.btn_process_hint.setToolTip("按原图上的绿色标记处理当前图片，完成后自动显示结果")
        self.hint_undo.clicked.connect(lambda: self.undo_stroke("hint"))
        self.hint_clear.clicked.connect(lambda: self.clear_strokes("hint"))
        self.btn_process_hint.clicked.connect(self.process_current)

        self.hint_hint = ElidedLabel("")
        self.hint_hint.setProperty("role", "muted")

        row.addWidget(self.hint_check)
        row.addWidget(size_label)
        row.addWidget(self.hint_slider)
        row.addWidget(self.hint_size_label)
        row.addStretch(1)
        row.addWidget(self.btn_process_hint)
        details.addWidget(self.hint_undo)
        details.addWidget(self.hint_clear)
        details.addWidget(self.hint_hint, 1)
        column.addLayout(row)
        column.addLayout(details)
        return bar

    def _build_brush_bar(self) -> QWidget:
        bar = QWidget()
        column = QVBoxLayout(bar)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(6)
        details = QHBoxLayout()
        details.setSpacing(6)

        self.brush_check = QCheckBox("画笔涂抹")
        self.brush_check.setToolTip("在「结果」页按住左键涂掉多余部分；按住 Alt 反向涂抹")
        self.brush_check.toggled.connect(self._sync_paint_mode)

        self.btn_erase = QPushButton("擦除")
        self.btn_keep = QPushButton("保留")
        for btn, value in ((self.btn_erase, ERASE), (self.btn_keep, KEEP)):
            btn.setCheckable(True)
            btn.setFixedWidth(58)
            btn.clicked.connect(lambda _=False, v=value: self._set_brush_value(v))
        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.btn_erase)
        group.addButton(self.btn_keep)
        self.btn_erase.setChecked(True)
        self.btn_erase.setToolTip("把涂到的地方变成透明（去掉多余内容）")
        self.btn_keep.setToolTip("把涂到的地方恢复成前景（找回被误删的主体）")

        size_label = QLabel("笔刷")
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(4, 220)
        self.brush_slider.setValue(int(self.brush_radius))
        self.brush_slider.setFixedWidth(110)
        self.brush_slider.valueChanged.connect(self._set_brush_radius)
        self.brush_size_label = QLabel(f"{int(self.brush_radius)} px")
        self.brush_size_label.setFixedWidth(48)

        self.btn_undo = QPushButton("撤销")
        self.btn_clear_strokes = QPushButton("清除涂抹")
        self.btn_undo.clicked.connect(self.undo_stroke)
        self.btn_clear_strokes.clicked.connect(self.clear_strokes)

        self.btn_save_edit = QPushButton("保存修改")
        self.btn_save_edit.setObjectName("primary")
        self.btn_save_edit.clicked.connect(self.save_manual_edit)

        self.brush_hint = ElidedLabel("")
        self.brush_hint.setProperty("role", "muted")

        row.addWidget(self.brush_check)
        row.addWidget(self.btn_erase)
        row.addWidget(self.btn_keep)
        row.addWidget(size_label)
        row.addWidget(self.brush_slider)
        row.addWidget(self.brush_size_label)
        row.addStretch(1)
        row.addWidget(self.btn_save_edit)
        details.addWidget(self.btn_undo)
        details.addWidget(self.btn_clear_strokes)
        details.addWidget(self.brush_hint, 1)
        column.addLayout(row)
        column.addLayout(details)
        return bar

    def _connect_paint_view(self, view: PaintImageView, role: str) -> None:
        view.strokeBegin.connect(lambda x, y, v, r=role: self._on_stroke_begin(r, x, y, v))
        view.strokeMove.connect(lambda x0, y0, x1, y1, r=role: self._on_stroke_move(r, x0, y0, x1, y1))
        view.strokeEnd.connect(lambda r=role: self._on_stroke_end(r))
        view.undoRequested.connect(lambda r=role: self.undo_stroke(r))
        view.brushSizeStep.connect(self._on_brush_size_step)

    def _on_tab_changed(self, index: int) -> None:
        self._sync_paint_mode()

    def _sync_paint_mode(self, *_: Any) -> None:
        current = self.tabs.currentWidget()
        if self.hint_view is not None and current is self.hint_view:
            # 原图页：标记要抠的主体，不需要先处理
            self._update_brush_bar()
            return
        if self.paint_view is None:
            return
        # 结果页：精修；勾了画笔但当前图还没有结果，就先自动处理一次
        if self.brush_check.isChecked() and not self._brush_ready():
            if self._busy:
                return
            if self.items:
                self.status.setText("正在处理当前图片，完成后即可涂抹…")
                self.process_current()
            else:
                self.status.setText("请先添加图片，再勾选「画笔涂抹」")
                self.brush_check.setChecked(False)
            return
        self._update_brush_bar()

    def _brush_ready(self) -> bool:
        """当前选中的图片是否已经有可涂抹的结果。"""
        path = self.table.current_path()
        if path is None:
            return False
        result = self.results.get(str(path))
        return bool(result is not None and result.ok)

    def _set_brush_value(self, value: int) -> None:
        self.brush_value = value
        if self.paint_view is not None:
            self.paint_view.brush_value = value
        self._update_brush_bar()

    def _set_brush_radius(self, value: int) -> None:
        self.brush_radius = float(value)
        for view in (self.paint_view, self.hint_view):
            if view is not None:
                view.brush_radius = float(value)
        # 两条工具条的滑块互相同步（模块可能只支持其中一种）
        for name in ("hint_slider", "brush_slider"):
            slider = getattr(self, name, None)
            if slider is not None and slider.value() != int(value):
                slider.blockSignals(True)
                slider.setValue(int(value))
                slider.blockSignals(False)
        for name in ("brush_size_label", "hint_size_label"):
            label = getattr(self, name, None)
            if label is not None:
                label.setText(f"{int(value)} px")

    def _on_brush_size_step(self, step: int) -> None:
        self._set_brush_radius(int(self.brush_radius + step * 6))

    # ---------------------------------------------------------- 会话
    def _current_path(self) -> Optional[Path]:
        return self.table.current_path()

    def _current_session(self) -> Optional[EditSession]:
        path = self._current_path()
        if path is None:
            return None
        return self.sessions.get(str(path))

    def _current_hint(self, create: bool = False) -> Optional[MaskEdit]:
        """取当前图片的「主体标记」；create=True 时按原图预览尺寸新建。"""
        path = self._current_path()
        if path is None:
            return None
        key = str(path)
        hint = self.hints.get(key)
        if hint is None and create:
            data = self._original_preview(path)
            if data is None:
                return None
            hint = MaskEdit(data[0].shape[:2])
            self.hints[key] = hint
        return hint

    def _ensure_session(self, path: Path, result: ModuleResult) -> EditSession:
        """为某张图准备涂抹会话（基础图/基础 alpha 都缩到预览分辨率）。"""
        key = str(path)
        self.result_cache.load(result.item)
        self.result_cache.evict_except(result.item)
        base_rgb = downscale(result.item.meta.get("result_rgb", result.item.rgb))
        alpha = result.item.meta.get("edit_base_alpha", result.item.alpha)
        if alpha is None:
            alpha = result.item.meta.get("preview_alpha")
        if alpha is None:
            alpha = np.full(base_rgb.shape[:2], 255, np.uint8)
        base_alpha = downscale(alpha)

        revision = result.item.meta.get("result_revision")
        session = self.sessions.get(key)
        if session is None:
            session = EditSession(base_rgb, base_alpha, MaskEdit(base_alpha.shape[:2]), revision=revision)
            self.sessions[key] = session
        else:
            session.base_rgb = base_rgb
            session.base_alpha = base_alpha
            session.composite = None  # 底图换了，下一次重画要整块重算
            if session.edit.shape != tuple(base_alpha.shape[:2]):
                session.edit = session.edit.scaled(base_alpha.shape[:2])
            if not session.edit.is_empty and (revision is None or revision != session.revision):
                session.dirty = True  # 重新处理后需要再存一次涂抹结果
            session.revision = revision
        # 无笔迹的会话可重新生成；有修改的会话始终保留。
        self.sessions.pop(key, None)
        self.sessions[key] = session
        for old_key in list(self.sessions):
            if len(self.sessions) <= 6:
                break
            if old_key != key and self.sessions[old_key].edit.is_empty and not self.sessions[old_key].dirty:
                self.sessions.pop(old_key)
        return session

    def _composed_alpha(self, session: EditSession) -> np.ndarray:
        return session.edit.apply(session.base_alpha)

    # ---------------------------------------------------------- 笔画
    def _on_stroke_begin(self, role: str, x: float, y: float, value: int) -> None:
        if role == "hint":
            hint = self._current_hint(create=True)
            if hint is None:
                return
            hint.begin(x, y, self.brush_radius, KEEP)
            self._repaint_original(
                rect=_segment_rect(x, y, x, y, self.brush_radius, hint.shape)
            )
            return
        session = self._current_session()
        if session is None:
            return
        session.edit.begin(x, y, self.brush_radius, value)
        self._repaint_result(
            session, rect=_segment_rect(x, y, x, y, self.brush_radius, session.edit.shape)
        )

    def _on_stroke_move(self, role: str, x0: float, y0: float, x1: float, y1: float) -> None:
        if role == "hint":
            hint = self._current_hint()
            if hint is None:
                return
            hint.extend(x1, y1)
            self._repaint_original(
                rect=_segment_rect(x0, y0, x1, y1, self.brush_radius, hint.shape)
            )
            return
        session = self._current_session()
        if session is None:
            return
        session.edit.extend(x1, y1)
        self._repaint_result(
            session, rect=_segment_rect(x0, y0, x1, y1, self.brush_radius, session.edit.shape)
        )

    def _on_stroke_end(self, role: str) -> None:
        path = self._current_path()
        if role == "hint":
            hint = self._current_hint()
            if hint is None:
                return
            hint.end()
            self._store_hint(path, hint)
            self._repaint_original()  # 收笔时整块重算一次，保证与涂抹层完全一致
            if path is not None:
                self.table.set_status(path, "已标记")
            self.status.setText("已标记主体；点「按标记抠图」查看结果")
            self._update_brush_bar()
            return

        session = self._current_session()
        if session is None:
            return
        session.edit.end()
        session.dirty = True
        if path is not None:
            self.table.set_status(path, "已涂抹")
        self._repaint_result(session)
        self._update_brush_bar()

    def _store_hint(self, path: Optional[Path], hint: MaskEdit) -> None:
        """把标记写回图片条目（处理时模块会读取 item.meta['fg_hint']）。"""
        if path is None:
            return
        item = next((i for i in self.items if i.path == path), None)
        if item is None:
            return
        if hint.is_empty:
            item.meta.pop("fg_hint", None)
        else:
            item.meta["fg_hint"] = (hint.foreground_mask() * 255.0).astype(np.uint8)

    def _repaint_original(
        self,
        rect: Optional[tuple[int, int, int, int]] = None,
        fit: bool = False,
    ) -> None:
        """重画原图预览：有标记时叠加半透明绿色，方便看清标了哪块。"""
        view = self.views.get("原图")
        path = self._current_path()
        if view is None or path is None:
            return
        data = self._original_preview(path)
        if data is None:
            view.clear_image()
            return
        rgb, alpha = data
        key = str(path)
        display = self._hint_display.get(key)
        if display is None or display.shape != rgb.shape:
            display = rgb.copy()
            self._hint_display[key] = display
            rect = None

        hint = self.hints.get(key)
        if hint is None or hint.is_empty:
            if rect is None:
                np.copyto(display, rgb)
                view.set_numpy(display, alpha, fit=fit)
            return

        green = np.array([0, 255, 0], np.float32).reshape(1, 1, 3)
        if rect is None:
            mask = hint.foreground_mask(rgb.shape[:2])
            self._hint_mask_cache[key] = mask
            m = mask[:, :, None] * 0.45
            np.copyto(
                display,
                (rgb.astype(np.float32) * (1.0 - m) + green * m).astype(np.uint8),
            )
            view.set_numpy(display, alpha, fit=fit)
            return

        mask = self._hint_mask_cache.get(key)
        if mask is None or mask.shape != rgb.shape[:2]:
            self._repaint_original()  # 缓存缺失 → 整块重算
            return
        x0, y0, x1, y1 = rect
        mask[y0:y1, x0:x1] = hint.keep_region(x0, y0, x1, y1)
        m = mask[y0:y1, x0:x1, None] * 0.45
        sub = (rgb[y0:y1, x0:x1].astype(np.float32) * (1.0 - m) + green * m).astype(np.uint8)
        display[y0:y1, x0:x1] = sub
        view.update_region(sub, alpha[y0:y1, x0:x1] if alpha is not None else None, x0, y0)

    def _repaint_result(
        self, session: EditSession, rect: Optional[tuple[int, int, int, int]] = None
    ) -> None:
        """重画结果预览；给了 rect 就只更新那一块（涂抹时不整图重建）。"""
        result_view = self.views.get("结果")
        mask_view = self.views.get("遮罩")
        if session.composite is None or session.composite.shape != session.base_alpha.shape:
            rect = None

        if rect is None:
            alpha = self._composed_alpha(session)
            session.composite = alpha
            if result_view is not None:
                result_view.set_numpy(session.base_rgb, alpha, fit=False)
            if mask_view is not None and mask_view.isVisible():
                mask_view.set_numpy(alpha, fit=False)
            return

        x0, y0, x1, y1 = rect
        sub = session.edit.apply_region(session.base_alpha, x0, y0, x1, y1)
        assert session.composite is not None
        session.composite[y0:y1, x0:x1] = sub
        if result_view is not None:
            result_view.update_region(session.base_rgb[y0:y1, x0:x1], sub, x0, y0)
        if mask_view is not None and mask_view.isVisible():
            mask_view.update_region(np.dstack([sub, sub, sub]), None, x0, y0)

    def undo_stroke(self, role: str = "edit") -> None:
        path = self._current_path()
        if role == "hint":
            hint = self._current_hint()
            if hint is None or not hint.undo():
                return
            self._store_hint(path, hint)
            self._repaint_original()
            self._update_brush_bar()
            self.status.setText("已撤销一笔标记")
            return
        session = self._current_session()
        if session is None or not session.edit.undo():
            return
        session.dirty = True
        if path is not None:
            self.table.set_status(path, "已涂抹")
        self._repaint_result(session)
        self._update_brush_bar()
        self.status.setText("已撤销一笔")

    def clear_strokes(self, role: str = "edit") -> None:
        path = self._current_path()
        if role == "hint":
            hint = self._current_hint()
            if hint is None or hint.is_empty:
                return
            hint.clear()
            self._store_hint(path, hint)
            self._repaint_original()
            self._update_brush_bar()
            self.status.setText("已清除主体标记，将恢复全自动抠图")
            return
        session = self._current_session()
        if session is None or session.edit.is_empty:
            return
        session.edit.clear()
        session.dirty = True
        if path is not None:
            self.table.set_status(path, "已涂抹")
        self._repaint_result(session)
        self._update_brush_bar()
        self.status.setText("已清除涂抹")

    def save_manual_edit(self) -> None:
        path = self.table.current_path()
        if path is None:
            return
        result = self.results.get(str(path))
        session = self.sessions.get(str(path))
        if result is None or not result.ok or session is None:
            QMessageBox.information(self, "提示", "请先处理这张图片，再进行涂抹")
            return
        if session.edit.is_empty and not session.dirty:
            QMessageBox.information(self, "提示", "还没有涂抹内容")
            return

        params = self.module.coerce_params(self.panel.values())
        image_run_dir = result.item.meta.get("run_dir") or result.output_dir
        if image_run_dir is not None:
            params["_out_dir"] = str(image_run_dir)
        else:
            params["_out_dir"] = str(io_utils.make_run_dir(
                self.config.output_dir if self.config else "output", self.module.name
            ))

        edited_full = session.edit.apply(result.item.meta.get("edit_base_alpha", result.item.alpha))
        try:
            outputs = self.module.save_manual_mask(result.item, params, edited_full)
        except Exception as exc:  # pragma: no cover - 取决于文件系统
            log.exception("保存涂抹结果失败")
            QMessageBox.warning(self, "保存失败", str(exc))
            return

        try:
            self.result_cache.store(result.item)
        except (OSError, ValueError):
            log.warning("修改已导出，但结果暂存失败", exc_info=True)

        session.dirty = False
        self._run_dir = next(iter(outputs.values())).parent if outputs else self._run_dir
        self.table.set_status(path, "完成")
        names = "、".join(f"{k}:{v.name}" for k, v in outputs.items())
        self.status.setText(f"涂抹结果已保存 → {names}")
        self._update_brush_bar()

    def _update_brush_bar(self) -> None:
        """刷新两条画笔工具条的可见性/可用状态与提示。"""
        path = self._current_path()
        result = self.results.get(str(path)) if path is not None else None
        session = self.sessions.get(str(path)) if path is not None else None
        hint = self.hints.get(str(path)) if path is not None else None
        if session is None and path is not None and result is not None and result.ok:
            self.result_cache.load(result.item)
            if result.item.rgb is not None:
                session = self._ensure_session(path, result)

        idle = not self._busy
        current = self.tabs.currentWidget()
        edit_ready = bool(result is not None and result.ok and session is not None) and idle
        hint_ready = bool(path is not None and self._original_preview(path) is not None) and idle

        # ---------------- 原图页：标记要抠的主体 ----------------
        if self.hint_view is not None and self.hint_bar is not None:
            self.hint_check.setEnabled(idle and path is not None)
            self.hint_slider.setEnabled(idle)
            has_hint = bool(hint is not None and not hint.is_empty)
            self.hint_undo.setEnabled(hint_ready and has_hint)
            self.hint_clear.setEnabled(hint_ready and has_hint)
            self.btn_process_hint.setEnabled(hint_ready and has_hint)
            painting_hint = (
                hint_ready and self.hint_check.isChecked() and current is self.hint_view
            )
            self.hint_view.set_painting_enabled(painting_hint)

            if path is None:
                self.hint_hint.setText("先添加图片")
            elif not self.hint_check.isChecked():
                self.hint_hint.setText("勾选后在主体上涂画，再点按标记抠图")
            else:
                mode = str(self.panel.values().get("hint_mode") or "subject")
                tip = "涂画后点按标记抠图" if mode == "subject" else "只抠标记范围；涂画后点按标记抠图"
                self.hint_hint.setText(tip)

        # ---------------- 结果页：精修 ----------------
        if self.paint_view is not None and self.edit_bar is not None:
            self.brush_check.setEnabled(idle)
            self.btn_erase.setEnabled(idle)
            self.btn_keep.setEnabled(idle)
            self.brush_slider.setEnabled(idle)
            has_strokes = bool(session is not None and not session.edit.is_empty)
            for btn, enabled in (
                (self.btn_undo, edit_ready and has_strokes),
                (self.btn_clear_strokes, edit_ready and has_strokes),
                (self.btn_save_edit, edit_ready and (has_strokes or bool(session and session.dirty))),
            ):
                btn.setEnabled(enabled)
            painting = edit_ready and self.brush_check.isChecked() and current is self.paint_view
            self.paint_view.set_painting_enabled(painting)

            if not edit_ready:
                self.brush_hint.setText("先处理图片")
            elif not self.brush_check.isChecked():
                self.brush_hint.setText("勾选后修补结果")
            else:
                self.brush_hint.setText("涂抹时红色擦除、绿色补回；松开后显示成品 · Alt 反向")

        if self.hint_view is not None and self.hint_bar is not None:
            # 两层工具条同处一个 stack，靠当前页切换
            self.brush_stack.setCurrentWidget(
                self.hint_bar if current is self.hint_view else (self.edit_bar or self.hint_bar)
            )

    # ================================================================ 预览
    def _original_preview(self, path: Path) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        key = str(path)
        if key not in self._preview_cache:
            try:
                self._preview_cache[key] = load_preview(path)
                while len(self._preview_cache) > self._preview_cache_limit:
                    old_key, _ = self._preview_cache.popitem(last=False)
                    self._hint_display.pop(old_key, None)
                    self._hint_mask_cache.pop(old_key, None)
            except Exception as exc:
                log.warning("预览失败 %s：%s", path, exc)
                return None
        self._preview_cache.move_to_end(key)
        return self._preview_cache[key]

    def _on_row_changed(self, row: int) -> None:
        self._refresh_actions()
        path = self.table.current_path()
        if path is None:
            self.preview_title.setText("图片预览")
            if not self.items:
                for view in self.views.values():
                    view.clear_image()
            return
        size = self.table.item(row, 1).text()
        self.preview_title.setText(f"{path.name} · {size}" if size != "-" else path.name)
        self._update_preview(path)

    def _update_preview(self, path: Path) -> None:
        kinds = self.module.preview_kinds()
        result = self.results.get(str(path))
        if result is not None and result.ok:
            self.result_cache.load(result.item)

        if "原图" in self.views:
            if self.hint_view is not None:
                self._repaint_original(fit=True)
            else:
                data = self._original_preview(path)
                if data is not None:
                    self.views["原图"].set_numpy(data[0], data[1], fit=True)
                else:
                    self.views["原图"].clear_image()

        if result is not None and result.ok and result.item.rgb is not None:
            session = self._ensure_session(path, result)
            alpha = self._composed_alpha(session)
            session.composite = alpha
            if "结果" in self.views:
                self.views["结果"].set_numpy(session.base_rgb, alpha, fit=True)
            if "遮罩" in self.views:
                self.views["遮罩"].set_numpy(alpha, fit=True)
        else:
            for kind in kinds:
                if kind != "原图":
                    self.views[kind].clear_image()
        self._update_brush_bar()

    # ================================================================ 执行
    def _targets(self, current_only: bool) -> list[ImageItem]:
        if current_only:
            path = self.table.current_path()
            if path is None:
                return []
            return [item for item in self.items if item.path == path]
        if not self.items:
            return []
        return list(self.items)

    def _start(self, items: list[ImageItem]) -> None:
        if getattr(self.window(), "_closing", False):
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "提示", "已有任务在运行中")
            return
        if not items:
            QMessageBox.information(self, "提示", "请先添加图片")
            return
        if self.module.placeholder:
            QMessageBox.information(self, "提示", f"{self.module.title} 模块尚未实现")
            return

        try:
            params = self.module.coerce_params(self.panel.values())
        except ValueError as exc:
            self.status.setText(str(exc))
            QMessageBox.warning(self, "参数需要调整", str(exc))
            return
        self.results = {k: v for k, v in self.results.items() if k not in {str(i.path) for i in items}}
        for item in items:
            self.result_cache.remove(str(item.path))
            self.table.set_status(item.path, "等待处理")

        self._set_busy(True)
        self.progress.setRange(0, len(items))
        self.progress.setValue(0)

        if self._worker is not None:
            self._worker.deleteLater()
        worker = ModuleWorker(
            self.module, items, params, self.config, self,
            result_cache=self.result_cache if self.module.supports_manual_mask else None,
        )
        worker.progress.connect(self._on_progress)
        worker.itemDone.connect(self._on_item_done)
        worker.finishedAll.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def process_current(self) -> None:
        self._start(self._targets(current_only=True))

    def process_all(self) -> None:
        self._start(self._targets(current_only=False))

    def process_selected(self) -> None:
        selected = set(self.table.selected_paths())
        self._start([item for item in self.items if item.path in selected])

    def cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.status.setText("正在取消…")
            self.btn_cancel.setEnabled(False)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.btn_run_current.setEnabled(not busy)
        self.btn_run_all.setEnabled(not busy)
        self.btn_run_selected.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_add.setEnabled(not busy)
        self.btn_add_dir.setEnabled(not busy)
        self.btn_clear.setEnabled(not busy)
        self.btn_remove.setEnabled(not busy)
        self.panel.setEnabled(not busy and not self.module.placeholder)
        self._refresh_actions()
        self._update_brush_bar()

    def _refresh_actions(self) -> None:
        idle = not self._busy
        selected = bool(self.table.selected_paths())
        self.btn_run_current.setEnabled(idle and self.table.current_path() is not None)
        self.btn_run_all.setEnabled(idle and bool(self.items))
        self.btn_run_selected.setEnabled(idle and selected)
        self.btn_remove.setEnabled(idle and selected)
        self.btn_clear.setEnabled(idle and bool(self.items))

    # ---------------------------------------------------------- 回调
    def _on_progress(self, done: int, total: int, message: str) -> None:
        if getattr(self.window(), "_closing", False):
            return
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status.setText(f"{done}/{total} · {message}")
        if self._worker is not None and done < len(self._worker.items) and message.startswith("处理中"):
            self.table.set_status(self._worker.items[done].path, "处理中")

    def _on_item_done(self, result: ModuleResult) -> None:
        if getattr(self.window(), "_closing", False):
            return
        item = result.item
        self.results[str(item.path)] = result
        self.table.set_status(
            item.path, "完成" if result.ok else "失败", result.elapsed
        )
        if not result.ok and result.message:
            log.warning("%s：%s", item.name, result.message)
            row = self.table.row_for(item.path)
            if row is not None:
                self.table.item(row, 2).setToolTip(result.message)
        if result.ok:
            backend = (item.meta.get("process_params") or {}).get("backend", "")
            cache_note = "复用推理" if item.meta.get("inference_cached") else ""
            self.runtime_badge.setText(" · ".join(str(x) for x in (str(backend).upper(), item.meta.get("device_label"), cache_note) if x))
        if self.table.current_path() == item.path:
            self._update_preview(item.path)
        self._update_brush_bar()

    def _on_finished(self, results: list) -> None:
        if getattr(self.window(), "_closing", False):
            return
        self._set_busy(False)
        ok = sum(1 for r in results if getattr(r, "ok", False))
        failed = len(results) - ok
        cancelled = self._worker.cancelled if self._worker is not None else False
        if self._worker is not None:
            if self._worker.run_dir is not None:
                self._run_dir = Path(self._worker.run_dir)
            completed = {r.item.path for r in results}
            for item in self._worker.items:
                if item.path not in completed:
                    self.table.set_status(item.path, "已取消" if cancelled else "未完成")
            self._worker.items = []
        msg = f"完成 {ok} 张"
        if failed:
            msg += f"，失败 {failed} 张"
        if cancelled:
            msg += "（已取消）"
        if self._run_dir is not None:
            msg += f"　输出目录：{self._run_dir}"
        self.status.setText(msg)
        result_view = self.views.get("结果")
        if result_view is not None:
            current = self.table.current_path()
            successful = [r for r in results if getattr(r, "ok", False)]
            target = next((r for r in successful if r.item.path == current), None)
            if target is None and successful:
                target = successful[0]
            if target is not None:
                if target.item.path != current:
                    row = self.table.row_for(target.item.path)
                    if row is not None:
                        self.table.selectRow(row)
                self.tabs.setCurrentWidget(result_view)
                QTimer.singleShot(0, result_view.fit_to_view)
        self._update_brush_bar()
        failures = [r for r in results if not getattr(r, "ok", False)]
        if failures and len(failures) <= 3 and not getattr(self.window(), "_closing", False):
            QMessageBox.warning(
                self,
                "部分图片处理失败",
                "\n".join(f"{r.item.name}：{r.message}" for r in failures),
            )

    def _on_failed(self, message: str) -> None:
        if getattr(self.window(), "_closing", False):
            return
        self._set_busy(False)
        self.status.setText(f"出错：{message}")
        if not getattr(self.window(), "_closing", False):
            QMessageBox.critical(self, "任务出错", message)

    # ================================================================ 输出
    def open_output_dir(self) -> None:
        target: Optional[Path] = self._run_dir
        if target is None:
            for result in self.results.values():
                if result.outputs:
                    target = Path(next(iter(result.outputs.values()))).parent
                    break
        if target is None:
            params = self.panel.values()
            raw = str(params.get("out_dir") or "").strip()
            target = Path(raw) if raw else (
                Path(self.config.output_dir) / self.module.name if self.config else Path("output")
            )
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
        io_utils.open_in_explorer(target)
        self.status.setText(f"输出目录：{target}")

    # ================================================================ 预设
    def _load_preset_list(self) -> None:
        self.preset_combo.clear()
        if self.config is None:
            return
        self.preset_combo.addItem("（选择预设）", "")
        for path in sorted(Path(self.config.presets_dir).glob(f"{self.module.name}_*.json")):
            self.preset_combo.addItem(path.stem.replace(f"{self.module.name}_", ""), str(path))

    def _save_preset(self) -> None:
        if self.config is None:
            return
        name, ok = QInputDialog.getText(self, "保存预设", "预设名称：")
        if not ok or not name.strip():
            return
        target = Path(self.config.presets_dir) / f"{self.module.name}_{name.strip()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.panel.values(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._load_preset_list()
        self.status.setText(f"预设已保存：{target.name}")

    def _load_preset(self) -> None:
        path = self.preset_combo.currentData()
        if not path:
            return
        try:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError("预设内容必须是参数对象")
            values = self.module.coerce_params(values)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "读取失败", str(exc))
            return
        self.panel.set_values(values)
        self.status.setText(f"已载入预设：{Path(path).stem}")
