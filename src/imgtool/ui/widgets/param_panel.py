"""根据模块的 param_spec() 自动生成参数面板。"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal, QTimer, QEvent
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.base import BaseModule, ParamSpec


class ScrollComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class ScrollSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class ScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class ParamPanel(QWidget):
    """把模块参数渲染成一组控件，并提供 values()/set_values()。"""

    valueChanged = Signal()

    def __init__(self, module: BaseModule, parent=None) -> None:
        super().__init__(parent)
        self.module = module
        self._specs: dict[str, ParamSpec] = {}
        self._widgets: dict[str, QWidget] = {}
        self._labels: dict[str, QLabel] = {}
        self._colors: dict[str, str] = {}
        self._height_pending = False
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        groups: dict[str, list[ParamSpec]] = {}
        for spec in self.module.param_spec():
            self._specs[spec.key] = spec
            groups.setdefault(spec.group, []).append(spec)

        self.group_tabs = QTabWidget()
        for group_name, specs in groups.items():
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 14, 4, 6)
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setVerticalSpacing(8)
            form.setHorizontalSpacing(0)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            for spec in specs:
                widget = self._make_widget(spec)
                if spec.tooltip:
                    widget.setToolTip(spec.tooltip)
                label = QLabel(spec.label)
                label.setProperty("role", "muted")
                label.setWordWrap(True)
                if spec.tooltip:
                    label.setToolTip(spec.tooltip)
                self._labels[spec.key] = label
                self._widgets[spec.key] = widget
                form.addRow(label, widget)
            page_layout.addLayout(form)
            page_layout.addStretch(1)
            self.group_tabs.addTab(page, group_name)
            page.installEventFilter(self)
        root.addWidget(self.group_tabs, 1)
        self.group_tabs.currentChanged.connect(self._queue_height_update)
        self._refresh_enabled()

    def _queue_height_update(self, *_: Any) -> None:
        if not self._height_pending:
            self._height_pending = True
            QTimer.singleShot(0, self._fit_current_group)

    def _fit_current_group(self) -> None:
        self._height_pending = False
        page = self.group_tabs.currentWidget()
        if page is None:
            return
        page.layout().activate()
        content_height = page.layout().sizeHint().height()
        height = content_height + self.group_tabs.tabBar().sizeHint().height() + 4
        if self.group_tabs.height() != height:
            self.group_tabs.setFixedHeight(height)
        if self.height() != height:
            self.setFixedHeight(height)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.LayoutRequest, QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._queue_height_update()
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------
    def _make_widget(self, spec: ParamSpec) -> QWidget:
        kind = spec.kind
        if kind == "choice":
            combo = ScrollComboBox()
            for value, label in spec.choices or []:
                combo.addItem(str(label), value)
            index = combo.findData(spec.default)
            combo.setCurrentIndex(max(0, index))
            combo.currentIndexChanged.connect(self._on_change)
            return combo

        if kind == "int":
            spin = ScrollSpinBox()
            spin.setRange(int(spec.minimum), int(spec.maximum))
            spin.setSingleStep(max(1, int(spec.step)))
            spin.setSuffix(spec.suffix)
            spin.setValue(int(spec.default or 0))
            spin.valueChanged.connect(self._on_change)
            return spin

        if kind == "float":
            spin = ScrollDoubleSpinBox()
            spin.setRange(float(spec.minimum), float(spec.maximum))
            spin.setSingleStep(float(spec.step))
            spin.setDecimals(int(spec.decimals))
            spin.setSuffix(spec.suffix)
            spin.setValue(float(spec.default or 0.0))
            spin.valueChanged.connect(self._on_change)
            return spin

        if kind == "bool":
            check = QCheckBox()
            check.setChecked(bool(spec.default))
            check.stateChanged.connect(self._on_change)
            return check

        if kind == "color":
            hex_value = str(spec.default or "#ffffff")
            self._colors[spec.key] = hex_value
            button = QPushButton(hex_value.upper())
            button.setObjectName("colorButton")
            button.clicked.connect(lambda _=False, key=spec.key: self._pick_color(key))
            self._apply_color_style(button, hex_value)
            return button

        if kind == "path":
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            edit = QLineEdit(str(spec.default or ""))
            edit.setPlaceholderText("留空 = 自动")
            edit.textChanged.connect(self._on_change)
            browse = QPushButton("…")
            browse.setFixedWidth(32)
            browse.clicked.connect(lambda _=False, key=spec.key: self._pick_dir(key))
            row.addWidget(edit, 1)
            row.addWidget(browse)
            container.setProperty("edit", edit)
            return container

        edit = QLineEdit(str(spec.default or ""))
        edit.textChanged.connect(self._on_change)
        return edit

    def _apply_color_style(self, button: QPushButton, color: str) -> None:
        qcolor = QColor(color)
        text = "#000000" if qcolor.lightness() > 140 else "#ffffff"
        button.setStyleSheet(
            f"QPushButton {{background:{color}; color:{text};"
            " border:1px solid #e5e5e7; border-radius:6px; padding:6px 10px;}"
            f"QPushButton:disabled {{background:{color}; color:#8b8f96;"
            " border:1px dashed #d9d9dc;}"
        )
        button.setText(color.upper())

    def _pick_color(self, key: str) -> None:
        current = QColor(self._colors.get(key, "#ffffff"))
        chosen = QColorDialog.getColor(current, self, "选择颜色")
        if chosen.isValid():
            self._colors[key] = chosen.name()
            button = self._widgets.get(key)
            if isinstance(button, QPushButton):
                self._apply_color_style(button, chosen.name())
            self._on_change()

    def _pick_dir(self, key: str) -> None:
        container = self._widgets.get(key)
        edit = container.property("edit") if container is not None else None
        start = edit.text().strip() if isinstance(edit, QLineEdit) else ""
        chosen = QFileDialog.getExistingDirectory(self, "选择输出目录", start)
        if chosen and isinstance(edit, QLineEdit):
            edit.setText(chosen)

    # ------------------------------------------------------------------
    def _on_change(self, *_: Any) -> None:
        self._refresh_enabled()
        self.valueChanged.emit()

    def _current_value(self, key: str) -> Any:
        spec = self._specs.get(key)
        widget = self._widgets.get(key)
        if spec is None or widget is None:
            return None
        if spec.kind == "choice" and isinstance(widget, QComboBox):
            return widget.currentData()
        if spec.kind == "int" and isinstance(widget, QSpinBox):
            return widget.value()
        if spec.kind == "float" and isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if spec.kind == "bool" and isinstance(widget, QCheckBox):
            return widget.isChecked()
        if spec.kind == "color":
            return self._colors.get(key, spec.default)
        if spec.kind == "path":
            edit = widget.property("edit")
            return edit.text() if isinstance(edit, QLineEdit) else ""
        if isinstance(widget, QLineEdit):
            return widget.text()
        return spec.default

    def _refresh_enabled(self) -> None:
        for key, spec in self._specs.items():
            if not spec.depends_on:
                continue
            other_key, condition = spec.depends_on
            value = self._current_value(other_key)
            if isinstance(condition, str):
                if condition == "nonzero":
                    enabled = value not in (0, 0.0, False, "", None)
                elif condition == "truthy":
                    enabled = bool(value)
                else:
                    enabled = str(value) == condition
            elif isinstance(condition, (list, tuple, set)):
                enabled = str(value) in {str(c) for c in condition}
            else:
                enabled = bool(value)

            widget = self._widgets.get(key)
            label = self._labels.get(key)
            if widget is not None:
                widget.setEnabled(bool(enabled))
                widget.setVisible(bool(enabled))
            if label is not None:
                label.setEnabled(bool(enabled))
                label.setVisible(bool(enabled))
        self._queue_height_update()

    # ------------------------------------------------------------------
    def values(self) -> dict[str, Any]:
        return {key: self._current_value(key) for key in self._specs}

    def set_values(self, values: dict[str, Any]) -> None:
        for key, value in (values or {}).items():
            spec = self._specs.get(key)
            widget = self._widgets.get(key)
            if spec is None or widget is None or value is None:
                continue
            if spec.kind == "choice" and isinstance(widget, QComboBox):
                index = widget.findData(value)
                if index < 0:
                    index = widget.findData(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif spec.kind == "int" and isinstance(widget, QSpinBox):
                widget.setValue(int(float(value)))
            elif spec.kind == "float" and isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif spec.kind == "bool" and isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif spec.kind == "color":
                self._colors[key] = str(value)
                if isinstance(widget, QPushButton):
                    self._apply_color_style(widget, str(value))
            elif spec.kind == "path":
                edit = widget.property("edit")
                if isinstance(edit, QLineEdit):
                    edit.setText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
        self._refresh_enabled()
        self.valueChanged.emit()

    def default_values(self) -> dict[str, Any]:
        return self.module.defaults()

    def set_choices(self, key: str, choices: list[tuple[Any, str]]) -> None:
        """刷新某个下拉框的候选项（保留当前选中值）。"""
        widget = self._widgets.get(key)
        if not isinstance(widget, QComboBox):
            return
        current = widget.currentData()
        widget.blockSignals(True)
        widget.clear()
        for value, label in choices:
            widget.addItem(str(label), value)
        index = widget.findData(current)
        widget.setCurrentIndex(index if index >= 0 else 0)
        widget.blockSignals(False)
