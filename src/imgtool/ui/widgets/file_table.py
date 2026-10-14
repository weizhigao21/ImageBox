"""文件列表（文件名 / 尺寸 / 状态 / 耗时）。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

from ...core.types import ImageItem


class FileTable(QTableWidget):
    """展示待处理/已处理文件，按行选中。"""

    currentRowChanged = Signal(int)

    HEADERS = ["文件", "尺寸", "状态", "耗时"]

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(False)
        self.setWordWrap(False)
        self.setShowGrid(False)
        self.verticalHeader().setDefaultSectionSize(42)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(1, 96)
        self.setColumnWidth(2, 84)
        self.setColumnWidth(3, 72)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._rows: dict[str, int] = {}
        self.itemSelectionChanged.connect(self._emit_selection)

    # ------------------------------------------------------------------
    def _emit_selection(self) -> None:
        self.currentRowChanged.emit(self.currentRow())

    def set_compact(self, compact: bool) -> None:
        """窄任务栏保留文件、状态和耗时，尺寸数据继续存储。"""
        self.setColumnHidden(1, compact)
        self.setColumnWidth(2, 70 if compact else 84)
        self.setColumnWidth(3, 56 if compact else 72)
        self.horizontalHeader().setMinimumSectionSize(40)

    def set_items(self, items: Iterable[ImageItem]) -> None:
        self.setRowCount(0)
        self._rows.clear()
        for item in items:
            self.append_item(item)

    def append_item(self, item: ImageItem) -> int:
        row = self.rowCount()
        self.insertRow(row)
        name = QTableWidgetItem(item.name)
        name.setToolTip(str(item.path))
        self.setItem(row, 0, name)
        self.setItem(row, 1, QTableWidgetItem("-"))
        self.setItem(row, 2, QTableWidgetItem("待处理"))
        self.setItem(row, 3, QTableWidgetItem("-"))
        self._rows[str(item.path)] = row
        return row

    def row_for(self, path: str | Path) -> Optional[int]:
        return self._rows.get(str(path))

    def set_size(self, path: str | Path, text: str) -> None:
        row = self.row_for(path)
        if row is not None:
            self.item(row, 1).setText(text)

    def set_status(self, path: str | Path, status: str, elapsed: float | None = None) -> None:
        row = self.row_for(path)
        if row is None:
            return
        self.item(row, 2).setText(status)
        self.item(row, 2).setToolTip(status)
        if elapsed is not None:
            self.item(row, 3).setText(f"{elapsed:.2f}s")
        color = {
            "完成": "#83907c",
            "失败": "#a2462e",
            "处理中": "#e0651c",
            "已涂抹": "#e0651c",
            "已标记": "#e0651c",
            "已取消": "#85858a",
        }.get(status, "#6b6b6e")
        from PySide6.QtGui import QColor
        self.item(row, 2).setForeground(QColor(color))

    def selected_paths(self) -> list[Path]:
        rows = sorted({index.row() for index in self.selectedIndexes()})
        out: list[Path] = []
        for row in rows:
            cell = self.item(row, 0)
            if cell is not None:
                out.append(Path(cell.toolTip()))
        return out

    def current_path(self) -> Optional[Path]:
        row = self.currentRow()
        if row < 0:
            return None
        cell = self.item(row, 0)
        return Path(cell.toolTip()) if cell is not None else None
