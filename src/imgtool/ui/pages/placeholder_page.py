"""占位模块页：模块规划中时显示。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...core.base import BaseModule


class PlaceholderPage(QWidget):
    """尚未实现的模块。"""

    def __init__(self, module: BaseModule, config: Any = None, parent=None) -> None:
        super().__init__(parent)
        self.module = module
        self.config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)

        title = QLabel(f"{module.icon}  {module.title}")
        title.setProperty("role", "title")
        layout.addWidget(title)

        badge = QLabel("开发中")
        badge.setStyleSheet(
            "background:#f1f1f1; color:#85858a; border-radius:4px; padding:4px 12px;"
        )
        badge.setFixedWidth(80)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(badge)

        desc = QLabel(module.description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#6b6b6e; font-size:14px;")
        layout.addWidget(desc)

        hint = QLabel(
            "功能正在准备中。现在可以使用抠图工具处理图片，\n"
            "也可以直接拖入图片，开始抠图。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280;")
        layout.addWidget(hint)

        layout.addStretch(1)
