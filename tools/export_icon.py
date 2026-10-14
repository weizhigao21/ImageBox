"""从 SVG 导出 PNG / Windows 多尺寸 ICO。运行：python tools/export_icon.py。"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication


def main() -> None:
    app = QApplication.instance() or QApplication([])
    assets = Path(__file__).resolve().parents[1] / "src" / "imgtool" / "ui" / "assets"
    renderer = QSvgRenderer(str(assets / "app-icon.svg"))
    if not renderer.isValid():
        raise ValueError("SVG 图标无效")
    image = QImage(1024, 1024, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    png = assets / "app-icon.png"
    if not image.save(str(png)):
        raise OSError(f"无法保存图标：{png}")
    with Image.open(png) as source:
        source.save(assets / "app-icon.ico", format="ICO", sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])
    print("Exported app-icon.png and app-icon.ico")


if __name__ == "__main__":
    main()
