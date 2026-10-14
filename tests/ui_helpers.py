"""UI 测试共用的小工具：启动离屏 Qt、发送真实鼠标事件、等待后台任务。"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt" and not os.environ.get("QT_QPA_FONTDIR"):
    # 离屏渲染时 Qt 不会自动加载系统字体，这里显式指过去，截图才有中文
    os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_app():
    """创建离屏 QApplication 并套上主题。"""
    from PySide6.QtWidgets import QApplication

    from imgtool.ui import theme

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.STYLESHEET)
    return app


def make_window(output_dir: Path):
    from imgtool.core.config import AppConfig
    from imgtool.ui.app import MainWindow

    config = AppConfig.load()
    config.set("output_dir", str(output_dir))
    window = MainWindow(config)
    window.resize(1280, 820)
    window.show()
    return window


def send_mouse(app, target, etype, pos, pressed: bool = True, button=None) -> None:
    """往控件发一个真实 QMouseEvent（pos 是控件本地坐标）。

    Qt 的约定：``button()`` 是「本次事件涉及的按键」（松开事件里就是被松开的那个），
    ``buttons()`` 是「当前仍按住的所有键」。
    """
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    released = etype == QEvent.Type.MouseButtonRelease
    held = button or Qt.MouseButton.LeftButton
    event = QMouseEvent(
        etype,
        QPointF(pos),
        QPointF(target.mapToGlobal(pos)),
        held,
        Qt.MouseButton.NoButton if released else held,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(target, event)
    app.processEvents()


def drag_on_scene(app, view, points) -> None:
    """按场景坐标（=图片像素坐标）在视图上画一条折线。"""
    from PySide6.QtCore import QEvent, QPointF

    viewport = view.viewport()
    offset = viewport.pos()  # viewport 相对 view 的偏移（边框）
    local = [view.mapFromScene(QPointF(x, y)) - offset for x, y in points]
    send_mouse(app, viewport, QEvent.Type.MouseMove, local[0])
    send_mouse(app, viewport, QEvent.Type.MouseButtonPress, local[0])
    for point in local[1:]:
        send_mouse(app, viewport, QEvent.Type.MouseMove, point)
    send_mouse(app, viewport, QEvent.Type.MouseButtonRelease, local[-1], pressed=False)
    app.processEvents()


def wait_for_worker(app, page, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        worker = getattr(page, "_worker", None)
        if worker is not None and not worker.isRunning():
            app.processEvents()
            return True
        time.sleep(0.05)
    return False
