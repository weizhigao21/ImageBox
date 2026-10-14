"""可选的 PySide6 界面：现成的「发现新版本 → 下载 → 立即更新」对话框。

只有需要 Qt 的宿主程序才导入本模块（`pyupdater` 核心不依赖 Qt）。
宿主需要提供两样东西：当前版本号、以及"如何真正退出程序"的回调
（很多程序把关闭窗口做成最小化到托盘，直接 quit 会不生效）。

用法：

    from pyupdater.qt import check_updates
    check_updates(self, cfg, current_version=__version__)              # 手动
    check_updates(self, cfg, current_version=__version__, silent=True, # 启动静默
                  on_available=lambda rel: self.set_status(f"有新版本 v{rel.version}"))

线程约定：业务信号命名 `done`，清理只连原生 `finished`。
"""
from __future__ import annotations

import logging
import os
import sys

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from pyupdater.apply import ApplyError, launch_updater
from pyupdater.check import (
    UpdateError,
    download_asset,
    fetch_latest_release,
    is_newer,
)
from pyupdater.config import UpdateConfig

logger = logging.getLogger(__name__)


def is_frozen() -> bool:
    """是否为打包后的可执行程序（源码运行时不支持自我更新）。"""
    return bool(getattr(sys, "frozen", False))


def _format_size(size: int) -> str:
    if size <= 0:
        return "大小未知"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def _alive(widget) -> bool:
    """包装器对应的 C++ 对象是否仍有效（对话框可能已被关闭销毁）。"""
    if widget is None:
        return True
    try:
        import shiboken6

        return shiboken6.isValid(widget)
    except Exception:
        return True


def _fallback_parent():
    """找不到原父窗口时，退回到当前可见的主窗口。"""
    app = QApplication.instance()
    if app is None:
        return None
    from PySide6.QtWidgets import QMainWindow

    for widget in app.topLevelWidgets():
        if isinstance(widget, QMainWindow) and widget.isVisible():
            return widget
    return None


def _default_quit() -> None:
    app = QApplication.instance()
    if app is not None:
        app.quit()


# ---- 后台线程 ----


class UpdateCheckWorker(QThread):
    """后台查询最新版本。"""

    done = Signal(object)   # ReleaseInfo
    failed = Signal(str)

    def __init__(self, cfg: UpdateConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        try:
            release = fetch_latest_release(self.cfg)
        except UpdateError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # 兜底：异常不允许逃逸线程
            logger.exception("检查更新异常")
            self.failed.emit(str(exc))
            return
        self.done.emit(release)


class UpdateDownloadWorker(QThread):
    """后台下载更新包（可取消）。"""

    progress = Signal(int, int)   # (已下载字节, 总字节)
    done = Signal(str)            # zip 路径
    failed = Signal(str)

    def __init__(self, cfg: UpdateConfig, release, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.release = release

    def cancel(self) -> None:
        self.requestInterruption()

    def run(self):
        try:
            path = download_asset(
                self.cfg,
                self.release,
                on_progress=self._emit_progress,
                cancel_check=self.isInterruptionRequested,
            )
        except UpdateError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            logger.exception("下载更新包异常")
            self.failed.emit(str(exc))
            return
        if not self.isInterruptionRequested():
            self.done.emit(str(path))

    def _emit_progress(self, done: int, total: int) -> None:
        if not self.isInterruptionRequested():
            self.progress.emit(done, total)


# ---- 对话框 ----


class UpdateDialog(QDialog):
    """发现新版本：展示说明 → 下载 → 立即更新并重启。"""

    def __init__(
        self,
        release,
        cfg: UpdateConfig,
        current_version: str,
        parent=None,
        quit_callback=None,
    ):
        super().__init__(parent)
        self.release = release
        self.cfg = cfg
        self.current_version = current_version
        self.quit_callback = quit_callback or _default_quit
        self._zip_path: str | None = None
        self._download_worker: UpdateDownloadWorker | None = None

        self.setWindowTitle("发现新版本")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)

        title = QLabel(f"发现新版本 v{release.version}")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(title)

        subtitle = QLabel(
            f"{cfg.app_name}　v{current_version}　→　v{release.version}"
            f"（{_format_size(release.asset_size)}）"
        )
        subtitle.setObjectName("updateSubtitle")
        layout.addWidget(subtitle)

        notes = QTextBrowser()
        notes.setObjectName("updateNotes")
        notes.setOpenExternalLinks(True)
        notes.setMinimumHeight(180)
        notes.setPlainText(str(release.notes or "（该版本没有填写更新说明）"))
        layout.addWidget(notes, 1)

        self.hint_label = QLabel("")
        self.hint_label.setObjectName("updateHint")
        layout.addWidget(self.hint_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        buttons = QHBoxLayout()
        self.browser_btn = QPushButton("在浏览器中打开")
        self.browser_btn.setCursor(Qt.PointingHandCursor)
        self.browser_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(release.page_url))
        )
        buttons.addWidget(self.browser_btn)
        buttons.addStretch()

        self.later_btn = QPushButton("稍后再说")
        self.later_btn.setCursor(Qt.PointingHandCursor)
        self.later_btn.clicked.connect(self.reject)
        buttons.addWidget(self.later_btn)

        self.action_btn = QPushButton("下载更新包")
        self.action_btn.setCursor(Qt.PointingHandCursor)
        self.action_btn.clicked.connect(self._on_action)
        buttons.addWidget(self.action_btn)
        layout.addLayout(buttons)

        if not is_frozen():
            self.hint_label.setText(
                "开发模式下不支持自动更新，请手动替换或到发布页下载。"
            )

    # ---- 交互 ----

    def _on_action(self) -> None:
        if self._zip_path is not None:
            self._apply_update()
        else:
            self._start_download()

    def _start_download(self) -> None:
        self.action_btn.setEnabled(False)
        self.later_btn.setText("取消下载")
        self.progress.setVisible(True)
        self.hint_label.setText("正在下载更新包…")

        worker = UpdateDownloadWorker(self.cfg, self.release, parent=None)
        self._download_worker = worker
        worker.progress.connect(self._on_progress)
        worker.done.connect(self._on_downloaded)
        worker.failed.connect(self._on_download_failed)
        worker.finished.connect(lambda w=worker: self._clear_worker(w))
        worker.finished.connect(worker.deleteLater)
        _register(worker)
        worker.start()

    def _clear_worker(self, worker) -> None:
        if self._download_worker is worker:
            self._download_worker = None

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.progress.setValue(min(100, int(done * 100 / total)))
            self.hint_label.setText(
                f"正在下载更新包… {done / 1024 / 1024:.1f} MB"
                f" / {total / 1024 / 1024:.1f} MB"
            )
        else:
            self.hint_label.setText(f"正在下载更新包… {done / 1024 / 1024:.1f} MB")

    def _on_downloaded(self, path: str) -> None:
        self._zip_path = path
        self.progress.setValue(100)
        self.hint_label.setText("更新包已就绪，点击「立即更新并重启」开始安装。")
        self.later_btn.setText("稍后再说")
        self.action_btn.setText("立即更新并重启")
        self.action_btn.setEnabled(True)

    def _on_download_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.later_btn.setText("稍后再说")
        self.action_btn.setEnabled(True)
        if self.isVisible():
            self.hint_label.setText(f"下载失败：{message}")
            QMessageBox.warning(
                self, "下载失败", f"{message}\n\n可点「在浏览器中打开」手动下载。"
            )

    def _apply_update(self) -> None:
        install_dir = os.path.dirname(os.path.abspath(sys.executable)) if is_frozen() else ""
        if not install_dir or not self._zip_path:
            self.hint_label.setText("无法确定安装目录，请到发布页手动下载。")
            return
        try:
            launch_updater(install_dir, self._zip_path, self.cfg)
        except ApplyError as exc:
            QMessageBox.warning(self, "无法启动更新程序", str(exc))
            return
        logger.info("已启动更新程序，准备退出主程序")
        self.accept()
        self.quit_callback()

    def reject(self) -> None:
        worker = self._download_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(3000)
        super().reject()


_FALLBACK_THREADS: set = set()


def _register(worker: QThread) -> None:
    """持有线程引用直到它结束——线程存活期间不能没人引用它（会被 GC 回收）。"""
    _FALLBACK_THREADS.add(worker)
    worker.finished.connect(lambda w=worker: _FALLBACK_THREADS.discard(w))


# ---- 对外入口 ----


def check_updates(
    parent,
    cfg: UpdateConfig,
    current_version: str,
    silent: bool = False,
    on_available=None,
    quit_callback=None,
) -> UpdateCheckWorker:
    """检查更新（后台线程），返回 worker 便于宿主保存引用。

    Args:
        silent: True 用于启动静默检查——失败不提示，发现新版本也不弹窗，
            只调用 `on_available(release)` 由调用方决定展示方式。
        on_available: silent 模式下发现新版本时的回调。
    """
    worker = UpdateCheckWorker(cfg, parent=None)

    def _on_done(release):
        if not is_newer(release.version, current_version):
            if not silent and _alive(parent):
                QMessageBox.information(
                    parent, "检查更新", f"当前已是最新版本 v{current_version}。"
                )
            return
        if silent:
            if on_available is not None:
                on_available(release)
            return
        owner = parent if _alive(parent) else _fallback_parent()
        UpdateDialog(release, cfg, current_version, owner, quit_callback).exec()

    def _on_failed(message: str):
        logger.info("检查更新失败: %s", message)
        if silent or not _alive(parent):
            return
        box = QMessageBox(parent)
        box.setWindowTitle("检查更新失败")
        box.setIcon(QMessageBox.Warning)
        box.setText(message)
        box.setInformativeText("可在浏览器中打开发布页手动下载完整包。")
        open_btn = box.addButton("打开发布页", QMessageBox.ActionRole)
        box.addButton("关闭", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl(cfg.releases_page))

    worker.done.connect(_on_done)
    worker.failed.connect(_on_failed)
    worker.finished.connect(worker.deleteLater)
    _register(worker)
    worker.start()
    return worker
