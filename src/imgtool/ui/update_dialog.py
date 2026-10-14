"""像匣更新窗口：发布源、检查、下载、安装与重启。"""

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QTextBrowser, QVBoxLayout

from .. import APP_NAME, __version__
from ..core.updates import UpdateService, is_newer
from .update_worker import UpdateWorker


class UpdateDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.service = UpdateService(window.config)
        self.cfg = None
        self.release = None
        self.zip_path = None
        self.worker = None
        self.setWindowTitle(f"{APP_NAME} · 检查更新")
        self.setWindowIcon(window.windowIcon())
        self.resize(570, 490)
        self.setMinimumSize(500, 420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)
        title = QLabel("软件更新")
        title.setProperty("role", "title")
        layout.addWidget(title)
        self.version_label = QLabel(f"当前版本 v{__version__}")
        self.version_label.setProperty("role", "muted")
        layout.addWidget(self.version_label)
        source_label = QLabel("发布仓库（GitHub）")
        source_label.setProperty("role", "muted")
        layout.addWidget(source_label)
        source_row = QHBoxLayout()
        self.repository = QLineEdit(self.service.repository())
        self.repository.setPlaceholderText("GitHub 用户名/imagebox 或仓库链接")
        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.save_source)
        source_row.addWidget(self.repository, 1)
        source_row.addWidget(self.save_button)
        layout.addLayout(source_row)
        self.status = QLabel("点击「检查更新」获取最新版本。" if self.repository.text() else "尚未配置更新源，请填写发布仓库后保存。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.notes = QTextBrowser()
        self.notes.setPlainText("更新说明会显示在这里。")
        layout.addWidget(self.notes, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.mode_hint = QLabel("更新时保留配置、模型和输出文件。")
        self.mode_hint.setProperty("role", "muted")
        self.mode_hint.setWordWrap(True)
        layout.addWidget(self.mode_hint)
        buttons = QHBoxLayout()
        self.browser_button = QPushButton("打开发布页")
        self.browser_button.setEnabled(bool(self.repository.text()))
        self.browser_button.clicked.connect(self.open_release_page)
        self.check_button = QPushButton("重新检查")
        self.check_button.clicked.connect(self.check)
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.reject)
        self.action_button = QPushButton("检查更新")
        self.action_button.setObjectName("primary")
        self.action_button.clicked.connect(self.action)
        for button in (self.browser_button, self.check_button, self.close_button, self.action_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)

    def save_source(self) -> bool:
        try:
            self.cfg = self.service.save_repository(self.repository.text())
        except Exception as exc:
            self.status.setText(str(exc))
            return False
        self.repository.setText(self.cfg.slug)
        self.version_label.setText(f"当前版本 v{__version__}")
        self.release = self.zip_path = None
        self.window._available_update = None
        self.browser_button.setEnabled(True)
        self.action_button.setText("检查更新")
        self.notes.setPlainText("更新说明会显示在这里。")
        self.status.setText("更新源已保存，可以检查更新。")
        return True

    def open_release_page(self) -> None:
        try:
            cfg = self.service.load()
            QDesktopServices.openUrl(QUrl(cfg.releases_page))
        except Exception as exc:
            self.status.setText(str(exc))

    def _busy(self, value: bool) -> None:
        for widget in (self.repository, self.save_button, self.check_button, self.action_button):
            widget.setEnabled(not value)
        self.close_button.setText("取消" if value else "关闭")

    def _start(self, release=None) -> None:
        worker = UpdateWorker(self.service, self.cfg, release, self.window)
        self.worker = worker
        self.window.register_update_worker(worker)
        worker.done.connect(self._checked if release is None else self._downloaded)
        worker.failed.connect(self._failed)
        worker.progress.connect(self._progress)
        worker.finished.connect(lambda w=worker: self._finished(w))
        self._busy(True)
        worker.start()

    def check(self) -> None:
        if self.worker is not None:
            return
        if self.repository.text().strip() != self.service.repository():
            if not self.save_source():
                return
        try:
            self.cfg = self.service.load()
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.release = self.zip_path = None
        self.action_button.setText("检查更新")
        self.version_label.setText(f"当前版本 v{__version__}")
        self.progress.setValue(0)
        self.status.setText("正在检查更新…")
        self._start()

    def _checked(self, release) -> None:
        if self.window._closing:
            return
        if not is_newer(release.version, __version__):
            self.status.setText(f"当前已是最新版本 v{__version__}。")
            self.notes.setPlainText(release.notes or "该版本没有填写更新说明。")
            return
        self.set_release(release)

    def set_release(self, release) -> None:
        self.release = release
        self.version_label.setText(f"当前 v{__version__}  →  新版 v{release.version}")
        self.notes.setPlainText(release.notes or "该版本没有填写更新说明。")
        self.status.setText(f"发现新版 v{release.version} · 更新包 {release.asset_size / 1024 / 1024:.1f} MB")
        self.action_button.setText("下载更新包")

    def action(self) -> None:
        if self.worker is not None or self.window._closing:
            return
        if self.zip_path is not None:
            self.install()
        elif self.release is not None:
            try:
                self.cfg = self.service.load()
            except Exception as exc:
                self.status.setText(str(exc))
                return
            self.progress.setValue(0)
            self.status.setText("正在下载更新包…")
            self._start(self.release)
        else:
            self.check()

    def _progress(self, done: int, total: int) -> None:
        if self.window._closing:
            return
        if total > 0:
            self.progress.setValue(min(100, int(done * 100 / total)))
        self.status.setText(f"正在下载 · {done / 1024 / 1024:.1f} MB" + (f" / {total / 1024 / 1024:.1f} MB" if total else ""))

    def _downloaded(self, path) -> None:
        if self.window._closing:
            return
        self.zip_path = Path(path)
        self.progress.setValue(100)
        import sys
        if getattr(sys, "frozen", False):
            self.action_button.setText("安装并重启")
            self.status.setText("更新包已验证，点击「安装并重启」开始更新。")
        else:
            self.action_button.setText("打开下载目录")
            self.status.setText("更新包已下载。源码模式不自动替换项目，请使用发布版安装更新。")

    def _failed(self, message) -> None:
        if not self.window._closing:
            self.status.setText(f"更新失败：{message}")

    def _finished(self, worker) -> None:
        if self.worker is worker:
            self.worker = None
        if not self.window._closing:
            self._busy(False)
            if worker.cancelled:
                self.status.setText("已取消更新操作。")

    def install(self) -> None:
        import sys
        if not getattr(sys, "frozen", False):
            from ..core.io_utils import open_in_explorer
            open_in_explorer(self.zip_path.parent)
            return
        if self.window._running_workers():
            self.status.setText("请先结束图片处理和模型下载，再安装更新。")
            return
        for index in range(self.window.stack.count()):
            page = self.window.stack.widget(index)
            if any(s.dirty for s in getattr(page, "sessions", {}).values()):
                self.status.setText("还有未保存的涂抹，请保存后再安装更新。")
                return
        try:
            self.service.install(self.zip_path, self.cfg)
        except Exception as exc:
            self.status.setText(f"无法安装更新：{exc}")
            return
        self.accept()
        self.window.close()

    def reject(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
        super().reject()
