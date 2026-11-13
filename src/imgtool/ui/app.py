"""主窗口：左侧模块列表 + 右侧模块页面。"""

from __future__ import annotations

import sys
from typing import Any, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QPushButton,
    QWidget,
)

from ..core import io_utils
from ..core.config import AppConfig
from ..core.device import environment_report
from ..core.log import get_logger, setup_logging
from ..core.registry import discover_modules
from .. import APP_NAME, __version__
from . import theme
from .branding import app_icon, brand_icon, configure_taskbar_identity
from .dnd import FileDropRelay, event_paths
from .pages.matting_page import MattingPage
from .pages.module_page import ModulePage
from .pages.placeholder_page import PlaceholderPage
from .pages.superres_page import SuperresPage

log = get_logger("app")

WINDOW_TITLE = f"{APP_NAME} · 图片处理工具箱"


class MainWindow(QMainWindow):
    def __init__(self, config: Optional[AppConfig] = None, parent=None) -> None:
        super().__init__(parent)
        self.config = config or AppConfig.load()
        self.modules = discover_modules()
        self._closing = False
        self._update_workers = set()
        self._update_dialog = None
        self._available_update = None
        self._close_timer = QTimer(self)
        self._close_timer.setInterval(100)
        self._close_timer.timeout.connect(self._finish_close)
        QApplication.instance().setStyle("Fusion")

        self.setWindowTitle(f"{WINDOW_TITLE}  v{__version__}")
        self.setWindowIcon(app_icon())
        self.resize(1380, 900)
        self.setMinimumSize(1080, 720)
        self._build()
        self._build_menu()
        self._setup_drag_drop()
        self.statusBar().showMessage(self._startup_status())
        if self.config.get("check_updates_on_startup", True):
            QTimer.singleShot(1500, self.check_updates_silently)

    @staticmethod
    def _startup_status() -> str:
        return f"Python {sys.version.split()[0]} | 运行环境：帮助 → 运行环境"

    # ------------------------------------------------------------------
    def _build(self) -> None:
        central = QWidget()
        central.setObjectName("shell")
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 左侧模块列表
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(208)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 24, 12, 18)
        side_layout.setSpacing(8)
        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        self.brand_mark = QLabel()
        self.brand_mark.setFixedSize(36, 36)
        self.brand_mark.setPixmap(brand_icon().pixmap(36, 36))
        self.brand_mark.setToolTip(WINDOW_TITLE)
        brand_row.addWidget(self.brand_mark)
        brand = QLabel(APP_NAME)
        brand.setProperty("role", "brand")
        brand_row.addWidget(brand, 1)
        side_layout.addLayout(brand_row)
        subtitle = QLabel("本地图片工作台")
        subtitle.setProperty("role", "muted")
        side_layout.addWidget(subtitle)
        side_layout.addSpacing(30)
        label = QLabel("处理工具")
        label.setProperty("role", "muted")
        side_layout.addWidget(label)
        self.module_list = QListWidget()
        side_layout.addWidget(self.module_list)
        environment = QPushButton("运行环境与设备")
        environment.setProperty("kind", "text")
        environment.clicked.connect(self.show_environment)
        side_layout.addWidget(environment)
        self.btn_update = QPushButton("检查更新")
        self.btn_update.setProperty("kind", "text")
        self.btn_update.clicked.connect(self.show_updates)
        side_layout.addWidget(self.btn_update)
        footer = QLabel(f"v{__version__}   ·   本地处理")
        footer.setProperty("role", "muted")
        side_layout.addWidget(footer)
        layout.addWidget(side)

        # 右侧页面
        self.stack = QStackedWidget()
        for module in self.modules:
            page = self._create_page(module)
            self.stack.addWidget(page)
            item = QListWidgetItem(f"{module.icon}  {module.title}")
            item.setData(Qt.ItemDataRole.UserRole, module.name)
            if module.placeholder:
                item.setText(f"{module.icon}  {module.title}（开发中）")
                item.setForeground(Qt.GlobalColor.gray)
            else:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.module_list.addItem(item)

        self.module_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        if self.modules:
            self.module_list.setCurrentRow(0)

        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

    def _create_page(self, module) -> QWidget:
        if module.placeholder:
            return PlaceholderPage(module, self.config)
        if module.name == "matting":
            page: QWidget = MattingPage(module, self.config)
        elif module.name == "superres":
            page = SuperresPage(module, self.config)
        else:
            page = ModulePage(module, self.config)
        drag_active = getattr(page, "dragActive", None)
        if drag_active is not None:
            drag_active.connect(self._on_drag_active)
        return page

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        act_output = QAction("打开输出目录", self)
        act_output.triggered.connect(self.open_output_directory)
        file_menu.addAction(act_output)
        act_models = QAction("打开模型目录", self)
        act_models.triggered.connect(lambda: io_utils.open_in_explorer(self.config.models_dir))
        file_menu.addAction(act_models)
        file_menu.addSeparator()
        act_quit = QAction("退出", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        help_menu = menubar.addMenu("帮助")
        act_env = QAction("运行环境", self)
        act_env.triggered.connect(self.show_environment)
        help_menu.addAction(act_env)
        act_update = QAction("检查更新", self)
        act_update.triggered.connect(self.show_updates)
        help_menu.addAction(act_update)
        act_about = QAction("关于", self)
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    def _setup_drag_drop(self) -> None:
        """让整个窗口都能接收拖进来的图片/文件夹。

        挂在窗口上的中继负责「其余任何位置」：Qt 在派发拖放时会从鼠标下的控件
        往上找第一个接受拖放的控件，找不到更深的就会落到窗口本身。
        """
        self.drop_relay = FileDropRelay(self, self)
        self.drop_relay.dropped.connect(self.import_files)
        self.drop_relay.dragActive.connect(self._on_drag_active)

    def _on_drag_active(self, active: bool) -> None:
        if active:
            self.statusBar().showMessage("松开鼠标即可导入图片 / 文件夹")
        else:
            self.statusBar().showMessage(self._startup_status())

    def import_files(self, paths: list[str]) -> int:
        """把文件/文件夹导入到当前模块页（占位模块会自动切到可用的模块）。"""
        if not paths or self._closing:
            return 0
        page = self.stack.currentWidget()
        if not hasattr(page, "add_paths"):
            for row in range(self.stack.count()):
                candidate = self.stack.widget(row)
                if hasattr(candidate, "add_paths"):
                    self.module_list.setCurrentRow(row)
                    page = candidate
                    break
        if not hasattr(page, "add_paths"):
            self.statusBar().showMessage("当前模块不支持导入图片", 5000)
            return 0
        added = int(page.add_paths(paths))
        if added:
            self.statusBar().showMessage(f"已导入 {added} 张图片，共 {len(page.items)} 张", 6000)
        else:
            self.statusBar().showMessage("没有找到可导入的图片（支持 png/jpg/webp/bmp/tif/gif）", 6000)
        return added

    # 兜底：正常由 FileDropRelay 处理，这里保证窗口自己收到事件时也能导入
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event_paths(event):
            self._on_drag_active(True)
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event_paths(event):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = event_paths(event)
        if paths:
            event.acceptProposedAction()
            self.import_files(paths)

    # ------------------------------------------------------------------
    def show_environment(self) -> None:
        report = environment_report()
        lines = [
            f"Python：{report['python']}（{report['platform']}）",
            f"Torch CUDA：{'可用' if report['cuda'] else '不可用'}　{report['cuda_note']}",
            f"torch：{report.get('torch') or '未安装'}",
            f"onnxruntime：{report.get('onnxruntime') or '未安装'}",
            f"执行提供者：{', '.join(report.get('onnx_providers') or []) or '-'}",
            "",
            f"项目目录：{self.config.root}",
            f"模型目录：{self.config.models_dir}",
            f"输出目录：{self.config.output_dir}",
        ]
        QMessageBox.information(self, "运行环境", "\n".join(lines))

    def show_about(self) -> None:
        QMessageBox.information(
            self,
            "关于",
            f"{WINDOW_TITLE} v{__version__}\n\n"
            "本地图片处理：抠图、主体标记、画笔精修与批处理。\n"
            "通过「帮助 → 检查更新」获取新版本。",
        )

    def open_output_directory(self) -> None:
        page = self.stack.currentWidget()
        if isinstance(page, ModulePage):
            page.open_output_dir()
        else:
            io_utils.open_in_explorer(self.config.output_dir)

    def current_module_name(self) -> Optional[str]:
        row = self.module_list.currentRow()
        if row < 0:
            return None
        item = self.module_list.item(row)
        return item.data(Qt.ItemDataRole.UserRole)

    def select_module(self, name: str) -> bool:
        for row in range(self.module_list.count()):
            if self.module_list.item(row).data(Qt.ItemDataRole.UserRole) == name:
                self.module_list.setCurrentRow(row)
                return True
        return False

    def closeEvent(self, event) -> None:  # noqa: N802
        self._closing = True
        workers = self._running_workers()
        if workers:
            event.ignore()
            self.statusBar().showMessage("正在结束处理与下载，任务结束后自动关闭…")
            for worker in workers:
                worker.cancel()
            self._close_timer.start()
            return
        self._close_timer.stop()
        if self._update_dialog is not None:
            self._update_dialog.hide()
        for index in range(self.stack.count()):
            cache = getattr(self.stack.widget(index), "result_cache", None)
            if cache is not None:
                cache.close()
        for module in self.modules:
            backend = getattr(module, "_backend", None)
            if backend is not None:
                backend.close()
            if hasattr(module, "_prediction_cache"):
                module._prediction_cache = None
        super().closeEvent(event)

    def _running_workers(self) -> list:
        workers = [w for w in self._update_workers if w.isRunning()]
        for index in range(self.stack.count()):
            page = self.stack.widget(index)
            for name in ("_worker", "_download_worker"):
                worker = getattr(page, name, None)
                if worker is not None and worker.isRunning():
                    workers.append(worker)
        return workers

    def register_update_worker(self, worker) -> None:
        self._update_workers.add(worker)
        worker.finished.connect(lambda w=worker: self._update_workers.discard(w))
        worker.finished.connect(worker.deleteLater)

    def show_updates(self) -> None:
        if self._closing:
            return
        from .update_dialog import UpdateDialog
        if self._update_dialog is None:
            self._update_dialog = UpdateDialog(self)
            if self._available_update is not None:
                self._update_dialog.set_release(self._available_update)
        self._update_dialog.show()
        self._update_dialog.raise_()
        self._update_dialog.activateWindow()

    def check_updates_silently(self) -> None:
        if self._closing:
            return
        if self._update_dialog is not None and self._update_dialog.isVisible():
            return
        from ..core.updates import UpdateService, is_newer
        from .update_worker import UpdateWorker
        service = UpdateService(self.config)
        try:
            cfg = service.load()
        except Exception:
            return
        worker = UpdateWorker(service, cfg, parent=self)
        self.register_update_worker(worker)
        def available(release):
            if not self._closing and is_newer(release.version, __version__):
                self._available_update = release
                self.btn_update.setText(f"发现新版 v{release.version}")
                self.statusBar().showMessage(f"发现像匣新版 v{release.version}，点击左侧检查更新。", 15000)
        worker.done.connect(available)
        worker.failed.connect(lambda msg: log.info("静默更新检查：%s", msg))
        worker.start()

    def _finish_close(self) -> None:
        if not self._running_workers():
            self.close()


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    module_name = None
    if "--module" in argv:
        index = argv.index("--module")
        if index + 1 < len(argv):
            module_name = argv[index + 1]
        del argv[index : index + 2]

    configure_taskbar_identity()
    app = QApplication(argv)
    app.setApplicationName(WINDOW_TITLE)
    app.setWindowIcon(app_icon())
    app.setStyleSheet(theme.STYLESHEET)

    config = AppConfig.load()
    setup_logging(
        str(config.get("log_level", "INFO")),
        config.logs_dir / f"imgtool_{io_utils.timestamp()}.log",
    )

    window = MainWindow(config)
    if module_name:
        window.select_module(module_name)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
