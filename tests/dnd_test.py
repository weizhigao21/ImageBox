"""拖放导入测试。

说明：PySide6 里用 Python 手工构造的 QDragEnterEvent，``mimeData()`` 会退化成裸
QObject（真实系统拖放不受影响），所以这里不投递合成事件，而是直接调用
``FileDropRelay.eventFilter``，用一个等价接口的假事件验证我们自己的处理逻辑；
另外检查各处控件确实打开了拖放。

用法：python tests/dnd_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt" and not os.environ.get("QT_QPA_FONTDIR"):
    os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imgtool.core import io_utils  # noqa: E402
from imgtool.core.config import AppConfig  # noqa: E402
from make_sample import make_sample  # noqa: E402

TMP = Path(__file__).resolve().parent / "_tmp" / "dnd"


class FakeDragEvent:
    """只实现 FileDropRelay 用到的接口。"""

    def __init__(self, etype, paths: list[str]) -> None:
        from PySide6.QtCore import QMimeData, QUrl

        self._type = etype
        self._mime = QMimeData()
        self._mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        self.accepted = False

    def type(self):  # noqa: A003
        return self._type

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self) -> None:
        self.accepted = True


def main() -> int:
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    from imgtool.ui import theme
    from imgtool.ui.app import MainWindow

    app = QApplication([])
    app.setStyleSheet(theme.STYLESHEET)

    config = AppConfig.load()
    config.set("output_dir", str(TMP / "out"))

    window = MainWindow(config)
    window.resize(1280, 820)
    window.show()
    app.processEvents()
    assert window.select_module("matting")
    page = window.stack.currentWidget()

    TMP.mkdir(parents=True, exist_ok=True)
    single = TMP / "one.png"
    rgb, _ = make_sample(480, 360, seed=3)
    io_utils.write_image(single, rgb)

    folder = TMP / "album"
    folder.mkdir(exist_ok=True)
    for i in range(2):
        io_utils.write_image(folder / f"pic_{i}.png", rgb)

    ok = True

    # ---------------------------------------------------------------- 1. 控件是否打开了拖放
    print("== 拖放开关 ==")
    flags = {
        "主窗口": window.acceptDrops(),
        "模块页": page.acceptDrops(),
        "文件列表 viewport": page.table.viewport().acceptDrops(),
        "预览区（原图）": page.views["原图"].acceptDrops(),
        "预览区 viewport": page.views["原图"].viewport().acceptDrops(),
    }
    for name, enabled in flags.items():
        print(f"  {'✅' if enabled else '❌'} {name}: acceptDrops={enabled}")
        ok = ok and enabled

    # ---------------------------------------------------------------- 2. 各落点的处理逻辑
    print("\n== 各落点导入 ==")

    def drop_through(relay, target, paths):
        return relay.eventFilter(target, FakeDragEvent(QEvent.Type.Drop, paths))

    targets = {
        "模块页": (page._drop_relays[0], page),
        "文件列表": (page._drop_relays[1], page.table),
        "预览区": (page._drop_relays[2], page.views["原图"]),
        "主窗口（兜底）": (window.drop_relay, window),
    }
    for name, (relay, target) in targets.items():
        page.clear_items()
        app.processEvents()
        handled = drop_through(relay, target, [str(single)])
        count = len(page.items)
        good = handled and count == 1
        print(f"  {'✅' if good else '❌'} 拖到「{name}」→ 事件已处理={handled}，导入 {count} 张")
        ok = ok and good

    # ---------------------------------------------------------------- 3. 文件夹 / 去重 / 非图片
    print("\n== 边界情况 ==")
    page.clear_items()
    drop_through(window.drop_relay, window, [str(folder)])
    print(f"  {'✅' if len(page.items) == 2 else '❌'} 拖入文件夹 album → 导入 {len(page.items)} 张（应为 2）")
    ok = ok and len(page.items) == 2

    drop_through(window.drop_relay, window, [str(folder)])
    print(f"  {'✅' if len(page.items) == 2 else '❌'} 重复拖同一文件夹 → 仍为 {len(page.items)} 张")
    ok = ok and len(page.items) == 2

    text_file = TMP / "note.txt"
    text_file.write_text("not an image", encoding="utf-8")
    page.clear_items()
    drop_through(window.drop_relay, window, [str(text_file)])
    print(f"  {'✅' if len(page.items) == 0 else '❌'} 拖入非图片文件 → 导入 {len(page.items)} 张（应为 0）")
    ok = ok and len(page.items) == 0

    # ---------------------------------------------------------------- 4. 占位模块上拖放
    page.clear_items()
    assert window.select_module("crop")
    app.processEvents()
    drop_through(window.drop_relay, window, [str(single)])
    switched = window.current_module_name()
    matting_page = window.stack.currentWidget()
    good = switched == "matting" and len(matting_page.items) == 1
    print(f"  {'✅' if good else '❌'} 在「裁剪（开发中）」上拖放 → 当前模块 {switched}，导入 {len(matting_page.items)} 张")
    ok = ok and good

    window.close()
    print("\n拖放测试通过 ✅" if ok else "\n拖放测试失败 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
