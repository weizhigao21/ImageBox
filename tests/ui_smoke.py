"""UI 冒烟测试：离屏渲染主窗口并截图，验证界面能正常构建与跑完一次抠图。

用法：python tests/ui_smoke.py [输出截图路径]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt" and not os.environ.get("QT_QPA_FONTDIR"):
    # 离屏渲染时 Qt 不会自动加载系统字体，这里显式指过去，截图才有中文
    os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imgtool.core import io_utils  # noqa: E402
from imgtool.core.config import AppConfig  # noqa: E402
from make_sample import make_sample  # noqa: E402

TMP = Path(__file__).resolve().parent / "_tmp"


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from imgtool.ui import theme
    from imgtool.ui.app import MainWindow

    app = QApplication([])
    app.setStyleSheet(theme.STYLESHEET)

    config = AppConfig.load()
    config.set("output_dir", str(TMP / "ui_out"))

    window = MainWindow(config)
    window.resize(1360, 880)
    window.show()
    app.processEvents()

    # 准备两张示例图
    TMP.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, size in enumerate([(800, 600), (640, 480)]):
        path = TMP / f"ui_sample_{i}.png"
        rgb, _ = make_sample(size[0], size[1], seed=7 + i)
        io_utils.write_image(path, rgb)
        paths.append(str(path))

    assert window.select_module("matting"), "未找到抠图模块"
    page = window.stack.currentWidget()
    page.add_paths(paths)
    page.table.selectRow(0)
    app.processEvents()

    shot_before = str(Path(sys.argv[1]).with_name("ui_before.png")) if len(sys.argv) > 1 else str(TMP / "ui_before.png")
    window.grab().save(shot_before)

    # 跑一次抠图
    page.panel.set_values({"backend": "simple", "output_mode": "all", "max_size": 800})
    page.process_current()

    deadline = time.time() + 120
    while time.time() < deadline:
        app.processEvents()
        worker = getattr(page, "_worker", None)
        if worker is not None and not worker.isRunning():
            break
        time.sleep(0.05)
    app.processEvents()
    time.sleep(0.3)
    app.processEvents()

    assert page.tabs.currentWidget() is page.views["结果"], "处理完成后应自动显示结果"
    result_view = page.views["结果"]
    assert result_view.verticalScrollBar().maximum() == 0, "结果预览应适应窗口，不裁掉上下边缘"
    assert result_view.horizontalScrollBar().maximum() == 0, "结果预览应适应窗口，不裁掉左右边缘"
    assert page.panel.group_tabs.count() == 3, "参数应按基础、边缘、输出分栏"

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else TMP / "ui_after.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(out))

    # 结果 / 遮罩 两个预览页也各截一张，便于确认预览链路
    shots = [out]
    for index, name in ((1, "ui_result.png"), (2, "ui_mask.png")):
        if index < page.tabs.count():
            page.tabs.setCurrentIndex(index)
            app.processEvents()
            time.sleep(0.2)
            app.processEvents()
            extra = out.with_name(name)
            window.grab().save(str(extra))
            shots.append(extra)
    print("截图：", shot_before, *shots)

    results = page.results
    print("处理结果条数：", len(results))
    ok = 0
    for result in results.values():
        print(f"  {result.item.name}: {'完成' if result.ok else '失败'} {result.message}")
        ok += int(result.ok)
    print("状态栏：", page.status.text())

    if ok == 0:
        print("!! 界面未能完成一次抠图")
        return 1
    print("UI 冒烟测试通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
