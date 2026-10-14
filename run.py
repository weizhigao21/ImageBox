"""启动入口。

    python run.py                  # 打开界面
    python run.py --module matting # 直接进入抠图页
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    if "--verify-runtime" in sys.argv:
        import json
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from imgtool import __version__
        from imgtool.core.config import AppConfig
        from imgtool.core import io_utils
        from imgtool.core.pipeline import Pipeline
        from imgtool.core.types import ImageItem
        import numpy as np
        from imgtool.ui import theme
        from imgtool.ui.app import MainWindow
        index = sys.argv.index("--verify-runtime")
        target = Path(sys.argv[index + 1])
        app = QApplication([])
        app.setStyleSheet(theme.STYLESHEET)
        config = AppConfig.load()
        config.set("check_updates_on_startup", False)
        window = MainWindow(config)
        names = [module.name for module in window.modules]
        assert "matting" in names, "抠图模块未打包"
        assert not window.windowIcon().isNull(), "图标未打包"
        sample = target.parent / "runtime-sample.png"
        rgb = np.full((128, 128, 3), 245, np.uint8)
        rgb[32:96, 32:96] = (30, 70, 110)
        io_utils.write_image(sample, rgb)
        assert io_utils.probe_size(sample) == (128, 128)
        module = next(m for m in window.modules if m.name == "matting")
        result = Pipeline(module, {"backend": "simple", "out_dir": str(target.parent / "runtime-output")}, config).run([ImageItem(sample)])[0]
        assert result.ok, result.message
        window.show_updates()
        app.processEvents()
        report = {"version": __version__, "modules": names, "root": str(config.root),
                  "update_entry": window._update_dialog is not None, "image_processing": result.ok}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        window.close()
        return 0
    from imgtool.ui.app import main as ui_main

    return ui_main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        if "--verify-runtime" in sys.argv:
            import json
            import traceback
            index = sys.argv.index("--verify-runtime")
            Path(sys.argv[index + 1]).write_text(json.dumps({"error": traceback.format_exc()}, ensure_ascii=False), encoding="utf-8")
            raise SystemExit(1)
        raise
