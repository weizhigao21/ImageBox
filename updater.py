"""独立更新程序入口；通过 updater.spec 构建 updater.exe。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pyupdater.updater_main import main

if __name__ == "__main__":
    raise SystemExit(main())
