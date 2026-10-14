"""项目图标：矢量源用于导航，ICO 用于窗口与 Windows 任务栏。"""

from pathlib import Path
import sys

from PySide6.QtGui import QIcon

ASSET_DIR = Path(__file__).resolve().parent / "assets"


def app_icon() -> QIcon:
    icon_path = ASSET_DIR / "app-icon.ico"
    if not icon_path.exists():
        icon_path = ASSET_DIR / "app-icon.svg"
    return QIcon(str(icon_path))


def brand_icon() -> QIcon:
    return QIcon(str(ASSET_DIR / "app-icon.svg"))


def configure_taskbar_identity() -> None:
    """Windows 单独分组此程序，使任务栏使用窗口图标。"""
    if sys.platform == "win32":
        import ctypes
        set_identity = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        set_identity.argtypes = [ctypes.c_wchar_p]
        set_identity.restype = ctypes.c_long
        set_identity("ImageBox.Workbench")
