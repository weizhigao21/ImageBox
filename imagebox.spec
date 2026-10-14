# -*- mode: python ; coding: utf-8 -*-
"""像匣 Windows onedir 发布包（简易 + ONNX；BiRefNet 为可选开发后端）。"""
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH)
sys.path.insert(0, str(root / "src"))
a = Analysis(
    [str(root / "run.py")], pathex=[str(root / "src")],
    # PySide6 6.11 的稳定 ABI 依赖 python3.dll，需要与 python310.dll 一起分发。
    binaries=[(str(Path(sys.base_prefix) / "python3.dll"), ".")],
    datas=[(str(root / "src/imgtool/ui/assets"), "imgtool/ui/assets")],
    hiddenimports=collect_submodules("imgtool.modules") + ["PySide6.QtSvg"],
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=["torch", "torchvision", "transformers", "timm", "tensorflow", "matplotlib", "pandas", "scipy", "pyupdater.qt", "pyupdater.updater_main"],
    noarchive=False,
)
# Windows Qt 使用系统 ICU API，排除 PATH 中 Poppler/Conda 的同名版本。
a.binaries = [item for item in a.binaries if item[0].casefold() != "icuuc.dll" and not item[0].casefold().startswith("icudt")]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="ImageBox",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, icon=str(root / "src/imgtool/ui/assets/app-icon.ico"),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ImageBox")
