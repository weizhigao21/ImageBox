# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：独立更新程序（单文件）。

用法（在宿主项目里也用同一份配置即可）：
    pyinstaller updater.spec --noconfirm --clean

单文件的理由：更新程序会被主程序复制到 %TEMP% 单独运行（运行中的 exe
无法原地替换），一个文件最省事；它只依赖标准库（进度窗用 tkinter）+
`pyupdater` 核心，体积小、启动快，不引入 PySide6。

把 `ICON` 换成宿主自己的图标路径（不存在则自动忽略，不会打包失败）。
"""

import os

_ICON_CANDIDATE = "src/imgtool/ui/assets/app-icon.ico"
ICON = [_ICON_CANDIDATE] if os.path.exists(_ICON_CANDIDATE) else None

a = Analysis(
    ["updater.py"],
    pathex=[os.path.abspath("src")],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 更新程序不该拖上任何第三方依赖：排除宿主用到的重型库
    excludes=[
        # 更新程序只做"替换"，不需要联网：requests 全链路都不该进包
        "requests",
        "urllib3",
        "certifi",
        "charset_normalizer",
        "idna",
        "PySide6",
        "PyQt5",
        "PyQt6",
        "shiboken6",
        "lxml",
        "opencc",
        "openai",
        "PIL",
        "Pillow",
        "curl_cffi",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "torch",
        "torchvision",
        "torchaudio",
        "numba",
        "llvmlite",
        "imageio",
        "imageio_ffmpeg",
        "playwright",
        "sklearn",
        "scikit_learn",
        "keras",
        "tensorflow",
        "pywin32",
        "win32",
        "win32com",
        "win32api",
        "win32con",
        "win32gui",
        "Pythonwin",
        "pythonnet",
        "sounddevice",
        "werkzeug",
        "flask",
        "websockets",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="updater",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)
