"""图片处理工具箱（imgtool）。

模块化设计：每个图像处理能力都是一个独立模块（modules/<name>），
通过 core.registry 自动发现，共用同一套处理管线（core.pipeline）与 UI 外壳（ui.app）。
"""

__version__ = "0.3.0"
APP_NAME = "像匣"
APP_ID = "ImageBox"
EXE_NAME = "ImageBox.exe"
__all__ = ["__version__", "APP_NAME", "APP_ID", "EXE_NAME"]
