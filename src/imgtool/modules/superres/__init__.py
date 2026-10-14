"""超分模块（规划中，占位）。"""

from ...core.base import PlaceholderModule

MODULE = PlaceholderModule(
    name="superres",
    title="超分",
    icon="⤢",
    description="把低分辨率图片放大并补充细节（Real-ESRGAN / SwinIR 等）。",
)
MODULE.order = 20

__all__ = ["MODULE"]
