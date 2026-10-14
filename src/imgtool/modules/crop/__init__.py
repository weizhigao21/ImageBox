"""裁剪模块（规划中，占位）。"""

from ...core.base import PlaceholderModule

MODULE = PlaceholderModule(
    name="crop",
    title="裁剪",
    icon="⬚",
    description="批量裁剪、按比例/尺寸裁切、智能主体裁剪、白边自动去除。",
)
MODULE.order = 30

__all__ = ["MODULE"]
