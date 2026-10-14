"""冒烟测试：不依赖 UI，直接验证 注册表 → 抠图 → 输出 全链路。

用法：
    python tests/smoke_test.py [--backend simple|onnx] [--image 路径]

退出码 0 表示通过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imgtool.core import io_utils  # noqa: E402
from imgtool.core.config import AppConfig  # noqa: E402
from imgtool.core.pipeline import Pipeline  # noqa: E402
from imgtool.core.registry import discover_modules, get_module  # noqa: E402
from imgtool.core.types import ImageItem, TaskContext  # noqa: E402
from make_sample import make_busy_sample, make_sample  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / float(union)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="simple")
    parser.add_argument("--image", default="")
    parser.add_argument("--min-iou", type=float, default=0.85)
    parser.add_argument("--min-busy-iou", type=float, default=0.6)
    args = parser.parse_args()

    ok = True
    print("== 环境 ==")
    from imgtool.core.device import summary_line

    print(summary_line())

    print("\n== 模块注册表 ==")
    modules = discover_modules(force=True)
    for m in modules:
        print(f"  [{'占位' if m.placeholder else '可用'}] {m.name:<10} {m.title}")
    if not any(m.name == "matting" for m in modules):
        print("!! 未发现抠图模块")
        return 1

    module = get_module("matting")
    assert module is not None

    tmp = Path(__file__).resolve().parent / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    out_dir = tmp / "out"
    config = AppConfig.load()
    config.set("output_dir", str(out_dir))
    module.config = config  # type: ignore[attr-defined]

    print(f"\n== 抠图（后端={args.backend}）==")
    if args.image:
        src = Path(args.image)
        gt_mask = None
    else:
        src = tmp / "sample.png"
        rgb, gt_mask = make_sample()
        io_utils.write_image(src, rgb)

    base_params = {
        "backend": args.backend,
        "onnx_model": "auto",
        "device": "auto",
        "max_size": 1600,
        "output_mode": "all",
        "decontaminate": True,
        "out_dir": str(out_dir),
    }
    progress: list[str] = []
    ctx = TaskContext(
        config=config,
        on_progress=lambda d, t, m: progress.append(f"{d}/{t} {m}"),
    )

    pipe = Pipeline(module, base_params, config)
    results = pipe.run([ImageItem(src)], ctx)
    for line in progress:
        print("   ", line)

    res = results[0]
    if not res.ok:
        print(f"!! 处理失败：{res.message}")
        return 1
    print(f"    耗时 {res.elapsed:.2f}s")
    for kind, path in res.outputs.items():
        size_kb = path.stat().st_size / 1024
        print(f"    输出[{kind}] {path.name}  ({size_kb:.0f} KB)")

    alpha = res.item.alpha
    assert alpha is not None
    fg_ratio = float((alpha > 127).mean())
    print(f"    前景占比 {fg_ratio:.1%}")
    if not (0.001 < fg_ratio < 0.995):
        print("!! 前景占比异常（几乎全透明或全不透明）")
        ok = False

    if gt_mask is not None and args.backend == "simple":
        score = iou((alpha > 127).astype(np.uint8), gt_mask)
        print(f"    与真值 IoU = {score:.3f}（阈值 {args.min_iou}）")
        if score < args.min_iou:
            print("!! 抠图精度不达标")
            ok = False

    # ---------------------------------------------------------------- 兜底路径
    if not args.image and args.backend == "simple":
        print("\n== 兜底路径（杂乱背景，走中心椭圆先验）==")
        busy_rgb, busy_gt = make_busy_sample()
        busy_path = tmp / "busy.png"
        io_utils.write_image(busy_path, busy_rgb)
        busy_result = pipe.run([ImageItem(busy_path)], ctx)[0]
        if not busy_result.ok:
            print(f"!! 兜底路径失败：{busy_result.message}")
            ok = False
        else:
            busy_alpha = busy_result.item.alpha
            assert busy_alpha is not None
            busy_ratio = float((busy_alpha > 127).mean())
            busy_iou = iou((busy_alpha > 127).astype(np.uint8), busy_gt)
            print(f"    耗时 {busy_result.elapsed:.2f}s　前景占比 {busy_ratio:.1%}　IoU {busy_iou:.3f}")
            print(f"    输出：{', '.join(p.name for p in busy_result.outputs.values())}")
            if not (0.01 < busy_ratio < 0.99):
                print("!! 兜底路径结果退化")
                ok = False
            if busy_iou < args.min_busy_iou:
                print(f"!! 兜底路径精度不足（< {args.min_busy_iou}）")
                ok = False

    # ---------------------------------------------------------------- 后端可用性
    print("\n== 后端状态 ==")
    from imgtool.modules.matting.backends import backend_status

    status = backend_status(config)
    for name, label, available, reason in status:
        print(f"  [{'可用' if available else '不可用'}] {name:<10} {label}" + (f"　{reason}" if reason else ""))

    onnx_ok = any(name == "onnx" and available for name, _l, available, _r in status)
    if not onnx_ok:
        onnx_result = Pipeline(module, {**base_params, "backend": "onnx"}, config).run(
            [ImageItem(src)], ctx
        )[0]
        print(f"    onnx 无权重时：ok={onnx_result.ok}　message={onnx_result.message}")
        if onnx_result.ok:
            print("!! 无权重时 onnx 后端不应成功")
            ok = False
        elif "onnx" not in onnx_result.message.lower():
            print("!! 失败提示不清晰（应说明缺少 onnx 权重）")
            ok = False
    else:
        print("    onnx 权重已存在，跳过“缺权重提示”检查")

    print("\n== 结果 ==")
    print("通过 ✅" if ok else "失败 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
