"""在原图上涂抹「要抠出来的部分」，再据此抠图。

覆盖：
1. postprocess.apply_foreground_hint / apply_hint_subject 单元测试；
2. UI 链路 —— 在原图页涂抹 → 叠加绿色标记 → 「处理当前」；
3. 两种标记含义：
   - subject（默认）：涂一笔即可抠出**整只主体**，轮廓是主体自身的（回归：曾经等于矩形裁剪 + 带背景）
   - region：严格只抠涂抹到的范围；
4. 清除标记后恢复全自动。

用法：python tests/fg_hint_test.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ui_helpers import drag_on_scene, make_app, make_window, wait_for_worker

from imgtool.core import io_utils  # noqa: E402
from imgtool.modules.matting import postprocess  # noqa: E402
from imgtool.modules.matting.backends.simple_backend import SimpleBackend  # noqa: E402
from make_sample import make_sample  # noqa: E402

TMP = Path(__file__).resolve().parent / "_tmp" / "fg_hint"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 127
    b = b > 0
    return float(np.logical_and(a, b).sum()) / float(max(1, np.logical_or(a, b).sum()))


def test_postprocess() -> bool:
    print("== 后处理单元测试 ==")
    ok = True
    alpha = np.full((100, 100), 1.0, np.float32)
    hint = np.zeros((100, 100), np.uint8)
    hint[40:60, 40:60] = 255

    out = postprocess.apply_foreground_hint(alpha, hint, expand=0, feather=1.0)
    unit_ok = out[50, 50] > 0.95 and out[10, 10] < 0.02 and out[50, 39] < 0.5
    print(f"  {'✅' if unit_ok else '❌'} region：范围内 {out[50, 50]:.2f}、范围外 {out[10, 10]:.2f}、边界 {out[50, 39]:.2f}")
    ok = ok and unit_ok

    grown = postprocess.apply_foreground_hint(alpha, hint, expand=10, feather=1.0)
    grow_ok = grown[50, 34] > 0.5 and grown[50, 20] < 0.05
    print(f"  {'✅' if grow_ok else '❌'} region 外扩 10px：34px 处 {grown[50, 34]:.2f}，20px 处 {grown[50, 20]:.2f}")
    ok = ok and grow_ok

    # subject：两个分开的方块，只涂到左边那个 → 只保留左边
    two = np.zeros((100, 100), np.float32)
    two[40:60, 10:30] = 1.0
    two[40:60, 70:90] = 1.0
    seed = np.zeros((100, 100), np.uint8)
    seed[50, 20] = 255
    kept = postprocess.apply_hint_subject(two, seed, expand=4, feather=0)
    subj_ok = kept[50, 20] > 0.9 and kept[50, 80] < 0.1
    print(f"  {'✅' if subj_ok else '❌'} subject：涂到的成分保留 {kept[50, 20]:.2f}，没涂到的成分去掉 {kept[50, 80]:.2f}")
    ok = ok and subj_ok

    # ONNX 类后端不直接接收标记；后处理仍必须能补回明确涂到的漏抠像素。
    missing = np.zeros((100, 100), np.float32)
    missing[40:60, 10:30] = 1.0
    missed_hint = np.zeros((100, 100), np.uint8)
    missed_hint[50, 80] = 255
    recovered = postprocess.apply_hint_subject(missing, missed_hint, expand=4, feather=0)
    recover_ok = recovered[50, 80] > 0.9 and recovered[50, 20] > 0.9
    print(f"  {'✅' if recover_ok else '❌'} 模型漏抠时，原图笔迹仍能补回像素")
    ok = ok and recover_ok

    # 色差先验漏掉第二个物体时，画在它上面的前景种子必须参与 GrabCut。
    rgb = np.full((256, 320, 3), 150, np.uint8)
    cv2.circle(rgb, (88, 128), 33, (230, 40, 40), -1)
    cv2.circle(rgb, (232, 128), 33, (40, 200, 40), -1)
    rough = np.zeros(rgb.shape[:2], np.uint8)
    cv2.circle(rough, (88, 128), 33, 255, -1)
    hint = np.zeros_like(rough)
    cv2.circle(hint, (232, 128), 10, 255, -1)
    backend = SimpleBackend()
    plain = backend._refine_with_grabcut(rgb, rough, True)
    guided = backend._refine_with_grabcut(rgb, rough, True, hint)
    seed_ok = plain[128, 232] < 0.5 and guided[128, 232] > 0.5
    print(f"  {'✅' if seed_ok else '❌'} 主体标记能补回色差先验漏掉的部分")
    ok = ok and seed_ok

    # 杂乱背景时，小笔迹应扩展到物体边缘，不能只保留笔刷形状。
    rng = np.random.default_rng(42)
    busy = rng.integers(80, 150, size=(240, 320, 3), dtype=np.uint8)
    cv2.circle(busy, (160, 120), 48, (230, 40, 40), -1)
    mark = np.zeros(busy.shape[:2], np.uint8)
    cv2.circle(mark, (160, 120), 8, 255, -1)
    expanded = backend._refine_with_grabcut(busy, np.zeros_like(mark), False, mark)
    grow_ok = expanded[120, 200] > 0.5 and expanded[120, 30] < 0.5
    print(f"  {'✅' if grow_ok else '❌'} 杂乱背景下从笔迹扩展到主体边缘")
    ok = ok and grow_ok
    return ok


def main() -> int:
    app = make_app()
    ok = test_postprocess()

    TMP.mkdir(parents=True, exist_ok=True)
    window = make_window(TMP / "out")
    app.processEvents()
    assert window.select_module("matting")
    page = window.stack.currentWidget()
    assert page.hint_view is not None, "抠图页应有「原图标记」画笔"

    src = TMP / "hint_sample.png"
    rgb, gt = make_sample(640, 480, seed=9)
    io_utils.write_image(src, rgb)
    page.add_paths([str(src)])
    page.table.selectRow(0)
    app.processEvents()
    item = page.items[0]
    h, w = rgb.shape[:2]
    gt_area = float(gt.mean())

    print("\n== 原图页工具条 ==")
    bar_ok = page.tabs.currentWidget() is page.hint_view and page.brush_stack.currentWidget() is page.hint_bar
    print(f"  {'✅' if bar_ok else '❌'} 默认停在原图页且显示「标记主体」工具条")
    ok = ok and bar_ok

    page.hint_check.click()
    app.processEvents()
    print(f"  {'✅' if page.hint_view.painting_enabled else '❌'} 勾选后直接可涂（不需要先处理）")
    ok = ok and page.hint_view.painting_enabled

    def process(use_hint_button: bool = False) -> np.ndarray:
        (page.btn_process_hint if use_hint_button else page.btn_run_current).click()
        assert wait_for_worker(app, page), "处理超时"
        result = page.results.get(str(item.path))
        assert result is not None and result.ok, getattr(result, "message", "处理失败")
        return result.item.alpha

    def paint(points) -> None:
        drag_on_scene(app, page.hint_view, points)

    # ================================================================ A. 默认：涂一笔 → 整只主体
    print("\n== A. 默认模式（subject）：在主体上随意涂一笔 ==")
    page._set_brush_radius(26)
    paint([(w * 0.46, h * 0.46), (w * 0.52, h * 0.54)])

    hint = page.hints.get(str(item.path))
    print(f"  {'✅' if hint is not None and not hint.is_empty else '❌'} 标记已记录：笔画数={hint.stroke_count if hint else 0}")
    ok = ok and hint is not None and not hint.is_empty

    stored = item.meta.get("fg_hint")
    store_ok = stored is not None and stored.shape[:2] == (h, w) and int(stored.max()) == 255
    print(f"  {'✅' if store_ok else '❌'} 已写入处理参数：shape={None if stored is None else stored.shape}")
    ok = ok and store_ok

    status_text = page.table.item(page.table.row_for(item.path), 2).text()
    print(f"  {'✅' if status_text == '已标记' else '❌'} 列表状态：{status_text}")
    ok = ok and status_text == "已标记"

    shown = page.hint_view._item.pixmap().toImage()
    marked = shown.pixelColor(int(w * 0.49), int(h * 0.50))
    plain = shown.pixelColor(int(w * 0.08), int(h * 0.08))
    green_ok = marked.green() > 150 and marked.green() > marked.red() and plain.green() < marked.green()
    print(
        f"  {'✅' if green_ok else '❌'} 原图显示绿色标记：标记处 "
        f"({marked.red()},{marked.green()},{marked.blue()})，未标记处 "
        f"({plain.red()},{plain.green()},{plain.blue()})"
    )
    ok = ok and green_ok

    hint_action_ok = page.btn_process_hint.isEnabled()
    print(f"  {'✅' if hint_action_ok else '❌'} 原图旁的「按标记抠图」按钮可用")
    ok = ok and hint_action_ok
    alpha = process(use_hint_button=True)
    switched_ok = page.tabs.currentWidget() is page.views["结果"]
    print(f"  {'✅' if switched_ok else '❌'} 点击后自动切到结果页")
    ok = ok and switched_ok
    score = iou(alpha, gt)
    corner = int(alpha[int(h * 0.08), int(w * 0.08)])
    whole_ok = score > 0.85 and corner == 0
    print(
        f"  {'✅' if whole_ok else '❌'} 涂一笔就抠出整只主体：与真值 IoU={score:.3f}，"
        f"背景角落 alpha={corner}（回归：曾经只是把涂抹范围矩形裁下来）"
    )
    ok = ok and whole_ok

    band_px = int(np.count_nonzero(stored > 0))
    mask_px = int(np.count_nonzero(alpha > 127))
    body_ok = mask_px > 3 * band_px
    print(
        f"  {'✅' if body_ok else '❌'} 结果远大于涂抹面积（说明是识别出的主体而非裁剪）："
        f"主体 {mask_px}px vs 涂抹 {band_px}px"
    )
    ok = ok and body_ok

    page.tabs.setCurrentWidget(page.views["结果"])
    app.processEvents()
    page.views["结果"].fit_to_view()
    app.processEvents()
    window.grab().save(str(TMP / "fg_hint_subject.png"))

    # ================================================================ B. region：严格只抠涂抹范围
    print("\n== B. 严格模式（region）：只抠涂抹到的范围 ==")
    page.tabs.setCurrentWidget(page.hint_view)
    page.panel.set_values({"hint_mode": "region", "fg_hint_expand": 12})
    page.clear_strokes("hint")
    app.processEvents()
    page._set_brush_radius(40)
    paint([(w * 0.36, h * 0.30), (w * 0.36, h * 0.52), (w * 0.36, h * 0.75)])
    stored = item.meta.get("fg_hint")
    alpha = process()

    left = int(alpha[int(h * 0.52), int(w * 0.36)])
    right = int(alpha[int(h * 0.52), int(w * 0.68)])
    outside = int(alpha[int(h * 0.08), int(w * 0.08)])
    region_ok = left > 200 and right == 0 and outside == 0
    print(f"  {'✅' if region_ok else '❌'} 只在涂抹范围内抠出：涂抹处 {left}，主体右半边 {right}，背景 {outside}")
    ok = ok and region_ok

    def row_width(mask: np.ndarray, y_ratio: float) -> int:
        return int((mask[int(mask.shape[0] * y_ratio)] > 127).sum())

    mid_width = row_width(alpha, 0.52)
    top_width = row_width(alpha, 0.26)
    shaped = top_width < 0.6 * max(1, mid_width)
    print(
        f"  {'✅' if shaped else '❌'} 仍是主体轮廓而不是矩形裁剪："
        f"0.52h 处宽 {mid_width}px，0.26h 处宽 {top_width}px"
    )
    ok = ok and shaped

    mask_px = int(np.count_nonzero(alpha > 127))
    bbox_area = 0
    if stored is not None and stored.any():
        ys, xs = np.nonzero(stored > 0)
        bbox_area = int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    not_block = bbox_area == 0 or mask_px < 0.9 * bbox_area
    print(f"  {'✅' if not_block else '❌'} 不是整块裁下来：主体 {mask_px}px vs 涂抹包围盒 {bbox_area}px")
    ok = ok and not_block

    # ================================================================ C. 清除标记 → 全自动
    print("\n== C. 清除标记后恢复全自动 ==")
    page.clear_strokes("hint")
    app.processEvents()
    cleared_ok = item.meta.get("fg_hint") is None
    print(f"  {'✅' if cleared_ok else '❌'} 标记已清除：fg_hint={item.meta.get('fg_hint')}")
    ok = ok and cleared_ok

    alpha3 = process()
    score3 = iou(alpha3, gt)
    auto_ok = score3 > 0.9
    print(f"  {'✅' if auto_ok else '❌'} 恢复全自动：与真值 IoU={score3:.3f}")
    ok = ok and auto_ok

    # ================================================================ 放大后标记：缩放保持 + 流畅度
    print("\n== 放大后在原图上涂抹：缩放不能被弹回，且要够快 ==")
    page.tabs.setCurrentWidget(page.hint_view)
    page.hint_check.setChecked(True)
    app.processEvents()
    view = page.hint_view
    view.zoom_by(3.0)
    app.processEvents()
    zoom_before = float(view.transform().m11())
    trace = [(w * (0.44 + 0.0008 * i), h * (0.36 + 0.0012 * i)) for i in range(40)]
    started = time.perf_counter()
    paint(trace)
    per_move = (time.perf_counter() - started) / len(trace) * 1000.0
    zoom_after = float(view.transform().m11())
    zoom_ok = abs(zoom_after - zoom_before) < 1e-6
    speed_ok = per_move < 25.0
    print(f"  {'✅' if zoom_ok else '❌'} 涂抹后缩放保持不变：{zoom_before:.2f} → {zoom_after:.2f}")
    print(f"  {'✅' if speed_ok else '❌'} 每次鼠标移动平均耗时 {per_move:.1f} ms（阈值 25ms）")
    ok = ok and zoom_ok and speed_ok

    # ================================================================ 截图
    page.tabs.setCurrentWidget(page.hint_view)
    page.hint_check.setChecked(True)
    page._set_brush_radius(30)
    paint([(w * 0.44, h * 0.36), (w * 0.56, h * 0.60)])
    app.processEvents()
    page.hint_view.fit_to_view()
    app.processEvents()
    window.grab().save(str(TMP / "fg_hint_ui.png"))

    window.close()
    print("\n主体标记测试通过 ✅" if ok else "\n主体标记测试失败 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
