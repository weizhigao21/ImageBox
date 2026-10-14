"""结果页「画笔精修」测试。

覆盖三层：
1. core.mask_edit.MaskEdit —— 擦除 / 保留 / 撤销 / 换分辨率；
2. UI 交互 —— 结果页勾选画笔会自动补一次处理（回归：曾经跳到空白页无法涂抹）；
3. 真实鼠标事件 —— 往 viewport 发 QMouseEvent 落笔，验证端到端真的能涂到。

用法：python tests/brush_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from ui_helpers import drag_on_scene, make_app, make_window, send_mouse, wait_for_worker

from imgtool.core import io_utils  # noqa: E402
from imgtool.core.mask_edit import ERASE, KEEP, MaskEdit  # noqa: E402
from make_sample import make_sample  # noqa: E402

TMP = Path(__file__).resolve().parent / "_tmp" / "brush"


def test_mask_edit() -> bool:
    print("== MaskEdit 单元测试 ==")
    ok = True
    h, w = 200, 300
    alpha = np.full((h, w), 255, np.uint8)
    edit = MaskEdit((h, w))

    edit.begin(50, 100, 20, ERASE)
    edit.extend(250, 100)
    edit.end()
    out = edit.apply(alpha)
    erased_ok = (
        out[100, 150] == 0
        and out[10, 10] == 255
        and np.all(out[edit.layer == ERASE] == 0)
        and np.all(edit.apply_region(alpha, 20, 70, 80, 130)[edit.layer[70:130, 20:80] == ERASE] == 0)
    )
    print(f"  {'✅' if erased_ok else '❌'} 擦除：笔迹处 alpha={out[100, 150]}，远处 alpha={out[10, 10]}")
    ok = ok and erased_ok

    edit.begin(150, 100, 20, KEEP)
    edit.end()
    out = edit.apply(alpha)
    keep_ok = out[100, 150] == 255 and np.all(out[edit.layer == KEEP] == 255)
    print(f"  {'✅' if keep_ok else '❌'} 保留：同处 alpha={out[100, 150]}（应回到 255）")
    ok = ok and keep_ok

    edit.undo()
    out = edit.apply(alpha)
    undo_ok = out[100, 150] == 0
    print(f"  {'✅' if undo_ok else '❌'} 撤销一笔后 alpha={out[100, 150]}（应回到 0）")
    ok = ok and undo_ok

    soft = np.full((h, w), 128, np.uint8)
    out = edit.apply(soft)
    soft_ok = out[100, 150] == 0 and out[10, 10] == 128
    print(f"  {'✅' if soft_ok else '❌'} 半透明输入：笔迹处 {out[100, 150]}，未涂处 {out[10, 10]}")
    ok = ok and soft_ok

    big = edit.apply(np.full((h * 3, w * 2), 255, np.uint8))
    scale_ok = big.shape == (h * 3, w * 2) and big[300, 300] == 0 and big[30, 30] == 255
    print(f"  {'✅' if scale_ok else '❌'} 放大到 {big.shape[1]}x{big.shape[0]}：笔迹处 {big[300, 300]}，远处 {big[30, 30]}")
    ok = ok and scale_ok

    edit.clear()
    clear_ok = edit.is_empty and edit.apply(alpha)[100, 150] == 255
    print(f"  {'✅' if clear_ok else '❌'} 清除涂抹后 is_empty={edit.is_empty}")
    ok = ok and clear_ok
    return ok


def main() -> int:
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtWidgets import QApplication

    app = make_app()
    ok = test_mask_edit()

    TMP.mkdir(parents=True, exist_ok=True)
    window = make_window(TMP / "out")
    app.processEvents()
    assert window.select_module("matting")
    page = window.stack.currentWidget()
    assert page.paint_view is not None, "抠图页应带可涂抹的预览视图"

    # 精修工具条在「结果」页，先切过去（真实用户也是这么点的）
    page.tabs.setCurrentWidget(page.views["结果"])
    app.processEvents()

    # ---------------------------------------------------------------- 没有图片时
    print("\n== 还没添加图片就勾画笔 ==")
    page.brush_check.click()
    app.processEvents()
    no_image_ok = not page.brush_check.isChecked() and "添加图片" in page.status.text()
    print(f"  {'✅' if no_image_ok else '❌'} 自动取消勾选并提示：checked={page.brush_check.isChecked()}，提示「{page.status.text()}」")
    ok = ok and no_image_ok

    src = TMP / "brush_sample.png"
    rgb, _ = make_sample(640, 480, seed=11)
    io_utils.write_image(src, rgb)
    page.add_paths([str(src)])
    page.table.selectRow(0)
    app.processEvents()

    # ---------------------------------------------------------------- 勾选前
    print("\n== 勾选前：开关必须可点（回归：曾经整条工具条被禁用）==")
    enabled_ok = page.brush_check.isEnabled()
    centre = page.brush_check.rect().center()
    at_widget = QApplication.widgetAt(page.brush_check.mapToGlobal(centre))
    not_covered = at_widget is page.brush_check or at_widget is None
    print(f"  {'✅' if enabled_ok else '❌'} 「画笔涂抹」可点击：isEnabled={enabled_ok}")
    print(f"  {'✅' if not_covered else '❌'} 没有被遮挡：命中控件={type(at_widget).__name__ if at_widget else 'None'}")
    ok = ok and enabled_ok and not_covered

    # ---------------------------------------------------------------- 勾选 → 自动处理
    print("\n== 勾选画笔：应自动先处理一次 ==")
    page.brush_check.click()
    app.processEvents()
    auto_started = page._worker is not None and page._worker.isRunning()
    print(f"  {'✅' if auto_started else '❌'} 自动开始处理：worker 运行中={auto_started}")
    print(f"     状态提示：{page.status.text()}")
    ok = ok and auto_started

    finished = wait_for_worker(app, page)
    painting_ok = finished and page.paint_view.painting_enabled
    print(
        f"  {'✅' if painting_ok else '❌'} 处理完成并自动开启涂抹："
        f"painting_enabled={page.paint_view.painting_enabled}"
    )
    print(f"     提示：{page.brush_hint.text()}")
    ok = ok and painting_ok

    path = page.table.current_path()
    result = page.results.get(str(path))
    if result is None or not result.ok:
        print("!! 抠图未完成，无法继续测试涂抹")
        return 1
    session = page.sessions.get(str(path))
    base_alpha = page._composed_alpha(session).copy()
    h, w = base_alpha.shape[:2]
    print(f"  基础 alpha 前景占比 {(base_alpha > 127).mean():.1%}，预览尺寸 {w}x{h}")

    # ---------------------------------------------------------------- 真实鼠标落笔
    print("\n== 真实鼠标事件落笔（发到 viewport）==")
    viewport = page.paint_view.viewport()
    middle = viewport.rect().center()
    scene_middle = page.paint_view.mapToScene(middle)
    send_mouse(app, viewport, QEvent.Type.MouseMove, middle)
    send_mouse(app, viewport, QEvent.Type.MouseButtonPress, middle)
    send_mouse(app, viewport, QEvent.Type.MouseMove, middle + middle / 5)
    overlay = page.paint_view._edit_overlay
    mark = overlay.pixmap().toImage().pixelColor(round(scene_middle.x()), round(scene_middle.y()))
    active_red = overlay.isVisible() and mark.red() > mark.green() and mark.alpha() >= 100
    sx, sy = round(scene_middle.x()), round(scene_middle.y())
    preview_alpha = page.paint_view._item.pixmap().toImage().pixelColor(sx, sy).alpha()
    expected_alpha = int(session.composite[sy, sx])
    preview_ok = abs(preview_alpha - expected_alpha) <= 1
    send_mouse(app, viewport, QEvent.Type.MouseButtonRelease, middle, pressed=False)
    app.processEvents()
    overlay_cleared = not overlay.isVisible() and overlay.pixmap().isNull()

    after_mouse = page._composed_alpha(session)
    changed = int((base_alpha != after_mouse).sum())
    mouse_ok = changed > 0 and session.edit.stroke_count == 1
    print(f"  {'✅' if mouse_ok else '❌'} 鼠标涂抹生效：alpha 变化 {changed} 像素，笔画数 {session.edit.stroke_count}")
    print(f"     列表状态：{page.table.item(page.table.row_for(path), 2).text()}")
    print(f"  {'✅' if active_red and overlay_cleared else '❌'} 红色提示仅在落笔期间显示，松开后消失")
    print(f"  {'✅' if preview_ok else '❌'} 涂抹时预览 alpha 与掩码一致：{preview_alpha} / {expected_alpha}")
    ok = ok and mouse_ok and active_red and overlay_cleared and preview_ok

    # ---------------------------------------------------------------- 撤销
    print("\n== 撤销 ==")
    page.undo_stroke()
    app.processEvents()
    diff = int((base_alpha != page._composed_alpha(session)).sum())
    print(f"  {'✅' if diff == 0 else '❌'} 撤销后回到原始 alpha（差异像素 {diff}）")
    ok = ok and diff == 0

    # ---------------------------------------------------------------- 精确涂抹
    print("\n== 涂抹并检查状态 ==")
    page._set_brush_radius(60)
    page._set_brush_value(ERASE)
    drag_on_scene(app, page.paint_view, [(w * 0.5, h * 0.5), (w * 0.62, h * 0.5)])
    app.processEvents()

    alpha_after = page._composed_alpha(session)
    centre = int(alpha_after[int(h * 0.5), int(w * 0.5)])
    paint_ok = centre == 0 and session.dirty and page.btn_save_edit.isEnabled() and page.btn_undo.isEnabled()
    print(f"  {'✅' if paint_ok else '❌'} 涂抹中心 alpha {base_alpha[int(h*0.5), int(w*0.5)]} → {centre}，dirty={session.dirty}，保存/撤销可用")
    ok = ok and paint_ok

    print(f"  {'✅' if not overlay.isVisible() else '❌'} 涂抹完成后不保留彩色痕迹")
    ok = ok and not overlay.isVisible()
    page._set_brush_value(KEEP)
    keep_pos = page.paint_view.mapFromScene(QPointF(w * 0.5, h * 0.5)) - viewport.pos()
    send_mouse(app, viewport, QEvent.Type.MouseMove, keep_pos)
    send_mouse(app, viewport, QEvent.Type.MouseButtonPress, keep_pos)
    restored = overlay.pixmap().toImage().pixelColor(int(w * 0.5), int(h * 0.5))
    keep_overlay_ok = overlay.isVisible() and restored.green() > restored.red() and restored.alpha() >= 100
    send_mouse(app, viewport, QEvent.Type.MouseButtonRelease, keep_pos, pressed=False)
    keep_overlay_ok = keep_overlay_ok and not overlay.isVisible()
    page.undo_stroke()
    page._set_brush_value(ERASE)
    print(f"  {'✅' if keep_overlay_ok else '❌'} 补回笔迹显示为绿色")
    ok = ok and keep_overlay_ok

    # ---------------------------------------------------------------- 保存
    print("\n== 保存修改 ==")
    page.save_manual_edit()
    app.processEvents()

    rgba_path = Path(result.outputs["rgba"])
    _, saved_alpha = io_utils.read_image_rgba(rgba_path)
    ph, pw = saved_alpha.shape[:2]
    hole = int(saved_alpha[int(ph * 0.5), int(pw * 0.5)])
    edge = int(saved_alpha[int(ph * 0.28), int(pw * 0.5)])
    base_edge = int(result.item.alpha[int(ph * 0.28), int(pw * 0.5)])
    save_ok = hole == 0 and edge > 200 and abs(edge - base_edge) <= 2
    print(
        f"  {'✅' if save_ok else '❌'} 输出文件 {rgba_path.name}：涂抹处 alpha={hole}，"
        f"主体其它位置 alpha={edge}（修改前 {base_edge}）"
    )
    ok = ok and save_ok

    extra = list(rgba_path.parent.glob(f"{rgba_path.stem}_[0-9]*.png"))
    print(f"  {'✅' if not extra else '❌'} 没有产生重复文件（{len(extra)} 个 _N 副本）")
    ok = ok and not extra

    # ---------------------------------------------------------------- 缩放 + 涂抹性能
    print("\n== 放大后涂抹：缩放不能被弹回，且每次移动要够快（回归：又卡又会跳回适应大小）==")
    view = page.paint_view
    view.zoom_by(5.0)
    app.processEvents()
    zoom_before = float(view.transform().m11())
    trace = [(w * (0.45 + 0.0006 * i), h * (0.45 + 0.0004 * i)) for i in range(40)]
    started = time.perf_counter()
    drag_on_scene(app, view, trace)
    per_move = (time.perf_counter() - started) / len(trace) * 1000.0
    zoom_after = float(view.transform().m11())
    zoom_ok = abs(zoom_after - zoom_before) < 1e-6
    speed_ok = per_move < 25.0
    print(f"  {'✅' if zoom_ok else '❌'} 涂抹后缩放保持不变：{zoom_before:.2f} → {zoom_after:.2f}")
    print(f"  {'✅' if speed_ok else '❌'} 每次鼠标移动平均耗时 {per_move:.1f} ms（阈值 25ms）")
    ok = ok and zoom_ok and speed_ok

    # ---------------------------------------------------------------- 中键平移
    print("\n== 按住滚轮（中键）拖动平移：画笔开启时也要能移动图片 ==")
    from PySide6.QtCore import QPoint, Qt

    viewport = view.viewport()
    hbar, vbar = view.horizontalScrollBar(), view.verticalScrollBar()
    h0, v0 = hbar.value(), vbar.value()
    zoom_before = float(view.transform().m11())
    strokes_before = session.edit.stroke_count
    start = viewport.rect().center()
    send_mouse(app, viewport, QEvent.Type.MouseButtonPress, start, button=Qt.MouseButton.MiddleButton)
    send_mouse(app, viewport, QEvent.Type.MouseMove, start - QPoint(50, 30), button=Qt.MouseButton.MiddleButton)
    send_mouse(app, viewport, QEvent.Type.MouseMove, start - QPoint(100, 60), button=Qt.MouseButton.MiddleButton)
    send_mouse(
        app, viewport, QEvent.Type.MouseButtonRelease, start - QPoint(100, 60),
        pressed=False, button=Qt.MouseButton.MiddleButton,
    )
    app.processEvents()

    pan_moved = (hbar.value(), vbar.value()) != (h0, v0)
    pan_no_paint = session.edit.stroke_count == strokes_before
    pan_zoom_kept = abs(float(view.transform().m11()) - zoom_before) < 1e-6
    cursor_ok = viewport.cursor().shape() == Qt.CursorShape.CrossCursor
    print(f"  {'✅' if pan_moved else '❌'} 图片被平移：滚动位置 ({h0},{v0}) → ({hbar.value()},{vbar.value()})")
    print(f"  {'✅' if pan_no_paint else '❌'} 平移不会误涂抹：笔画数 {strokes_before} → {session.edit.stroke_count}")
    print(f"  {'✅' if pan_zoom_kept else '❌'} 平移不改变缩放：{zoom_before:.2f} → {float(view.transform().m11()):.2f}")
    print(f"  {'✅' if cursor_ok else '❌'} 松开后光标恢复为笔刷十字：{viewport.cursor().shape()}")
    ok = ok and pan_moved and pan_no_paint and pan_zoom_kept and cursor_ok

    # 关掉画笔时左键拖动也能平移（原有行为）
    page.brush_check.setChecked(False)
    app.processEvents()
    h1, v1 = hbar.value(), vbar.value()
    send_mouse(app, viewport, QEvent.Type.MouseButtonPress, start)
    send_mouse(app, viewport, QEvent.Type.MouseMove, start - QPoint(40, 25))
    send_mouse(app, viewport, QEvent.Type.MouseButtonRelease, start - QPoint(40, 25), pressed=False)
    app.processEvents()
    left_pan = (hbar.value(), vbar.value()) != (h1, v1)
    print(f"  {'✅' if left_pan else '❌'} 关闭画笔后左键拖动也能平移：({h1},{v1}) → ({hbar.value()},{vbar.value()})")
    ok = ok and left_pan
    page.brush_check.setChecked(True)
    app.processEvents()

    # ---------------------------------------------------------------- 截图
    print("\n== 截图 ==")
    page.tabs.setCurrentIndex(1)
    page.brush_check.setChecked(True)
    page._set_brush_value(ERASE)
    page._set_brush_radius(34)
    drag_on_scene(app, page.paint_view, [
        (w * 0.40, h * 0.34),
        (w * 0.48, h * 0.45),
        (w * 0.58, h * 0.55),
        (w * 0.66, h * 0.68),
    ])
    page.views["结果"].fit_to_view()
    app.processEvents()
    window.grab().save(str(TMP / "brush_ui.png"))

    window.close()
    print("\n画笔涂抹测试通过 ✅" if ok else "\n画笔涂抹测试失败 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
