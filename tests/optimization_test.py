"""回归：整批进度、阶段取消、参数验证、缓存、下载与安全退出。"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ui_helpers import make_app

import numpy as np
from PySide6.QtCore import QThread, QPoint

from imgtool.core.base import BaseModule, ParamSpec
from imgtool.core.config import AppConfig
from imgtool.core.device import onnx_providers
from imgtool.core.io_utils import write_image
from imgtool.core.pipeline import Pipeline
from imgtool.core.result_cache import ResultCache
from imgtool.core.types import ImageItem, ModuleResult, TaskCancelled, TaskContext
from imgtool.modules.matting.backends.downloads import download_file
from imgtool.modules.matting.module import MattingModule
from imgtool.ui.app import MainWindow
from imgtool.ui.workers import ModuleWorker


class DummyModule(BaseModule):
    name = "dummy"

    def process(self, item, params, ctx):
        if item.stem == "bad":
            raise ValueError("broken image")
        return ModuleResult(item)


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = AppConfig(root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_directory_picker_survives_restart_and_controls_export(self):
        self.config.set("check_updates_on_startup", False)
        window = MainWindow(self.config)
        self.addCleanup(window.close)
        page = window.stack.currentWidget()
        directory = self.root / "图片输出 带空格"
        with patch("imgtool.ui.widgets.param_panel.QFileDialog.getExistingDirectory", return_value=str(directory)):
            page.panel._pick_dir("out_dir")
        settings_path = self.root / "config" / "settings.json"
        saved = json.loads(settings_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["matting"]["out_dir"], str(directory))
        window.close()
        with patch("imgtool.core.config.project_root", return_value=self.root):
            restored = AppConfig.load()
        reopened = MainWindow(restored)
        self.addCleanup(reopened.close)
        page = reopened.stack.currentWidget()
        self.assertEqual(page.panel.values()["out_dir"], str(directory))
        image = np.full((128, 128, 3), 245, np.uint8)
        image[32:96, 32:96] = (30, 70, 110)
        sample = self.root / "sample.png"
        write_image(sample, image)
        page.panel.set_values({"backend": "simple", "decontaminate": False})
        result = Pipeline(page.module, page.panel.values(), restored).run([ImageItem(sample)])[0]
        self.assertTrue(result.ok, result.message)
        self.assertTrue(result.outputs)
        self.assertTrue(all(Path(p).parent == directory for p in result.outputs.values()))
        with patch("imgtool.ui.app.io_utils.open_in_explorer") as opened:
            reopened.open_output_directory()
        opened.assert_called_once_with(directory)
        reopened.close()

    def test_clearing_saved_output_directory_restores_automatic_output(self):
        self.config.set("check_updates_on_startup", False)
        self.config.sub("matting")["out_dir"] = str(self.root / "previous")
        window = MainWindow(self.config)
        self.addCleanup(window.close)
        page = window.stack.currentWidget()
        edit = page.panel._widgets["out_dir"].property("edit")
        edit.clear()
        with patch("imgtool.core.config.project_root", return_value=self.root):
            restored = AppConfig.load()
        self.assertEqual(restored.sub("matting")["out_dir"], "")
        directory = Pipeline(page.module, page.panel.values(), restored).resolve_run_dir()
        self.assertEqual(directory.parent, self.root / "output" / "matting")
        window.select_module("crop")
        with patch("imgtool.ui.app.io_utils.open_in_explorer") as opened:
            window.open_output_directory()
        opened.assert_called_once_with(self.config.output_dir)
        window.close()

    def test_output_directory_write_error_preserves_previous_configuration(self):
        self.config.set("check_updates_on_startup", False)
        self.config.sub("matting").update(out_dir="previous", backend="simple")
        settings_path = self.config.save()
        original = settings_path.read_bytes()
        window = MainWindow(self.config)
        self.addCleanup(window.close)
        page = window.stack.currentWidget()
        with patch("pathlib.Path.replace", side_effect=PermissionError("read only")), self.assertLogs("imgtool.page", level="WARNING"):
            page.panel.set_values({"out_dir": "new location"})
        self.assertEqual(settings_path.read_bytes(), original)
        self.assertEqual(self.config.sub("matting")["out_dir"], "previous")
        self.assertEqual(self.config.sub("matting")["backend"], "simple")
        self.assertFalse(settings_path.with_name(settings_path.name + ".tmp").exists())
        self.assertIn("未能保存", page.status.text())
        window.close()

    def test_unrelated_parameter_changes_do_not_write_output_configuration(self):
        self.config.set("check_updates_on_startup", False)
        window = MainWindow(self.config)
        self.addCleanup(window.close)
        with patch.object(self.config, "save") as save:
            window.stack.currentWidget().panel.set_values({"feather": 2.5})
        save.assert_not_called()
        window.close()

    def test_worker_reports_whole_batch_and_continues_after_failure(self):
        worker = ModuleWorker(DummyModule(), [ImageItem(f"{x}.png") for x in ("a", "bad", "c")], {}, self.config)
        progress, results = [], []
        worker.progress.connect(lambda d, t, m: progress.append((d, t)))
        worker.itemDone.connect(results.append)
        with self.assertLogs(level="ERROR"):
            worker.run()
        self.assertEqual(progress, [(0, 3), (1, 3), (1, 3), (2, 3), (2, 3), (3, 3)])
        self.assertEqual([r.ok for r in results], [True, False, True])
        self.assertIsNotNone(worker.run_dir)

    def test_cancelled_item_is_not_reported_as_failure(self):
        class CancelModule(DummyModule):
            def process(self, item, params, ctx):
                raise TaskCancelled("cancel")
        emitted = []
        results = Pipeline(CancelModule(), {}, self.config).run(
            [ImageItem("a.png")], on_item_done=emitted.append
        )
        self.assertEqual(results, [])
        self.assertEqual(emitted, [])

    def test_invalid_parameters_fail_before_processing(self):
        spec = ParamSpec("size", "尺寸", "int", 100, minimum=10, maximum=200)
        for value in ("nan", "inf", "-1", "201", "12.5", "abc"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "尺寸"):
                spec.coerce(value)
        self.assertEqual(spec.coerce("100"), 100)
        with self.assertRaises(ValueError):
            ParamSpec("backend", "后端", "choice", "simple", choices=[("simple", "简易")]).coerce("typo")
        with self.assertRaises(ValueError):
            ParamSpec("flag", "开关", "bool", False).coerce("typo")

    def test_onnx_cuda_does_not_depend_on_torch(self):
        fake = SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        with patch.dict(sys.modules, {"onnxruntime": fake}), patch("imgtool.core.device.resolve_device", side_effect=AssertionError("torch probe")):
            self.assertEqual(onnx_providers("auto"), ["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.assertEqual(onnx_providers("cpu"), ["CPUExecutionProvider"])

    def test_postprocessing_reuses_prediction_and_hint_invalidates_reuse(self):
        path = self.root / "test.png"
        write_image(path, np.full((40, 50, 3), 128, np.uint8))
        item = ImageItem(path)
        module = MattingModule()
        module.config = self.config
        calls = []
        def predict(rgb, hint=None):
            calls.append(hint)
            return np.full(rgb.shape[:2], 0.6, np.float32)
        backend = SimpleNamespace(predict=predict, post_cleanup=False)
        params = {"backend": "simple", "decontaminate": False, "_out_dir": str(self.root / "out")}
        with patch.object(module, "get_backend", return_value=backend):
            module.process(item, params, TaskContext())
            module.process(item, {**params, "feather": 2}, TaskContext())
            self.assertEqual(len(calls), 1)
            self.assertTrue(item.meta["inference_cached"])
            item.meta["fg_hint"] = np.ones((40, 50), np.uint8) * 255
            module.process(item, params, TaskContext())
            self.assertEqual(len(calls), 2)
            self.assertFalse(item.meta["inference_cached"])
            item.meta.pop("fg_hint")
            # 同路径图像被更新后，不能使用旧推理。
            write_image(path, np.zeros((30, 20, 3), np.uint8))
            module.process(item, params, TaskContext())
            self.assertEqual(len(calls), 3)

    def test_onnx_size_change_does_not_recreate_model_session(self):
        module = MattingModule()
        module.config = self.config
        first = module.get_backend({"backend": "onnx", "max_size": 800})
        second = module.get_backend({"backend": "onnx", "max_size": 1600})
        self.assertIs(first, second)

    def test_cancel_after_inference_prevents_output(self):
        path = self.root / "image.png"
        write_image(path, np.full((20, 20, 3), 128, np.uint8))
        cancelled = [False]
        def predict(rgb, hint=None):
            cancelled[0] = True
            return np.ones(rgb.shape[:2], np.float32)
        module = MattingModule()
        with patch.object(module, "get_backend", return_value=SimpleNamespace(predict=predict)):
            result = Pipeline(module, {"backend": "simple"}, self.config).run(
                [ImageItem(path)], TaskContext(cancel=lambda: cancelled[0])
            )
        self.assertEqual(result, [])
        self.assertEqual(list(self.config.output_dir.rglob("*.png")), [])

    def test_replaced_weight_recreates_backend(self):
        module = MattingModule()
        module.config = self.config
        weight = self.config.models_dir / "custom.onnx"
        weight.write_bytes(b"old")
        first = module.get_backend({"backend": "onnx", "onnx_model": "custom.onnx"})
        weight.write_bytes(b"new weight version")
        second = module.get_backend({"backend": "onnx", "onnx_model": "custom.onnx"})
        self.assertIsNot(first, second)

    def test_result_cache_releases_pixels_and_restores_manual_changes(self):
        cache = ResultCache()
        try:
            a, b = ImageItem("a.png"), ImageItem("b.png")
            for n, item in enumerate((a, b)):
                item.rgb = np.full((20, 30, 3), n + 10, np.uint8)
                item.alpha = np.full((20, 30), n + 100, np.uint8)
                item.meta["result_rgb"] = item.rgb.copy()
                cache.store(item)
            cache.evict_except(a)
            self.assertIsNone(b.rgb)
            self.assertIsNone(b.alpha)
            cache.load(b)
            self.assertTrue(np.all(b.alpha == 101))
            b.alpha[:] = 0
            cache.store(b)
            cache.release(b)
            cache.load(b)
            self.assertTrue(np.all(b.alpha == 0))
            self.assertTrue(np.all(b.meta["result_rgb"] == 11))
        finally:
            cache.close()

    def test_incomplete_download_preserves_existing_model(self):
        dest = self.root / "model.onnx"
        dest.write_bytes(b"existing")
        response = io.BytesIO(b"short")
        response.headers = {"Content-Length": "100"}
        with patch("urllib.request.urlopen", return_value=response), self.assertRaises(ValueError):
            download_file("https://example.invalid/model", dest)
        self.assertEqual(dest.read_bytes(), b"existing")
        self.assertFalse(dest.with_suffix(".onnx.part").exists())

    def test_cancelled_download_leaves_no_partial_file(self):
        dest = self.root / "model.onnx"
        with patch("urllib.request.urlopen") as request, self.assertRaises(TaskCancelled):
            download_file("https://example.invalid/model", dest, cancel=lambda: True)
        request.assert_not_called()
        self.assertFalse(dest.exists())
        self.assertFalse(dest.with_suffix(".onnx.part").exists())

    def test_ui_selection_preview_cache_and_small_layout(self):
        window = MainWindow(self.config)
        window.resize(1080, 720)
        window.show()
        self.app.processEvents()
        page = window.stack.currentWidget()
        self.assertFalse(window.windowIcon().isNull())
        self.assertFalse(window.windowIcon().pixmap(16, 16).isNull())
        paths = []
        for n in range(8):
            path = self.root / f"sample{n}.png"
            write_image(path, np.full((20, 30, 3), n, np.uint8))
            paths.append(path)
        page.add_paths([str(p) for p in paths])
        for path in paths:
            page._original_preview(path)
        self.assertLessEqual(len(page._preview_cache), 6)
        page.table.selectRow(3)
        with patch.object(page, "_start") as start:
            page.process_selected()
        self.assertEqual([i.path for i in start.call_args.args[0]], [paths[3]])
        self.assertGreaterEqual(page.views["原图"].width(), 400)
        self.assertGreaterEqual(page.panel.width(), 260)
        page.bg_combo.setCurrentIndex(1)
        self.assertEqual(page.views["结果"].backgroundBrush().color().name(), "#ffffff")
        window.close()

    def test_compact_inspector_keeps_presets_near_fields_and_scrolls_together(self):
        window = MainWindow(self.config)
        window.resize(1080, 720)
        window.show()
        page = window.stack.currentWidget()
        page.panel.set_values({"backend": "onnx"})
        for _ in range(4):
            self.app.processEvents()
        self.assertLessEqual(page.inspector_scroll.mapTo(page, QPoint(0, 0)).y(), 65)
        body = page.inspector_scroll.widget()
        fields = [w for w in page.panel._widgets.values() if w.isVisible()]
        bottom = max(w.mapTo(body, QPoint(0, 0)).y() + w.height() for w in fields)
        preset_top = page.preset_label.mapTo(body, QPoint(0, 0)).y()
        self.assertGreaterEqual(preset_top - bottom, 6)
        self.assertLessEqual(preset_top - bottom, 40)
        page.panel.group_tabs.setCurrentIndex(1)
        page.panel.set_values({"outline_width": 10})
        for _ in range(4):
            self.app.processEvents()
        self.assertTrue(page.panel._widgets["outline_color"].isVisible())
        scroll_bar = page.inspector_scroll.verticalScrollBar()
        self.assertGreater(scroll_bar.maximum(), 0)
        scroll_bar.setValue(scroll_bar.maximum())
        self.app.processEvents()
        viewport = page.inspector_scroll.viewport()
        preset_y = page.preset_combo.mapTo(viewport, QPoint(0, 0)).y()
        self.assertGreaterEqual(preset_y, 0)
        self.assertLessEqual(preset_y + page.preset_combo.height(), viewport.height())
        window.close()

    def test_task_sidebar_keeps_selection_and_canvas_space_when_switching(self):
        window = MainWindow(self.config)
        window.resize(1080, 720)
        window.show()
        page = window.stack.currentWidget()
        for _ in range(4):
            self.app.processEvents()
        title_pos = page.page_title.mapTo(page, QPoint(0, 0))
        desc_pos = page.page_description.mapTo(page, QPoint(0, 0))
        self.assertLess(abs(title_pos.y() - desc_pos.y()), 10)
        self.assertGreater(desc_pos.x(), title_pos.x() + page.page_title.width())
        paths = []
        for n in range(2):
            path = self.root / f"task{n}.png"
            write_image(path, np.full((40, 60, 3), n + 10, np.uint8))
            paths.append(path)
        page.add_paths([str(p) for p in paths])
        self.app.processEvents()
        self.assertIs(page.inspector_stack.currentWidget(), page.task_page)
        page.table.selectRow(1)
        selected = page.table.current_path()
        canvas = page.views["原图"]
        table_x = page.table.mapTo(page, QPoint(0, 0)).x()
        canvas_right = canvas.mapTo(page, QPoint(0, 0)).x() + canvas.width()
        self.assertGreater(table_x, canvas_right)
        self.assertGreater(canvas.height(), 350)
        page.panel.set_values({"feather": 3.5})
        page.btn_params.click()
        self.app.processEvents()
        self.assertIs(page.inspector_stack.currentWidget(), page.inspector_scroll)
        self.assertEqual(page.table.current_path(), selected)
        canvas_height = canvas.height()
        page.btn_tasks.click()
        self.app.processEvents()
        self.assertEqual(canvas.height(), canvas_height)
        self.assertEqual(page.panel.values()["feather"], 3.5)
        self.assertEqual(page.table.current_path(), selected)
        self.assertIn("task1.png", page.preview_title.text())
        page.remove_selected()
        self.assertEqual(page.table.current_path(), paths[0])
        page.clear_items()
        self.assertEqual(page.table.rowCount(), 0)
        self.assertEqual(page.preview_title.text(), "图片预览")
        window.close()

    def test_close_waits_for_processing_and_download_without_blocking_ui(self):
        class SlowWorker(QThread):
            cancelled = False
            def run(self):
                self.msleep(250)
            def cancel(self):
                self.cancelled = True
        window = MainWindow(self.config)
        window.show()
        page = window.stack.currentWidget()
        workers = [SlowWorker(page), SlowWorker(page)]
        page._worker, page._download_worker = workers
        self.app.processEvents()
        for worker in workers:
            worker.start()
        start = time.perf_counter()
        window.close()
        self.assertLess(time.perf_counter() - start, 0.15)
        self.assertTrue(window.isVisible())
        self.assertTrue(all(w.cancelled for w in workers))
        deadline = time.monotonic() + 3
        while window.isVisible() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertFalse(window.isVisible())
        self.assertTrue(all(not w.isRunning() for w in workers))

    def test_batch_switch_and_repeated_manual_save_keep_result_consistent(self):
        window = MainWindow(self.config)
        window.show()
        page = window.stack.currentWidget()
        paths = []
        for n in range(3):
            path = self.root / f"batch{n}.png"
            # 简单白底商品，中心有深色主体。
            rgb = np.full((200, 300, 3), 245, np.uint8)
            rgb[50:150, 80:220] = (30, 50, 90)
            write_image(path, rgb)
            paths.append(path)
        page.add_paths([str(p) for p in paths])
        page.panel.set_values({"backend": "simple", "decontaminate": False, "out_dir": str(self.root / "out")})
        page.process_all()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.app.processEvents()
            if not page._busy and not page._worker.isRunning():
                break
            time.sleep(0.01)
        self.assertEqual(len(page.results), 3)
        self.assertEqual((page.progress.value(), page.progress.maximum()), (3, 3))
        self.assertEqual(sum(item.rgb is not None for item in page.items), 1)
        page.table.selectRow(2)
        self.app.processEvents()
        item = page.results[str(paths[2])].item
        self.assertIsNotNone(item.alpha)
        session = page.sessions[str(paths[2])]
        session.edit.begin(150, 100, 14, 1)
        session.edit.end()
        page.save_manual_edit()
        first_saved = item.alpha.copy()
        page.save_manual_edit()
        np.testing.assert_array_equal(item.alpha, first_saved)
        page.table.selectRow(0)
        page.table.selectRow(2)
        self.app.processEvents()
        np.testing.assert_array_equal(item.alpha, first_saved)
        page.save_manual_edit()
        np.testing.assert_array_equal(item.alpha, first_saved)
        self.assertFalse(page.sessions[str(paths[2])].dirty)
        page.undo_stroke()
        self.assertTrue(page.btn_save_edit.isEnabled())
        page.save_manual_edit()
        np.testing.assert_array_equal(item.alpha, item.meta["edit_base_alpha"])
        window.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
