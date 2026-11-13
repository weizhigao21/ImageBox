"""超分模块自测：分块推理数学、取消、整链路输出、参数与注册表。

运行：python tests/superres_test.py
不需要任何模型权重：用「最近邻 4x」假后端验证分块拼装的精确性。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imgtool.core.pipeline import Pipeline  # noqa: E402
from imgtool.core.registry import discover_modules  # noqa: E402
from imgtool.core.types import ImageItem, TaskContext  # noqa: E402
from imgtool.modules.superres.backend import (  # noqa: E402
    SuperresBackend,
    tiled_forward,
)
from imgtool.modules.superres.module import MODULE  # noqa: E402


def nearest_x4(rgb: np.ndarray) -> np.ndarray:
    """逐像素 4x 最近邻放大（位置无关，可用于校验分块拼装是否严丝合缝）。"""
    return np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1)


class FakeBackend(SuperresBackend):
    name = "fake"
    scale = 4

    def load(self) -> None:
        self._loaded = True

    def _forward(self, rgb_f32: np.ndarray) -> np.ndarray:
        return nearest_x4(rgb_f32)


class SuperresTests(unittest.TestCase):
    # ---------------------------------------------------------- 分块推理
    def test_tiled_matches_direct(self):
        rng = np.random.default_rng(7)
        img = rng.random((90, 130, 3), dtype=np.float32)
        direct = nearest_x4(img)
        for tile, pad in [(64, 10), (33, 5), (90, 0), (256, 10), (1, 2)]:
            with self.subTest(tile=tile, pad=pad):
                tiled = tiled_forward(img, nearest_x4, scale=4, tile=tile, tile_pad=pad)
                self.assertEqual(tiled.shape, (360, 520, 3))
                self.assertTrue(
                    np.array_equal(tiled, direct),
                    f"tile={tile} pad={pad} 时分块拼装与整图结果不一致",
                )

    def test_tiled_progress_and_cancel(self):
        img = np.zeros((70, 70, 3), dtype=np.float32)
        seen = []
        tiled_forward(img, nearest_x4, scale=4, tile=32, on_progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(seen[-1], (9, 9))  # ceil(70/32)=3 → 3x3 分块

        calls = {"n": 0}

        def cancel() -> None:
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("任务已取消")

        with self.assertRaises(RuntimeError):
            tiled_forward(img, nearest_x4, scale=4, tile=32, check_cancelled=cancel)

    def test_whole_image_pass_when_tile_zero(self):
        img = np.zeros((40, 40, 3), dtype=np.float32)
        out = tiled_forward(img, nearest_x4, scale=4, tile=0)
        self.assertEqual(out.shape, (160, 160, 3))

    # ---------------------------------------------------------- 模块整链路
    def _inject_backend(self, backend):
        MODULE.get_backend = lambda params: backend  # type: ignore[method-assign]

    def _restore_backend(self):
        try:
            del MODULE.get_backend  # type: ignore[attr-defined]
        except AttributeError:
            pass

    def test_process_writes_upscaled_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "中文 路径.png"  # 中文路径兼容
            h, w = 80, 100
            x = np.linspace(0, 255, w, dtype=np.float32)
            img = np.stack([np.tile(x, (h, 1))] * 3, axis=-1).astype(np.uint8)
            from imgtool.core import io_utils

            io_utils.write_image(src, img)

            params = {
                "backend": "onnx",
                "model": "x4plus",
                "device": "cpu",
                "tile": 64,
                "tile_pad": 4,
                "output_format": "png",
                "name_template": "{stem}_x4",
                "overwrite": False,
                "out_dir": "",
            }
            self._inject_backend(FakeBackend({}, device="cpu"))
            try:
                results = Pipeline(MODULE, params).run([ImageItem(path=src)], run_dir=tmp / "out")
            finally:
                self._restore_backend()
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertTrue(result.ok, result.message)
            out_path = result.outputs["main"]
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.suffix, ".png")
            up = io_utils.read_image(out_path)
            self.assertEqual(up.shape, (h * 4, w * 4, 3))
            self.assertEqual(result.item.meta["result_size"], (w * 4, h * 4))
            self.assertEqual(result.item.meta["device_label"], "CPU")

    def test_process_jpg_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "a.jpg"
            from imgtool.core import io_utils

            io_utils.write_image(src, np.full((40, 40, 3), 128, dtype=np.uint8))
            params = {
                "backend": "onnx",
                "model": "x4plus",
                "tile": 0,
                "output_format": "jpg",
                "jpg_quality": 90,
                "name_template": "{stem}_x4",
            }
            self._inject_backend(FakeBackend({}, device="cpu"))
            try:
                results = Pipeline(MODULE, params).run([ImageItem(path=src)], run_dir=tmp / "out")
            finally:
                self._restore_backend()
            self.assertTrue(results[0].ok, results[0].message)
            self.assertEqual(results[0].outputs["main"].suffix, ".jpg")

    def test_backend_error_reports_not_raises(self):
        class Broken(FakeBackend):
            def load(self) -> None:
                raise FileNotFoundError("找不到权重（模型=x4plus）")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "a.png"
            from imgtool.core import io_utils

            io_utils.write_image(src, np.zeros((20, 20, 3), dtype=np.uint8))
            self._inject_backend(Broken({}, device="cpu"))
            try:
                results = Pipeline(MODULE, {"backend": "onnx", "model": "x4plus", "tile": 0}).run(
                    [ImageItem(path=src)], run_dir=tmp / "out"
                )
            finally:
                self._restore_backend()
            self.assertFalse(results[0].ok)
            self.assertIn("找不到权重", results[0].message)

    # ---------------------------------------------------------- 参数与注册表
    def test_param_defaults_and_coercion(self):
        defaults = MODULE.defaults()
        self.assertEqual(defaults["tile"], 256)
        self.assertEqual(defaults["model"], "x4plus")
        self.assertEqual(defaults["name_template"], "{stem}_x4")
        coerced = MODULE.coerce_params({"tile": "512", "overwrite": "true"})
        self.assertEqual(coerced["tile"], 512)
        self.assertIs(coerced["overwrite"], True)

    def test_registry_discovers_real_module(self):
        modules = {m.name: m for m in discover_modules()}
        self.assertIn("superres", modules)
        self.assertFalse(modules["superres"].placeholder)

    def test_output_path_no_overwrite_duplicates(self):
        from imgtool.core.io_utils import build_output_path

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "a_x4.png").write_bytes(b"x")
            p1 = build_output_path(tmp, "a.png", "{stem}_x4", ".png")
            self.assertEqual(p1.name, "a_x4_1.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
