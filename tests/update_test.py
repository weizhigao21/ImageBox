"""更新接入回归；使用模拟发布接口和独立临时安装目录。"""

import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from ui_helpers import make_app

from imgtool import EXE_NAME, __version__
from imgtool.core.config import AppConfig, project_root
from imgtool.core.updates import PROTECTED_DATA, UpdateService, parse_repository
from imgtool.ui.app import MainWindow
from pyupdater import apply, updater_main
from pyupdater.check import ReleaseInfo, UpdateError, download_asset, is_newer, pick_asset
from pyupdater.config import ConfigError


def release(size=100):
    return ReleaseInfo("0.4.0", "v0.4.0", "像匣 v0.4.0", "改善边缘效果", "2026-10-15", "ImageBox-v0.4.0.zip", "https://example.invalid/update.zip", size, "https://example.invalid/releases")


class UpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = AppConfig(root=self.root)
        self.config.set("check_updates_on_startup", False)
        self.service = UpdateService(self.config)
        self.cfg = self.service.save_repository("test-owner/imagebox")

    def tearDown(self):
        self.tmp.cleanup()

    def wait(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if condition():
                return
            time.sleep(0.01)
        self.fail("后台任务超时")

    def zip(self):
        path = self.root / "ImageBox-v0.4.0.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(EXE_NAME, b"NEW")
            archive.writestr("_internal/program.dat", b"NEW-DATA")
            for directory in ("config", "models", "output", "logs"):
                archive.writestr(f"{directory}/keep.txt", "do not overwrite")
        return path

    def test_repository_and_required_protection(self):
        self.assertEqual(parse_repository("https://github.com/test-owner/imagebox.git"), ("test-owner", "imagebox"))
        for value in ("", "../imagebox", "https://other.invalid/u/repo", "a/b/c"):
            with self.assertRaises(ConfigError):
                parse_repository(value)
        self.assertTrue(all(self.cfg.is_protected(n) for n in PROTECTED_DATA))
        self.assertEqual(self.cfg.asset_name_for(__version__), f"ImageBox-v{__version__}.zip")

    def test_invalid_repository_does_not_overwrite_configuration(self):
        original = self.service.config_path.read_bytes()
        with self.assertRaises(ConfigError):
            self.service.save_repository("invalid")
        self.assertEqual(self.service.config_path.read_bytes(), original)

    def test_wrong_product_and_missing_source_are_reported(self):
        self.service.config_path.write_text(json.dumps({"owner": "", "repo": "imagebox", "exe_name": EXE_NAME}), encoding="utf-8")
        with self.assertRaises(ConfigError):
            self.service.load()
        data = {"owner": "a", "repo": "b", "exe_name": "Other.exe", "asset_prefix": "Other"}
        self.service.config_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ConfigError):
            self.service.load()

    def test_product_asset_is_selected_and_unrelated_zip_rejected(self):
        assets = [{"name": "Other-v0.4.0.zip"}, {"name": "ImageBox-v0.4.0.zip"}]
        self.assertEqual(pick_asset(assets, "0.4.0", "ImageBox")["name"], "ImageBox-v0.4.0.zip")
        self.assertIsNone(pick_asset(assets[:1], "0.4.0", "ImageBox"))
        self.assertTrue(is_newer("0.4.0", __version__))
        self.assertFalse(is_newer(__version__, __version__))

    def test_truncated_download_and_unsafe_filename_leave_no_update(self):
        response = SimpleNamespace(headers={"Content-Length": "100"}, raise_for_status=lambda: None, iter_content=lambda chunk: iter([b"short"]))
        mock = unittest.mock.MagicMock()
        mock.__enter__.return_value = response
        self.cfg.proxies = (None,)
        with patch("pyupdater.check.requests.get", return_value=mock), self.assertRaises(UpdateError):
            download_asset(self.cfg, release(), self.root / "download")
        self.assertEqual(list((self.root / "download").glob("*")), [])
        bad = release()
        bad.asset_name = "../escape.zip"
        with self.assertRaises(UpdateError):
            download_asset(self.cfg, bad, self.root / "download")

    def test_zip_validation_rejects_traversal_and_missing_executable(self):
        bad = self.root / "bad.zip"
        for name in ("../escape.txt", "unrelated.txt"):
            with zipfile.ZipFile(bad, "w") as archive:
                archive.writestr(name, "bad")
            with patch("imgtool.core.updates.download_asset", return_value=bad), self.assertRaises(Exception):
                self.service.download(self.cfg, release(), self.root / "downloads")
            self.assertFalse(bad.exists())

    def test_source_mode_never_launches_updater(self):
        with patch("imgtool.core.updates.launch_updater") as launch, self.assertRaises(apply.ApplyError):
            self.service.install(self.zip(), self.cfg)
        launch.assert_not_called()

    def test_frozen_root_is_executable_directory(self):
        with patch("sys.frozen", True, create=True), patch("sys.executable", str(self.root / EXE_NAME)), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(project_root(), self.root)
            service = UpdateService(AppConfig(root=self.root / "elsewhere"))
            self.assertEqual(service.root, self.root)

    def test_installer_receives_validated_data_protection_snapshot(self):
        from pyupdater.config import UpdateConfig
        self.cfg.data_dirs = ()
        original = self.service.config_path.read_bytes()
        with patch("sys.frozen", True, create=True), patch("sys.executable", str(self.root / EXE_NAME)), patch("imgtool.core.updates.launch_updater") as launch:
            self.service.install(self.zip(), self.cfg)
        job_cfg = launch.call_args.args[2]
        loaded = UpdateConfig.load(job_cfg.config_path)
        self.assertTrue(all(loaded.is_protected(n) for n in PROTECTED_DATA))
        self.assertEqual(self.service.config_path.read_bytes(), original)

    def test_ui_check_and_download_complete_without_blocking(self):
        window = MainWindow(self.config)
        window.show()
        window.show_updates()
        dialog = window._update_dialog
        with patch("imgtool.core.updates.fetch_latest_release", return_value=release()):
            dialog.check()
            self.wait(lambda: dialog.worker is None)
        self.assertEqual(dialog.action_button.text(), "下载更新包")
        self.assertIn("改善边缘效果", dialog.notes.toPlainText())
        with patch("imgtool.core.updates.download_asset", return_value=self.zip()):
            dialog.action()
            self.wait(lambda: dialog.worker is None)
        self.assertEqual(dialog.action_button.text(), "打开下载目录")
        self.assertEqual(dialog.progress.value(), 100)
        self.assertTrue(dialog.zip_path.exists())
        dialog.grab().save(str(Path(__file__).parent / "_tmp" / "update_dialog.png"))
        window.close()

    def test_ui_current_version_and_error_restore_controls(self):
        window = MainWindow(self.config)
        window.show_updates()
        dialog = window._update_dialog
        current = release()
        current.version = __version__
        current.tag = f"v{__version__}"
        current.asset_name = f"ImageBox-v{__version__}.zip"
        with patch("imgtool.core.updates.fetch_latest_release", return_value=current):
            dialog.check()
            self.wait(lambda: dialog.worker is None)
        self.assertIn("最新版本", dialog.status.text())
        with patch("imgtool.core.updates.fetch_latest_release", side_effect=UpdateError("网络不可用")):
            dialog.check()
            self.wait(lambda: dialog.worker is None)
        self.assertIn("网络不可用", dialog.status.text())
        self.assertTrue(dialog.action_button.isEnabled())
        window.close()

    def test_closing_update_dialog_cancels_without_destroying_thread(self):
        def delayed(cfg):
            time.sleep(0.2)
            return release()
        window = MainWindow(self.config)
        window.show()
        window.show_updates()
        dialog = window._update_dialog
        with patch("imgtool.core.updates.fetch_latest_release", side_effect=delayed):
            dialog.check()
            dialog.reject()
            self.wait(lambda: dialog.worker is None)
        self.assertIn("取消", dialog.status.text())
        self.assertFalse(window._update_workers)
        window.close()

    def test_failure_rolls_back_and_keeps_all_user_data(self):
        install = self.root / "installed"
        install.mkdir()
        (install / EXE_NAME).write_bytes(b"OLD")
        (install / "_internal").mkdir()
        (install / "_internal" / "program.dat").write_bytes(b"OLD-DATA")
        for directory in ("config", "models", "output", "logs"):
            (install / directory).mkdir()
            (install / directory / "keep.txt").write_bytes(b"USER-DATA")
        self.cfg.config_path = install / "update_config.json"
        self.cfg.save(self.cfg.config_path)
        config_bytes = self.cfg.config_path.read_bytes()
        args = updater_main.parse_args(["--dir", str(install), "--zip", str(self.zip()), "--pid", "0", "--no-ui"])
        self.cfg.health_check_seconds = 0
        with patch("pyupdater.updater_main.start_app", return_value=SimpleNamespace(poll=lambda: 1, returncode=1)):
            with self.assertRaises(apply.ApplyError):
                updater_main.run_update(args, self.cfg, updater_main.ProgressWindow(enabled=False))
        apply.rollback(install, install / self.cfg.backup_dir_name, self.cfg)
        self.assertEqual((install / EXE_NAME).read_bytes(), b"OLD")
        self.assertEqual((install / "_internal/program.dat").read_bytes(), b"OLD-DATA")
        for directory in ("config", "models", "output", "logs"):
            self.assertEqual((install / directory / "keep.txt").read_bytes(), b"USER-DATA")
        self.assertEqual(self.cfg.config_path.read_bytes(), config_bytes)

    def test_partial_backup_failure_does_not_delete_unmoved_program_files(self):
        install, payload = self.root / "installed", self.root / "payload"
        install.mkdir()
        payload.mkdir()
        (install / "a.exe").write_bytes(b"A")
        (install / "b.dll").write_bytes(b"B")
        original_transfer = apply._transfer
        def transfer(src, dst, on_status):
            if src.name == "b.dll":
                raise OSError("locked")
            original_transfer(src, dst, on_status)
        with patch("pyupdater.apply._transfer", side_effect=transfer), self.assertRaises(OSError):
            apply.replace_program_files(payload, install, self.cfg)
        self.assertEqual((install / "a.exe").read_bytes(), b"A")
        self.assertEqual((install / "b.dll").read_bytes(), b"B")


if __name__ == "__main__":
    unittest.main(verbosity=2)
