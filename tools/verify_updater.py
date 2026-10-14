"""真实 updater.exe 的失败回滚自检，所有写入只发生在独立临时测试目录。"""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    updater = ROOT / "dist" / "updater.exe"
    if not updater.is_file():
        raise FileNotFoundError("先运行 python tools/build_release.py --updater-only")
    stand_in = Path(r"C:\Windows\System32\where.exe")
    base = ROOT / "tests" / "_tmp"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-update-", dir=base) as directory:
        fixture = Path(directory).resolve()
        assert fixture.is_relative_to(base.resolve())
        install = fixture / "installed"
        install.mkdir()
        exe_name = "ImageBox.exe"
        original_exe = stand_in.read_bytes()
        (install / exe_name).write_bytes(original_exe)
        (install / "_internal").mkdir()
        (install / "_internal" / "version.txt").write_bytes(b"old")
        for folder in ("config", "models", "output", "logs"):
            (install / folder).mkdir()
            (install / folder / "keep.txt").write_bytes(b"user data")
        cfg = {"owner": "test-owner", "repo": "fixture", "exe_name": exe_name,
               "app_name": "更新自检", "asset_prefix": "ImageBox", "updater_name": "updater.exe",
               "data_dirs": ["config", "models", "output", "logs", "update_config.json"],
               "proxies": [None], "health_check_seconds": 0.5}
        config_path = install / "update_config.json"
        config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        original_config = config_path.read_bytes()
        package = fixture / "ImageBox-v9.9.9.zip"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr(exe_name, original_exe + b"new-version-marker")
            archive.writestr("_internal/version.txt", "new")
            archive.write(updater, "updater.exe")
            for folder in ("config", "models", "output", "logs"):
                archive.writestr(f"{folder}/keep.txt", "must not overwrite")
        result = subprocess.run(
            [str(updater), "--pid", "0", "--dir", str(install), "--zip", str(package), "--no-ui"],
            timeout=40, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert result.returncode == 1, "替身新版应启动后立即退出，触发回滚"
        assert (install / exe_name).read_bytes() == original_exe
        assert (install / "_internal/version.txt").read_bytes() == b"old"
        for folder in ("config", "models", "output", "logs"):
            assert (install / folder / "keep.txt").read_bytes() == b"user data"
        assert config_path.read_bytes() == original_config
        assert (install / "logs/updater.log").is_file()
        assert not (install / ".backup-update").exists()
    print("真实 updater.exe 验证通过：失败回滚，配置/模型/输出/日志保持原样。")


if __name__ == "__main__":
    main()
