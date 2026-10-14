"""配置更新源后构建完整发布包：python tools/build_release.py。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imgtool import APP_ID, __version__
from imgtool.core.config import AppConfig
from imgtool.core.updates import UpdateService


def package_release(app_dir: Path, cfg, output_dir: Path) -> Path:
    required = (cfg.exe_name, cfg.updater_name, "update_config.json")
    for filename in required:
        if not (app_dir / filename).is_file():
            raise ValueError(f"发布目录缺少 {filename}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / cfg.asset_name_for(__version__)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(app_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(app_dir)
            if relative.parts[0].lower() in {"config", "models", "output", "logs", ".backup-update"}:
                continue
            archive.write(path, relative.as_posix())
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="构建像匣主程序、独立更新程序和完整 zip 发布包")
    parser.add_argument("--repository", help="GitHub 用户名/仓库名；会保存到 update_config.json")
    parser.add_argument("--updater-only", action="store_true", help="只构建独立更新程序")
    parser.add_argument("--skip-build", action="store_true", help="使用已有 dist 产物生成 zip")
    args = parser.parse_args()
    service = UpdateService(AppConfig(root=ROOT))
    if args.repository:
        service.save_repository(args.repository)
    cfg = None if args.updater_only else service.load()
    if not args.skip_build:
        env = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(ROOT / "build" / ".pyinstaller-cache"), PYTHONUTF8="1")
        subprocess.run([sys.executable, "-m", "PyInstaller", "updater.spec", "--noconfirm", "--clean"], cwd=ROOT, env=env, check=True)
        if not args.updater_only:
            subprocess.run([sys.executable, "-m", "PyInstaller", "imagebox.spec", "--noconfirm", "--clean"], cwd=ROOT, env=env, check=True)
    if args.updater_only:
        print(ROOT / "dist" / "updater.exe")
        return
    app_dir = ROOT / "dist" / APP_ID
    shutil.copy2(ROOT / "dist" / "updater.exe", app_dir / cfg.updater_name)
    # 发布目录是独立副本，不触碰开发配置、模型和输出。
    shutil.copy2(service.config_path, app_dir / "update_config.json")
    shutil.copy2(ROOT / "docs" / "发行版使用说明.md", app_dir / "README.md")
    archive = package_release(app_dir, cfg, ROOT / "dist" / "releases")
    print(archive)
    print(f"Release tag: v{__version__}")


if __name__ == "__main__":
    main()
