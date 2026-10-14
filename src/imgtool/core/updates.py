"""像匣更新接入：发布源配置、包验证和独立更新程序交接。"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import urlparse

from pyupdater.apply import ApplyError, inspect_zip, launch_updater, payload_prefix
from pyupdater.check import ReleaseInfo, UpdateError, download_asset, fetch_latest_release, is_newer
from pyupdater.config import ConfigError, UpdateConfig

from .. import APP_ID, APP_NAME, EXE_NAME, __version__

PROTECTED_DATA = ("config", "models", "output", "logs", "update_config.json")


def parse_repository(value: str) -> tuple[str, str]:
    text = value.strip().rstrip("/")
    if text.startswith("https://"):
        parsed = urlparse(text)
        if parsed.netloc.lower() != "github.com" or parsed.query or parsed.fragment:
            raise ConfigError("请填写 GitHub 仓库链接或 用户名/仓库名")
        text = parsed.path.strip("/")
    text = text.removesuffix(".git")
    parts = text.split("/")
    if len(parts) != 2 or not re.fullmatch(r"[A-Za-z0-9-]+", parts[0]) or not re.fullmatch(r"[A-Za-z0-9_.-]+", parts[1]) or parts[1] in {".", ".."}:
        raise ConfigError("发布仓库格式应为 用户名/仓库名，例如 username/imagebox")
    return parts[0], parts[1]


class UpdateService:
    def __init__(self, config):
        self.root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(config.root)
        self.config_path = self.root / "update_config.json"

    def repository(self) -> str:
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            owner, repo = data.get("owner", ""), data.get("repo", "")
            return f"{owner}/{repo}" if owner and repo else ""
        except (OSError, ValueError, AttributeError):
            return ""

    def load(self) -> UpdateConfig:
        try:
            cfg = UpdateConfig.load(self.config_path)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"更新配置格式错误：{exc}") from exc
        return self._validate(cfg)

    def _validate(self, cfg) -> UpdateConfig:
        parse_repository(cfg.slug)
        if cfg.exe_name != EXE_NAME or cfg.updater_name != "updater.exe" or cfg.asset_prefix != APP_ID:
            raise ConfigError("像匣的更新配置应使用 ImageBox.exe、updater.exe 和 ImageBox 包名前缀")
        if cfg.backup_dir_name != ".backup-update":
            raise ConfigError("备份目录必须为 .backup-update")
        for name in cfg.data_dirs:
            if not isinstance(name, str) or "/" in name or "\\" in name or name in {"", ".", ".."}:
                raise ConfigError("保留目录必须是安装目录中的顶层名称")
            if name.casefold() in {"_internal", EXE_NAME.casefold(), "updater.exe"}:
                raise ConfigError("主程序和程序依赖不能设置为保留数据")
        cfg.data_dirs = tuple(dict.fromkeys((*PROTECTED_DATA, *cfg.data_dirs)))
        if any(p is not None and not isinstance(p, str) for p in cfg.proxies):
            raise ConfigError("代理候选应为地址字符串或 null")
        return cfg

    def save_repository(self, value: str) -> UpdateConfig:
        owner, repo = parse_repository(value)
        data = {}
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ConfigError(f"更新配置无法读取：{exc}") from exc
            if not isinstance(data, dict):
                raise ConfigError("更新配置必须是 JSON 对象")
        data.update(owner=owner, repo=repo, exe_name=EXE_NAME, app_name=APP_NAME,
                    asset_prefix=APP_ID, updater_name="updater.exe")
        data.setdefault("proxies", [None, "http://127.0.0.1:7897"])
        data.setdefault("exit_wait_seconds", 120)
        if not isinstance(data.get("data_dirs", []), list):
            raise ConfigError("保留目录必须是数组")
        if not all(isinstance(d, str) for d in data.get("data_dirs", [])):
            raise ConfigError("保留目录必须是名称字符串")
        data["data_dirs"] = list(dict.fromkeys((*PROTECTED_DATA, *data.get("data_dirs", []))))
        data.pop("config_path", None)
        try:
            cfg = UpdateConfig(**data)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"更新配置格式错误：{exc}") from exc
        self._validate(cfg)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.config_path)
        return self.load()

    def check(self, cfg: UpdateConfig) -> ReleaseInfo:
        release = fetch_latest_release(cfg)
        if not re.fullmatch(r"v?\d+\.\d+\.\d+", release.tag):
            raise UpdateError("发布版本号应为 v1.2.3 格式")
        if release.asset_name != cfg.asset_name_for(release.version):
            raise UpdateError(f"发布版本缺少对应的 Windows 更新包：{cfg.asset_name_for(release.version)}")
        return release

    def download(self, cfg, release, destination, on_progress=None, cancel=None) -> Path:
        path = download_asset(cfg, release, dest_dir=destination, on_progress=on_progress, cancel_check=cancel)
        try:
            payload_prefix(inspect_zip(path), cfg)
            with zipfile.ZipFile(path) as archive:
                damaged = archive.testzip()
                if damaged is not None:
                    raise ApplyError(f"更新包校验失败：{damaged}")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def install(self, zip_path: Path, cfg: UpdateConfig):
        if not getattr(sys, "frozen", False):
            raise ApplyError("源码运行不能替换开发目录。可下载更新包，自动安装需使用打包后的 ImageBox.exe。")
        if Path(sys.executable).name.casefold() != cfg.exe_name.casefold():
            raise ApplyError("当前程序名称与更新配置不一致")
        cfg = self._validate(cfg)
        payload_prefix(inspect_zip(zip_path), cfg)
        # 独立进程重新读配置，传入校验后的快照，确保保留规则不会丢失。
        snapshot = Path(zip_path).parent / "update-job-config.json"
        payload = asdict(cfg)
        payload.pop("config_path", None)
        snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return launch_updater(self.root, zip_path, replace(cfg, config_path=snapshot))


VERSION = __version__
