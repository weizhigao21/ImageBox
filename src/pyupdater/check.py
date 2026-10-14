"""更新检查：查询 GitHub Releases、比较版本、挑选更新包并下载。

纯标准库 + requests，不依赖 Qt（离线可测；耗时操作由调用方放进线程）。

发布约定（与 `updater.spec` / README 配套）：
- Release tag 形如 `v1.2.3`，与主程序自己的版本号（`1.2.3`）对齐；
- 更新包是**整个程序目录的 zip**（含主程序 exe 与 PyInstaller 的
  `_internal/`），文件名形如 `MyApp-v1.2.3.zip`。
"""
from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from pyupdater.config import UpdateConfig

logger = logging.getLogger(__name__)

_CACHE_DIR_NAME = "pyupdater-cache"
_UA = {"User-Agent": "pyupdater", "Accept": "application/vnd.github+json"}
_VERSION_RE = re.compile(r"(\d+(?:\.\d+)*)")
_CHUNK = 64 * 1024


class UpdateError(RuntimeError):
    """更新检查/下载失败；消息可直接展示给用户。"""


@dataclass
class ReleaseInfo:
    """一个可用的新版本（含挑中的更新包）。"""

    version: str          # 1.2.3（去掉 v 前缀）
    tag: str              # v1.2.3
    title: str
    notes: str            # 更新说明原文
    published_at: str
    asset_name: str
    asset_url: str
    asset_size: int       # 字节
    page_url: str


# ---- 版本比较 ----


def parse_version(text: str) -> tuple[int, ...]:
    """把 `v1.2.3` / `1.2.3` / `1.2.3-beta1` 解析成可比较的数字元组。

    只取**第一段连续数字**；解析不出返回空元组，调用方应视为"无法比较"
    而不是"已是最新"，避免把异常 tag 当成新版。
    """
    match = _VERSION_RE.search(str(text or ""))
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def is_newer(remote: str, local: str) -> bool:
    """远端版本是否比本地新；任一侧解析失败时保守返回 False。"""
    remote_parts = parse_version(remote)
    local_parts = parse_version(local)
    if not remote_parts or not local_parts:
        return False
    return remote_parts > local_parts


# ---- Release 解析 ----


def pick_asset(assets: list[dict], version: str = "", prefix: str = "") -> dict | None:
    """挑更新包：优先「含版本号」的 zip，其次「含前缀」的 zip，最后任意 zip。"""
    zips = [
        asset
        for asset in (assets or [])
        if str(asset.get("name", "")).lower().endswith(".zip")
    ]
    if not zips:
        return None
    if prefix:
        zips = [a for a in zips if str(a.get("name", "")).lower().startswith(prefix.lower() + "-")]
        if not zips:
            return None
    if version:
        for asset in zips:
            if version in str(asset.get("name", "")):
                return asset
    if prefix:
        for asset in zips:
            if prefix.lower() in str(asset.get("name", "")).lower():
                return asset
    return zips[0]


def parse_release(data: dict, cfg: UpdateConfig) -> ReleaseInfo | None:
    """把 GitHub API 的 release JSON 解析成 ReleaseInfo；无可用 zip 返回 None。"""
    tag = str(data.get("tag_name") or "").strip()
    version = ".".join(str(p) for p in parse_version(tag))
    asset = pick_asset(data.get("assets") or [], version, cfg.asset_prefix)
    if asset is None:
        return None
    if not version:  # tag 里没有数字：退回资源名，仍比不出高低但至少能展示
        version = str(asset.get("name") or "").removesuffix(".zip")
    return ReleaseInfo(
        version=version,
        tag=tag or f"v{version}",
        title=str(data.get("name") or tag or version),
        notes=str(data.get("body") or ""),
        published_at=str(data.get("published_at") or ""),
        asset_name=str(asset.get("name") or ""),
        asset_url=str(asset.get("browser_download_url") or ""),
        asset_size=int(asset.get("size") or 0),
        page_url=str(data.get("html_url") or cfg.releases_page),
    )


def _proxy_map(proxy: str | None) -> dict | None:
    return {"http": proxy, "https": proxy} if proxy else None


def fetch_latest_release(cfg: UpdateConfig, timeout: int = 15) -> ReleaseInfo:
    """查询最新 Release；失败抛 UpdateError（消息可直接展示）。

    依次尝试 `cfg.proxies` 里的候选：直连失败再走代理，都失败才报错。
    """
    last_error: Exception | None = None
    for proxy in cfg.proxies or (None,):
        try:
            resp = requests.get(
                cfg.releases_api,
                headers=_UA,
                timeout=timeout,
                proxies=_proxy_map(proxy),
            )
            if resp.status_code == 404:
                raise UpdateError("未找到已发布的版本：仓库可能尚无 Release、不存在或未公开")
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                raise UpdateError("GitHub 接口访问频繁被限流，请稍后再试")
            resp.raise_for_status()
            release = parse_release(resp.json(), cfg)
            if release is None:
                raise UpdateError("最新版本没有提供 zip 更新包")
            return release
        except UpdateError:
            raise
        except Exception as exc:  # 网络异常：换下一个代理候选
            last_error = exc
            logger.info("检查更新失败（%s）: %s", proxy or "直连", exc)
    raise UpdateError(f"无法连接 GitHub：{last_error}")


# ---- 下载 ----


def cache_dir() -> Path:
    """更新包缓存目录（%TEMP%/pyupdater-cache）。"""
    return Path(tempfile.gettempdir()) / _CACHE_DIR_NAME


def download_asset(
    cfg: UpdateConfig,
    release: ReleaseInfo,
    dest_dir: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    timeout: tuple[int, int] = (10, 60),
) -> Path:
    """下载更新包，返回 zip 路径（先写 `.part`，成功后改名，失败不残留）。

    Args:
        on_progress: 回调 (已下载字节, 总字节)；总字节未知时第二参为 0。
        cancel_check: 返回 True 时中止并抛 UpdateError。
    """
    if not release.asset_url:
        raise UpdateError("更新包地址为空")
    dest_dir = Path(dest_dir) if dest_dir else cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (release.asset_name or cfg.asset_name_for(release.version))
    if target.parent.resolve() != dest_dir.resolve() or "\\" in release.asset_name:
        raise UpdateError("更新包文件名非法")
    part = target.with_suffix(target.suffix + ".part")

    last_error: Exception | None = None
    for proxy in cfg.proxies or (None,):
        try:
            with requests.get(
                release.asset_url,
                headers={"User-Agent": _UA["User-Agent"]},
                timeout=timeout,
                stream=True,
                proxies=_proxy_map(proxy),
                allow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length") or release.asset_size or 0)
                done = 0
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(_CHUNK):
                        if cancel_check and cancel_check():
                            raise UpdateError("已取消下载")
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        if on_progress:
                            on_progress(done, total)
            if cancel_check and cancel_check():
                raise UpdateError("已取消下载")
            if part.stat().st_size <= 0:
                raise UpdateError("下载到的更新包为空")
            if (total > 0 and done != total) or (release.asset_size > 0 and done != release.asset_size):
                raise UpdateError("更新包下载不完整，请重新下载")
            target.unlink(missing_ok=True)
            part.replace(target)
            return target
        except UpdateError:
            part.unlink(missing_ok=True)
            raise
        except Exception as exc:  # 换下一个代理候选
            last_error = exc
            part.unlink(missing_ok=True)
            logger.info("下载更新包失败（%s）: %s", proxy or "直连", exc)
    raise UpdateError(f"下载更新包失败：{last_error}")


def cleanup_cache(keep: Path | None = None) -> None:
    """清理缓存目录（保留 `keep` 指向的文件）；失败只记日志。"""
    directory = cache_dir()
    if not directory.is_dir():
        return
    for entry in directory.iterdir():
        if keep is not None and entry == keep:
            continue
        try:
            if entry.is_dir():
                import shutil

                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        except OSError as exc:
            logger.info("清理更新缓存失败: %s (%s)", entry, exc)
