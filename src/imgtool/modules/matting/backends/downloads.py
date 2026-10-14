"""模型权重下载（可选，需要联网）。"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable, Optional

from ....core.log import get_logger
from ....core.types import TaskCancelled

log = get_logger("download")


def download_file(
    url: str,
    dest: str | Path,
    on_progress: Optional[Callable[[int, int], None]] = None,
    timeout: float = 30.0,
    cancel: Optional[Callable[[], bool]] = None,
) -> Path:
    """下载到目标路径（先写 .part 再改名，避免半截文件）。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def check_cancelled() -> None:
        if cancel is not None and cancel():
            raise TaskCancelled("下载已取消")

    log.info("下载 %s -> %s", url, dest)
    try:
        check_cancelled()
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or 0)
            with open(tmp, "wb") as fh:
                copied = 0
                while True:
                    check_cancelled()
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    copied += len(chunk)
                    if on_progress is not None:
                        on_progress(copied, total)
        check_cancelled()
        if copied == 0 or (total > 0 and copied != total):
            raise ValueError(f"权重下载不完整：收到 {copied} 字节，预期 {total} 字节")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest
