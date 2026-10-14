"""设备探测：CUDA / CPU 选择与运行环境报告。"""

from __future__ import annotations

import platform
import sys
from typing import Any

_CUDA_CACHE: bool | None = None


def _probe_torch_cuda() -> tuple[bool, str]:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - 取决于环境
        return False, f"未安装 torch（{exc}）"
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return True, f"CUDA 可用：{name}（{total:.1f} GB）"
        return False, "torch 报告 CUDA 不可用"
    except Exception as exc:  # pragma: no cover
        return False, f"CUDA 探测失败：{exc}"


def cuda_available() -> bool:
    global _CUDA_CACHE
    if _CUDA_CACHE is None:
        _CUDA_CACHE, _ = _probe_torch_cuda()
    return bool(_CUDA_CACHE)


def resolve_device(pref: str = "auto") -> str:
    """把 auto/cuda/cpu 解析成实际的 'cuda' 或 'cpu'。"""
    pref = (pref or "auto").lower()
    if pref in {"cuda", "gpu"}:
        return "cuda" if cuda_available() else "cpu"
    if pref == "cpu":
        return "cpu"
    return "cuda" if cuda_available() else "cpu"


def onnx_providers(pref: str = "auto") -> list[str]:
    """onnxruntime 的执行提供者列表（按优先级）。"""
    available: list[str] = []
    try:
        import onnxruntime as ort  # noqa: PLC0415

        available = list(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]

    # ONNX 的 CUDA 能力由自身提供者决定，不需要安装 torch。
    wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if pref.lower() != "cpu" else [
        "CPUExecutionProvider"
    ]
    providers = [p for p in wanted if p in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def backend_device_label(backend: str, pref: str = "auto") -> str:
    if backend == "simple":
        return "CPU"
    if backend == "onnx":
        return "GPU · CUDA" if "CUDAExecutionProvider" in onnx_providers(pref) else "CPU"
    return "GPU · CUDA" if resolve_device(pref) == "cuda" else "CPU"


def environment_report() -> dict[str, Any]:
    """给 UI 状态栏/诊断脚本用的环境摘要。"""
    ok_cuda, cuda_note = _probe_torch_cuda()
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "cuda": ok_cuda,
        "cuda_note": cuda_note,
        "torch": None,
        "onnxruntime": None,
        "onnx_providers": [],
    }
    try:
        import torch  # noqa: PLC0415

        report["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import onnxruntime as ort  # noqa: PLC0415

        report["onnxruntime"] = ort.__version__
        report["onnx_providers"] = list(ort.get_available_providers())
    except Exception:
        pass
    return report


def summary_line() -> str:
    rep = environment_report()
    torch_device = "CUDA 可用" if rep["cuda"] else "CPU"
    ort_device = "CUDA 可用" if "CUDAExecutionProvider" in rep["onnx_providers"] else "CPU"
    parts = [f"Python {rep['python']}", f"Torch {torch_device}", f"ONNX {ort_device}"]
    if rep.get("torch"):
        parts.append(f"torch {rep['torch']}")
    if rep.get("onnxruntime"):
        parts.append(f"onnxruntime {rep['onnxruntime']}")
    if not rep["cuda"] and rep.get("cuda_note"):
        parts.append(rep["cuda_note"])
    return " | ".join(parts)
