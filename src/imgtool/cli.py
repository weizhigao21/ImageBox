"""命令行入口：批量处理与自检。

用法示例：
    python src/imgtool/cli.py list
    python src/imgtool/cli.py run matting -i photo.jpg -o out
    python src/imgtool/cli.py run matting -i ./photos -o out --set backend=simple --set feather=2
    python src/imgtool/cli.py backends
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # 允许 `python src/imgtool/cli.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imgtool.core import io_utils  # noqa: E402
from imgtool.core.config import AppConfig  # noqa: E402
from imgtool.core.log import get_logger, setup_logging  # noqa: E402
from imgtool.core.pipeline import Pipeline  # noqa: E402
from imgtool.core.registry import discover_modules, get_module  # noqa: E402
from imgtool.core.types import ImageItem, ModuleResult, TaskContext  # noqa: E402

log = get_logger("cli")


def _parse_set(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--set 需要 key=value 形式：{item}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def cmd_list(args: argparse.Namespace) -> int:
    modules = discover_modules(force=True)
    print(f"共 {len(modules)} 个模块：")
    for module in modules:
        flag = "占位" if module.placeholder else "可用"
        print(f"  [{flag}] {module.name:<10} {module.title}　{module.description}")
        if module.placeholder:
            continue
        for spec in module.param_spec():
            default = spec.default
            extra = ""
            if spec.kind == "choice":
                extra = "  可选：" + " / ".join(str(v) for v, _ in (spec.choices or []))
            print(f"        --set {spec.key}={default}{extra}")
    return 0


def cmd_backends(args: argparse.Namespace) -> int:
    from imgtool.modules.matting.backends import backend_status
    from imgtool.modules.matting.backends.onnx_backend import list_local_models, models_dir

    config = AppConfig.load()
    print(f"模型目录：{models_dir(config)}")
    local = list_local_models(config)
    print(f"本地权重：{', '.join(local) if local else '（无）'}")
    print("抠图后端：")
    for name, label, ok, reason in backend_status(config):
        print(f"  [{'可用' if ok else '不可用'}] {name:<10} {label}" + (f"　{reason}" if reason else ""))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    module = get_module(args.module)
    if module is None:
        print(f"未找到模块：{args.module}", file=sys.stderr)
        return 2
    if module.placeholder:
        print(f"模块 {module.name} 尚未实现", file=sys.stderr)
        return 2

    files = io_utils.list_images(args.input, recursive=not args.no_recursive)
    if not files:
        print("没有找到可处理的图片", file=sys.stderr)
        return 2

    config = AppConfig.load()
    module.config = config

    params = _parse_set(args.set)
    if args.out:
        params["out_dir"] = args.out
    if args.output_mode:
        params["output_mode"] = args.output_mode

    print(f"模块：{module.title}　图片：{len(files)} 张")
    pipe = Pipeline(module, params, config)
    for key, value in sorted(pipe.params.items()):
        if not key.startswith("_"):
            print(f"  {key} = {value}")

    items = [ImageItem(path) for path in files]
    done = {"n": 0}

    def on_progress(count: int, total: int, message: str) -> None:
        done["n"] = count
        if message:
            print(f"  [{count}/{total}] {message}")

    ctx = TaskContext(config=config, on_progress=on_progress)
    def release_pixels(result: ModuleResult) -> None:
        # CLI 只需要导出路径，不保留整批的完整像素。
        result.item.rgb = None
        result.item.alpha = None
        for key in ("result_rgb", "preview_alpha", "edit_base_alpha"):
            result.item.meta.pop(key, None)

    results = pipe.run(items, ctx, on_item_done=release_pixels)

    ok = sum(1 for r in results if r.ok)
    failed = [r for r in results if not r.ok]
    print(f"\n完成 {ok}/{len(results)}")
    for result in failed:
        print(f"  失败：{result.item.name} → {result.message}")
    if results and results[0].output_dir:
        print(f"输出目录：{results[0].output_dir}")
    return 0 if not failed and len(results) == len(items) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imgtool", description="图片处理工具箱命令行", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出模块与参数")
    p_list.set_defaults(func=cmd_list)

    p_back = sub.add_parser("backends", help="查看抠图后端与权重状态")
    p_back.set_defaults(func=cmd_backends)

    p_run = sub.add_parser("run", help="批量运行某个模块")
    p_run.add_argument("module", help="模块名，例如 matting")
    p_run.add_argument("-i", "--input", nargs="+", required=True, help="图片文件或目录（可多个）")
    p_run.add_argument("-o", "--out", default="", help="输出目录，默认 output/<模块>/<时间戳>")
    p_run.add_argument("--set", action="append", help="设置参数，例如 --set backend=onnx")
    p_run.add_argument("--output-mode", default="", choices=["", "rgba", "color", "mask", "all"])
    p_run.add_argument("--no-recursive", action="store_true", help="目录不递归")
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AppConfig.load()
    setup_logging(args.log_level or str(config.get("log_level", "INFO")))
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
