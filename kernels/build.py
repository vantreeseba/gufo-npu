#!/usr/bin/env python3
"""Compile gufo-npu NPU designs to an xclbin plus instruction sequences.

Each design file (``--design PATH``) exposes ``compile_to(out_dir: Path,
dev: str)``. It writes <name>.xclbin and <name>.insts.elf unless it lists
other files in ``OUTPUTS``. A design with other outputs keeps its shapes in a
pure-data module next to it, <name>_variants.py (no IRON imports), which
defines ``OUTPUTS``; ``--list-outputs`` reads only that module, so CMake can
ask for a design's outputs at configure time without the kernel toolchain.
Compiling needs mlir-aie and llvm-aie (Peano) but no NPU.
"""

import argparse
import importlib.util
from pathlib import Path
import sys

# Keep the source tree clean when designs are imported from it.
sys.dont_write_bytecode = True


def load_module(path: Path, name: str):
    # A design imports its variants module as a sibling.
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design_path(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise SystemExit(f"unknown design: {path} does not exist")
    return path


def outputs_of(path: Path) -> list[str]:
    variants = path.with_name(f"{path.stem}_variants.py")
    if variants.is_file():
        return list(load_module(variants, f"gufo_npu_variants_{path.stem}").OUTPUTS)
    return [f"{path.stem}.xclbin", f"{path.stem}.insts.elf"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default="npu2", choices=["npu2"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--design", type=design_path, action="append", required=True)
    parser.add_argument(
        "--list-outputs",
        action="store_true",
        help="print each design's output file names, one per line, and exit",
    )
    args = parser.parse_args()
    if args.list_outputs:
        for path in args.design:
            print("\n".join(outputs_of(path)))
        return 0
    if args.out is None:
        parser.error("--out is required to compile")
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.design:
        name = path.stem
        design = load_module(path, f"gufo_npu_design_{name}")
        design.compile_to(args.out, args.dev)
        outputs = getattr(design, "OUTPUTS", [f"{name}.xclbin", f"{name}.insts.elf"])
        if list(outputs) != outputs_of(path):
            print(f"FAIL: {name}: OUTPUTS differ from what --list-outputs reports", file=sys.stderr)
            return 1
        for output in outputs:
            if not (args.out / output).is_file():
                print(f"FAIL: {name} did not produce {args.out / output}", file=sys.stderr)
                return 1
        print(f"compiled {name} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
