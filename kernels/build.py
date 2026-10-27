#!/usr/bin/env python3
"""Compile gufo-npu NPU designs to an xclbin plus an instruction ELF.

Each designs/<name>.py exposes ``compile_to(out_dir: Path, dev: str)``. It
writes <name>.xclbin and <name>.insts.elf unless it lists other files in
``OUTPUTS``.
Compiling needs mlir-aie and llvm-aie (Peano) but no NPU.
"""

import argparse
import importlib.util
from pathlib import Path
import sys

# Keep the source tree clean when designs are imported from it.
sys.dont_write_bytecode = True

DESIGNS_DIR = Path(__file__).resolve().parent / "designs"


def load_design(name: str):
    path = DESIGNS_DIR / f"{name}.py"
    if not path.is_file():
        raise SystemExit(f"unknown design {name!r}: {path} does not exist")
    spec = importlib.util.spec_from_file_location(f"gufo_npu_design_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default="npu2", choices=["npu2"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("designs", nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.designs:
        design = load_design(name)
        design.compile_to(args.out, args.dev)
        outputs = getattr(design, "OUTPUTS", [f"{name}.xclbin", f"{name}.insts.elf"])
        for output in outputs:
            if not (args.out / output).is_file():
                print(f"FAIL: {name} did not produce {args.out / output}", file=sys.stderr)
                return 1
        print(f"compiled {name} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
