"""Toolchain smoke test: one compute tile adds 1 to an int32 vector.

Proves the compile chain (IRON -> mlir-aie -> Peano -> xclbin) and the host
path (XRT hw context, ELF instructions, buffers). `gufo-npu smoke` runs it
and times the round trip of each dispatch.
"""

from pathlib import Path

import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.algorithms import transform
from aie.iron.device import from_name
import numpy as np

# Must match kSmokeElements in src/npu/device_xrt.cpp.
PROBLEM_SIZE = 4096
TILE_SIZE = 64


@iron.jit
def smoke_add_one(
    inp: In,
    out: Out,
    *,
    problem_size: CompileTime[int] = PROBLEM_SIZE,
    tile_size: CompileTime[int] = TILE_SIZE,
):
    tensor_ty = np.ndarray[(problem_size,), np.dtype[np.int32]]
    return transform(lambda x: x + 1, tensor_ty, tile_size=tile_size)


def compile_to(out_dir: Path, dev: str) -> None:
    iron.set_current_device(from_name(dev, n_cols=1))
    smoke_add_one.specialize(problem_size=PROBLEM_SIZE, tile_size=TILE_SIZE).compile(
        xclbin_path=out_dir / "smoke_add_one.xclbin",
        inst_path=out_dir / "smoke_add_one.insts.bin",
        elf_path=out_dir / "smoke_add_one.insts.elf",
    )
