"""DDR bandwidth probe: copy a 64 MiB int32 buffer through every column.

Uses both shim DMA channels in all 8 columns (mlir-aie's reference memcpy
layout). `gufo-npu bench bandwidth` times it; bytes moved = 2 x size (read
plus write). Decode streams weights from DDR, so this bounds tok/s.
"""

from pathlib import Path

import aie.iron as iron
from aie.iron import CompileTime, In, Out, kernels
from aie.iron.algorithms import transform_parallel
from aie.iron.device import from_name
import numpy as np

# Must match kBandwidthElements in src/npu/device_xrt.cpp.
SIZE = 16 * 1024 * 1024
TILE = 1024


@iron.jit
def bw_memcpy(inp: In, out: Out, *, size: CompileTime[int] = SIZE):
    return transform_parallel(
        kernels.passthrough(tile_size=TILE, dtype=np.int32),
        np.ndarray[(size,), np.dtype[np.int32]],
        tile_size=TILE,
        num_channels=2,
        pass_size_to_kernel=True,
    )


def compile_to(out_dir: Path, dev: str) -> None:
    iron.set_current_device(from_name(dev, n_cols=None))
    bw_memcpy.specialize(size=SIZE).compile(
        xclbin_path=out_dir / "bw_memcpy.xclbin",
        inst_path=out_dir / "bw_memcpy.insts.bin",
        elf_path=out_dir / "bw_memcpy.insts.elf",
    )
