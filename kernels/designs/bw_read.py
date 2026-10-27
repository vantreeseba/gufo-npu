"""Read-only DDR bandwidth probe: both shim MM2S channels of all 8 columns
stream a 256 MiB buffer into cores that discard it; each core returns 64
bytes at the end so the sequence has something to wait on.

`gufo-npu bench bandwidth` times it next to bw_memcpy. Decode only reads
weights, so this is its ceiling (~56 GB/s; one channel per column gives the
same).
"""

from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.utils.compile.utils import compile_mlir_module

# Must match kReadElements in src/npu/device_xrt.cpp.
SIZE = 64 * 1024 * 1024  # int32
TILE = 4096  # int32 per object (16 KiB)
COLS = 8
CHANNELS = 2


def program(dev):
    streams = COLS * CHANNELS
    per = SIZE // streams
    objects = per // TILE
    in_ty = np.ndarray[(TILE,), np.dtype[np.int32]]
    out_ty = np.ndarray[(16,), np.dtype[np.int32]]
    in_l3 = np.ndarray[(SIZE,), np.dtype[np.int32]]
    out_l3 = np.ndarray[(streams * 16,), np.dtype[np.int32]]

    def body(a, o):
        for _ in range_(objects):
            a.acquire(1)
            a.release(1)
        o.acquire(1)
        o.release(1)

    ins, outs, workers = [], [], []
    for c in range(COLS):
        for k in range(CHANNELS):
            fi = ObjectFifo(in_ty, name=f"in{c}_{k}", depth=2)
            fo = ObjectFifo(out_ty, name=f"out{c}_{k}", depth=1)
            ins.append(fi.prod(tile=Tile(c, 0)))
            outs.append(fo.cons(tile=Tile(c, 0)))
            workers.append(Worker(body, [fi.cons(), fo.prod()]))

    def sequence(A, O, ins, outs):
        tg = TaskGroup()
        for i in range(streams):
            tap = TensorAccessPattern(
                (1, SIZE), i * per, [1, 1, objects, TILE], [0, 0, TILE, 1]
            )
            ins[i].fill(A, tap, group=tg)
        for i in range(streams):
            tap = TensorAccessPattern(
                (1, streams * 16), i * 16, [1, 1, 1, 16], [0, 0, 0, 1]
            )
            outs[i].drain(O, tap, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [in_l3, out_l3, ins, outs])
    return Program(dev, rt, workers=workers).resolve_program()


def compile_to(out_dir: Path, dev: str) -> None:
    device = from_name(dev, n_cols=None)
    iron.set_current_device(device)
    work_dir = out_dir / "bw_read.work"
    work_dir.mkdir(parents=True, exist_ok=True)
    compile_mlir_module(
        program(device),
        device=device,
        work_dir=work_dir,
        xclbin_path=out_dir / "bw_read.xclbin",
        insts_path=out_dir / "bw_read.insts.bin",
        elf_path=out_dir / "bw_read.insts.elf",
    )
