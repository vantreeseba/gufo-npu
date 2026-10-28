"""Qwen3.6-35B-A3B prefill: compact q4nx weights (the decoder's own buffers,
src/core/q4nx.hpp) to bfp16ebs8, written in the order the prefill GEMM
(prefill_gemm.py, IRON's flm.GEMM at tile_n 64) reads its B operand.

Adapted from IRON's iron.operators.flm.dequant design (Apache-2.0): same
array use (4 rows x 8 columns, each column one 64-out-feature block at a
time, each core one q4nx block per object), with two changes:
- The core kernel (aie/prefill_dequant.cc) reads the compact 4736-byte block
  (8-bit sub-scales and minima under bf16 d and dmin) and rounds each weight
  once, from fp32 to bfp16.
- The fill reads the GEMV's block order (tiles of 4 blocks per k block, see
  q4nx::Block), so the decoder's matrices and expert arenas convert in place
  with no host repack. A 64-row block is tiles 2j and 2j + 1 of a 128-row
  group: two adjacent blocks per k block, 4 blocks apart.

Runtime arguments: QW (q4nx bytes, rows padded to 1024), OUT (bfp16ebs8, K x N
in flm.GEMM's packed order). One sequence per (K, N)
(prefill_dequant_variants.py); a call on a sub-buffer converts that part of a
stacked matrix (e.g. one expert of an arena).
"""

import filecmp
from pathlib import Path
import shutil

import numpy as np

import aie.iron as iron
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.helpers.util import v8bfp16ebs8
from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, TaskGroup
from aie.iron import Worker
from aie.iron.device import from_name
from aie.utils import config
from aie.utils.compile.utils import compile_mlir_module

from iron.operators.flm.gemm.design import (
    BFP16_GROUP,
    CT_MAX_K_FOR_N,
    K_TILE as K_TILE_B,
    N_TILE_DEFAULT as N_TILE,
    S,
    T,
)

import prefill_dequant_variants as V  # noqa: E402 (sibling; build.py puts it on the path)

XCLBIN = V.DEQUANT_NAME
KERNEL_SOURCE = Path(__file__).resolve().parent / "aie" / "prefill_dequant.cc"

M_TILE, K_TILE, GROUP = 32, 256, 32
BLOCK_BYTES = M_TILE * K_TILE // 2 + M_TILE * K_TILE // GROUP * 2 + M_TILE * 4
assert BLOCK_BYTES == 4736
GEMV_ROWS = 4  # q4nx tiles per GEMV column round (q4nx::kRows)
CT_K = CT_MAX_K_FOR_N[N_TILE]
ROWS, COLS = 4, 8

CORE_BLOCKS = M_TILE * K_TILE // BFP16_GROUP
SLAB_BLOCKS = N_TILE * K_TILE_B // BFP16_GROUP
HALF_BLOCKS = SLAB_BLOCKS // 2
RUN = (M_TILE // T) * (CT_K // S) * T
SPLIT = 2
DRAIN_DIMS = [
    (N_TILE // M_TILE, RUN),
    (K_TILE // CT_K, (N_TILE // T) * (CT_K // S) * T),
    (SPLIT, RUN // SPLIT),
    (RUN // SPLIT, 1),
]
DRAIN_SIZES = [d[0] for d in DRAIN_DIMS]
DRAIN_STRIDES = [d[1] for d in DRAIN_DIMS]
CORE_JOIN_OFFSETS = [0, CORE_BLOCKS]
HALVES = 2
# A block in 128-byte pieces, inside the BD's inner-size field.
PIECE = 128
assert BLOCK_BYTES % PIECE == 0

SHAPES = V.DEQUANT_SHAPES
sequence_name = V.dequant_sequence_name
OUTPUTS = V.OUTPUTS


def qw_bytes(k, n):
    return n * k * BLOCK_BYTES // (M_TILE * K_TILE)


def dequant_program(dev, K, N, cols=COLS):
    if K % K_TILE_B or N % 1024:
        raise ValueError(f"({K}, {N}): K must be a multiple of {K_TILE_B}, N of 1024")
    k_tiles = K // K_TILE_B
    blocks_per_row = K // K_TILE
    n_blocks = N // N_TILE
    out_blocks = K * N // BFP16_GROUP

    qw_l3_ty = np.ndarray[(qw_bytes(K, N),), np.dtype[np.uint8]]
    out_l3_ty = np.ndarray[(out_blocks,), np.dtype[v8bfp16ebs8]]
    qw_col_ty = np.ndarray[(ROWS * BLOCK_BYTES,), np.dtype[np.uint8]]
    qw_blk_ty = np.ndarray[(BLOCK_BYTES,), np.dtype[np.uint8]]
    out_half_ty = np.ndarray[(HALF_BLOCKS,), np.dtype[v8bfp16ebs8]]
    out_blk_ty = np.ndarray[(CORE_BLOCKS,), np.dtype[v8bfp16ebs8]]

    kernel = ExternalFunction(
        "qwen36_q4nx_dequant",
        source_file=str(KERNEL_SOURCE),
        arg_types=[qw_blk_ty, out_blk_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[
            f"-DQ4NX_M_TILE={M_TILE}",
            f"-DQ4NX_K_TILE={K_TILE}",
            f"-DQ4NX_GROUP={GROUP}",
            f"-DQ4NX_CT_K={CT_K}",
            f"-DQ4NX_S={S}",
            f"-DQ4NX_T={T}",
        ],
    )

    def core_body(qw_in, out_of, k):
        qw = qw_in.acquire(1)
        out = out_of.acquire(1)
        k(qw, out)
        qw_in.release(1)
        out_of.release(1)

    workers, qw_prods, out_conses = [], [], []
    for c in range(cols):
        of_qw = ObjectFifo(qw_col_ty, name=f"qw_{c}", depth=2)
        qw_prods.append(of_qw.prod())
        qw_cores = of_qw.cons().split(
            [BLOCK_BYTES * r for r in range(ROWS)],
            obj_types=[qw_blk_ty] * ROWS,
            names=[f"qw_{c}_{r}" for r in range(ROWS)],
        )
        out_cores, halves = [], []
        for h in range(HALVES):
            of_out = ObjectFifo(out_half_ty, name=f"w_{c}_{h}", depth=2)
            halves.append(of_out.cons())
            out_cores += of_out.prod().join(
                CORE_JOIN_OFFSETS,
                obj_types=[out_blk_ty] * (ROWS // HALVES),
                names=[f"w_{c}_{h}_{r}" for r in range(ROWS // HALVES)],
            )
        out_conses.append(halves)
        workers += [
            Worker(
                core_body,
                [qw_cores[r].cons(), out_cores[r].prod(), kernel],
                stack_size=2048,
            )
            for r in range(ROWS)
        ]

    # 64-row block cb, per k block: its two tiles, adjacent in the GEMV order.
    qw_sizes = [blocks_per_row, 2, BLOCK_BYTES // PIECE, PIECE]
    qw_strides = [GEMV_ROWS * BLOCK_BYTES, BLOCK_BYTES, PIECE, 1]

    def sequence(QW, OUT, qw_prod_hs, out_cons_hs):
        for cb0 in range(0, n_blocks, cols):
            columns = [(c, cb0 + c) for c in range(cols) if cb0 + c < n_blocks]
            tg_fill = TaskGroup()
            for c, cb in columns:
                offset = ((cb // 2) * blocks_per_row * GEMV_ROWS + (cb % 2) * 2) * BLOCK_BYTES
                qw_prod_hs[c].fill(
                    QW,
                    TensorAccessPattern((1, qw_bytes(K, N)), offset, qw_sizes, qw_strides),
                    group=tg_fill,
                )
            prev = None
            for kb in range(k_tiles):
                tg = TaskGroup()
                for c, cb in columns:
                    for h in range(HALVES):
                        out_cons_hs[c][h].drain(
                            OUT,
                            TensorAccessPattern(
                                (1, out_blocks),
                                (cb * k_tiles + kb) * SLAB_BLOCKS + h * HALF_BLOCKS,
                                DRAIN_SIZES,
                                DRAIN_STRIDES,
                            ),
                            wait=True,
                            group=tg,
                        )
                if prev is not None:
                    prev.finish()
                prev = tg
            prev.finish()
            # Never awaited before the drains: a core reads its fill before it
            # writes the output a drain takes (as IRON's design does). Each
            # column's fill channel only carries fills, and the drains above
            # are awaited in queue order per channel.
            tg_fill.finish()

    rt = Runtime(sequence, [qw_l3_ty, out_l3_ty, qw_prods, out_conses])
    return Program(dev, rt, workers=workers).resolve_program()


def compile_to(out_dir: Path, dev: str) -> None:
    device = from_name(dev, n_cols=None)
    iron.set_current_device(device)
    first = None
    for k, n in SHAPES:
        name = sequence_name(k, n)
        work_dir = out_dir / f"{name}.work"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
        compile_mlir_module(
            dequant_program(device, k, n),
            device=device,
            work_dir=work_dir,
            xclbin_path=work_dir / f"{name}.xclbin",
            insts_path=out_dir / f"{name}.insts.bin",
            elf_path=out_dir / f"{name}.insts.elf",
        )
        cores = sorted(
            p.relative_to(work_dir)
            for pattern in ("elfs_main_core_*/*.elf", "cdo_main/*.bin")
            for p in work_dir.glob(pattern)
        )
        if not cores:
            raise RuntimeError(f"{name}: no core ELFs or CDOs in {work_dir}")
        if first is None:
            first = (work_dir, cores)
            shutil.copyfile(work_dir / f"{name}.xclbin", out_dir / f"{XCLBIN}.xclbin")
        elif cores != first[1] or not all(
            filecmp.cmp(first[0] / c, work_dir / c, shallow=False) for c in cores
        ):
            raise RuntimeError(f"{name}: core program differs between shapes")
