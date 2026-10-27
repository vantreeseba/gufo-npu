"""q4nx GEMV for decode on the whole array: y (n_out) = W (n_out x dim) x.

Every core (8 columns x 4 rows) owns M_TILE out-features per round and
accumulates one q4nx block per K_TILE in-features. Each column's weights come
in through its own shim DMA as one contiguous slice per round, so W must be
packed in this design's order (see src/npu/q4nx.hpp, PackQ4nx):

    out-feature n = ((round * COLS + col) * ROWS + row) * M_TILE + m
    block index   = ((round * COLS + col) * K_BLOCKS + k) * ROWS + row

With ``norm`` the token arrives followed by its RMS norm weight (eps 1e-6)
and is normalized on-core first.

Adapted from IRON's iron.operators.flm.lm_head design (Apache-2.0).
Runtime arguments: x (bf16), w (q4nx bytes as uint32), y (bf16).
"""

from pathlib import Path
import shutil

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, ExternalFunction, ObjectFifo, Program, Runtime
from aie.iron import TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.utils import config
from aie.utils.compile.utils import compile_mlir_module

M_TILE, K_TILE, GROUP = 32, 256, 32
BLOCK_BYTES = M_TILE * K_TILE * 5 // 8  # 4-bit codes + bf16 scale and min per 32
COLS, ROWS = 8, 4
OUT_PER_ROUND = COLS * ROWS * M_TILE
# The sequence finishes each round's TaskGroup DEPTH rounds later, which frees
# its BDs for reuse (a shim tile holds 16).
DEPTH = 2
STACK_SIZE = 4 * 1024

KERNEL_SOURCE = Path(__file__).resolve().parent.parent / "aie" / "q4nx_gemv.cc"

bf16 = np.dtype[bfloat16]

# Decode shapes of Qwen3.5-2B: (name, dim, n_out, norm). n_out is padded to a
# multiple of OUT_PER_ROUND (the LM head's 248320 rows become 248832).
SHAPES = [
    ("q4nx_gemv_k2048_n2048", 2048, 2048, False),
    ("q4nx_gemv_k2048_n6144", 2048, 6144, False),
    ("q4nx_gemv_k6144_n2048", 6144, 2048, False),
    ("q4nx_gemv_k2048_n248832_norm", 2048, 248832, True),
]
ARTIFACTS = [name for name, *_ in SHAPES]


def _kernels(dim: int, norm: bool):
    x_ty = np.ndarray[((2 if norm else 1) * dim,), bf16]
    sums_ty = np.ndarray[(dim // GROUP,), bf16]
    y_acc_ty = np.ndarray[(M_TILE,), np.dtype[np.float32]]
    y_ty = np.ndarray[(M_TILE,), bf16]
    w_blk_ty = np.ndarray[(BLOCK_BYTES // 2,), bf16]
    block = ExternalFunction(
        "gemv_block",
        source_file=str(KERNEL_SOURCE),
        arg_types=[w_blk_ty, x_ty, y_acc_ty, sums_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[
            f"-DQ4NX_M_TILE={M_TILE}",
            f"-DQ4NX_K_TILE={K_TILE}",
            f"-DQ4NX_GROUP={GROUP}",
            f"-DGEMV_DIM={dim}",
            f"-DGEMV_NORM={int(norm)}",
            # Without it the bf16 mmul emulation runs about 8x slower.
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
    )
    bind = block.object_file.bind
    prep = bind("gemv_prep", [x_ty, sums_ty])
    zero = bind("gemv_zero", [y_acc_ty])
    store = bind("gemv_store", [y_ty, y_acc_ty])
    return block, prep, zero, store, (x_ty, sums_ty, y_acc_ty, y_ty, w_blk_ty)


def gemv_program(dev, dim: int, n_out: int, norm: bool):
    if dim % K_TILE or n_out % OUT_PER_ROUND:
        raise ValueError(f"unsupported shape dim={dim} n_out={n_out}")
    rounds = n_out // OUT_PER_ROUND
    k_blocks = dim // K_TILE
    block, prep, zero, store, types = _kernels(dim, norm)
    x_ty, sums_ty, y_acc_ty, y_blk_ty, w_blk_ty = types
    (w_blk,) = w_blk_ty.__args__[0]

    w_col_ty = np.ndarray[(ROWS, w_blk), bf16]
    y_col_ty = np.ndarray[(ROWS, M_TILE), bf16]
    shims = [Tile(j, 0) for j in range(COLS)]

    def core_fn(x_in, w_in, y_out, k_prep, k_zero, k_block, k_store, y_acc, sums):
        x = x_in.acquire(1)
        k_prep(x, sums)
        for _ in range_(rounds):
            k_zero(y_acc)
            for k in range_(k_blocks):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, sums, k)
                w_in.release(1)
            y = y_out.acquire(1)
            k_store(y, y_acc)
            y_out.release(1)
        x_in.release(1)

    words_per_col = n_out * dim * 5 // 8 // 4 // (rounds * COLS)
    x_l3_ty = x_ty
    w_l3_ty = np.ndarray[(n_out * dim * 5 // 8 // 4,), np.dtype[np.uint32]]
    y_l3_ty = np.ndarray[(n_out,), bf16]

    of_x = ObjectFifo(x_ty, name="x", depth=1)
    x_prod = of_x.prod(tile=shims[0])

    workers, w_prods, y_conses = [], [], []
    for j in range(COLS):
        of_w = ObjectFifo(w_col_ty, name=f"w{j}")
        w_prods.append(of_w.prod(tile=shims[j]))
        w_cores = of_w.cons().split(
            [w_blk * i for i in range(ROWS)],
            obj_types=[w_blk_ty] * ROWS,
            names=[f"w{j}_{i}" for i in range(ROWS)],
        )
        of_y = ObjectFifo(y_col_ty, name=f"y{j}")
        y_conses.append(of_y.cons(tile=shims[j]))
        y_cores = of_y.prod().join(
            [M_TILE * i for i in range(ROWS)],
            obj_types=[y_blk_ty] * ROWS,
            names=[f"y{j}_{i}" for i in range(ROWS)],
        )
        for i in range(ROWS):
            workers.append(
                Worker(
                    core_fn,
                    [
                        of_x.cons(),
                        w_cores[i].cons(),
                        y_cores[i].prod(),
                        prep,
                        zero,
                        block,
                        store,
                        Buffer(type=y_acc_ty, name=f"yacc_{i}_{j}"),
                        Buffer(type=sums_ty, name=f"sums_{i}_{j}"),
                    ],
                    stack_size=STACK_SIZE,
                )
            )

    def row_taps(count, width):
        return [
            TensorAccessPattern((count, width), i * width, [1, 1, 1, width], [0, 0, 0, 1])
            for i in range(count)
        ]

    y_taps = row_taps(rounds * COLS, ROWS * M_TILE)
    w_taps = row_taps(rounds * COLS, words_per_col)

    def sequence(X, W, Y, x_prod, w_prods, y_conses):
        # One broadcast sends the token to every core; it has no completion
        # token, so its BD is freed at the end.
        x_task = x_prod.fill(X, wait=False, managed=False)
        window = []
        for rnd in range(rounds):
            tg = TaskGroup()
            for col in range(COLS):
                i = rnd * COLS + col
                y_conses[col].drain(Y, y_taps[i], wait=True, group=tg)
                w_prods[col].fill(W, w_taps[i], wait=False, group=tg)
            window.append(tg)
            if len(window) > DEPTH:
                window.pop(0).finish()
        for tg in window:
            tg.finish()
        x_task.free()

    rt = Runtime(sequence, [x_l3_ty, w_l3_ty, y_l3_ty, x_prod, w_prods, y_conses])
    return Program(dev, rt, workers=workers).resolve_program()


def compile_to(out_dir: Path, dev: str) -> None:
    device = from_name(dev, n_cols=None)
    iron.set_current_device(device)
    for name, dim, n_out, norm in SHAPES:
        # IRON reuses any kernel object already in the work dir, even when
        # its source changed, so every build starts from an empty one.
        work_dir = out_dir / f"{name}.work"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
        compile_mlir_module(
            gemv_program(device, dim, n_out, norm),
            device=device,
            work_dir=work_dir,
            xclbin_path=out_dir / f"{name}.xclbin",
            insts_path=out_dir / f"{name}.insts.bin",
            elf_path=out_dir / f"{name}.insts.elf",
        )
