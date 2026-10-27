"""q4nx GEMV for decode on the whole array: y (n_out) = W (n_out x dim) x.

Every core (8 columns x 4 rows) owns M_TILE out-features per round and
accumulates one q4nx block per K_TILE in-features. Each column's weights come
in through its own shim DMA as one contiguous slice per round, so W must be
packed in this design's order (src/core/q4nx.hpp):

    out-feature n = ((round * COLS + col) * ROWS + row) * M_TILE + m
    block index   = ((round * COLS + col) * K_BLOCKS + k) * ROWS + row

One core program serves every shape: each core reads its k block count and
round count from a runtime parameter that the shape's instruction sequence
writes before it streams. The build emits one xclbin (q4nx_gemv.xclbin) and
one instruction ELF per shape (q4nx_gemv_k<dim>_n<rows>.insts.elf), all for
the same hardware context. The token buffer always carries MAX_DIM values;
the tail past dim is ignored.

Adapted from IRON's iron.operators.flm.lm_head design (Apache-2.0).
Runtime arguments: x (bf16), w (q4nx bytes as uint32), y (fp32).
"""

import filecmp
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
MAX_DIM = 6144
# The sequence finishes each round's TaskGroup DEPTH rounds later, which frees
# its BDs for reuse (a shim tile holds 16).
DEPTH = 2
STACK_SIZE = 4 * 1024

KERNEL_SOURCE = Path(__file__).resolve().parent.parent / "aie" / "q4nx_gemv.cc"
XCLBIN = "q4nx_gemv"

bf16 = np.dtype[bfloat16]

# Decode shapes of Qwen3.5-2B as (dim, padded rows): mixer outputs, fused
# attention input (q + gate, k, v), fused DeltaNet input (qkv + z), fused
# gate + up, down, and the LM head (248320 rows padded to 248832).
SHAPES = [
    (2048, 2048),
    (2048, 5120),
    (2048, 8192),
    (2048, 12288),
    (6144, 2048),
    (2048, 248832),
]


def sequence_name(dim: int, rows: int) -> str:
    return f"{XCLBIN}_k{dim}_n{rows}"


# build.py checks these: the xclbin, then one instruction ELF per shape.
OUTPUTS = [f"{XCLBIN}.xclbin"] + [f"{sequence_name(d, n)}.insts.elf" for d, n in SHAPES]


def _kernels():
    x_ty = np.ndarray[(MAX_DIM,), bf16]
    sums_ty = np.ndarray[(2 * MAX_DIM // GROUP,), bf16]  # hi, then lo
    y_acc_ty = np.ndarray[(M_TILE,), np.dtype[np.float32]]
    y_ty = np.ndarray[(M_TILE,), np.dtype[np.float32]]
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
            f"-DGEMV_MAX_DIM={MAX_DIM}",
            # Without it the bf16 mmul emulation runs about 8x slower.
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
    )
    bind = block.object_file.bind
    prep = bind("gemv_prep", [x_ty, sums_ty, np.int32])
    zero = bind("gemv_zero", [y_acc_ty])
    store = bind("gemv_store", [y_ty, y_acc_ty])
    return block, prep, zero, store, (x_ty, sums_ty, y_acc_ty, y_ty, w_blk_ty)


def gemv_program(dev, dim: int, n_out: int):
    """The design with the instruction sequence for one shape."""
    if dim % K_TILE or dim > MAX_DIM or n_out % OUT_PER_ROUND:
        raise ValueError(f"unsupported shape dim={dim} n_out={n_out}")
    rounds = n_out // OUT_PER_ROUND
    k_blocks = dim // K_TILE
    block, prep, zero, store, types = _kernels()
    x_ty, sums_ty, y_acc_ty, y_blk_ty, w_blk_ty = types
    (w_blk,) = w_blk_ty.__args__[0]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]  # k blocks, rounds

    w_col_ty = np.ndarray[(ROWS, w_blk), bf16]
    y_col_ty = np.ndarray[(ROWS, M_TILE), np.dtype[np.float32]]
    shims = [Tile(j, 0) for j in range(COLS)]

    def core_fn(x_in, w_in, y_out, k_prep, k_zero, k_block, k_store, y_acc, sums, rtp):
        # The sequence writes the RTP before it sends the token, so it is
        # current once the token arrives.
        x = x_in.acquire(1)
        k_prep(x, sums, rtp[0])
        for _ in range_(rtp[1]):
            k_zero(y_acc)
            for k in range_(rtp[0]):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, sums, k)
                w_in.release(1)
            y = y_out.acquire(1)
            k_store(y, y_acc)
            y_out.release(1)
        x_in.release(1)

    words_per_col = n_out * dim * 5 // 8 // 4 // (rounds * COLS)
    w_l3_ty = np.ndarray[(n_out * dim * 5 // 8 // 4,), np.dtype[np.uint32]]
    y_l3_ty = np.ndarray[(n_out,), np.dtype[np.float32]]

    of_x = ObjectFifo(x_ty, name="x", depth=1)
    x_prod = of_x.prod(tile=shims[0])

    workers, w_prods, y_conses, rtps = [], [], [], []
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
            rtp = Buffer(type=rtp_ty, name=f"rtp_{i}_{j}", use_write_rtp=True)
            rtps.append(rtp)
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
                        rtp,
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

    def sequence(X, W, Y, x_prod, w_prods, y_conses, rtps):
        for rtp in rtps:
            rtp[0] = k_blocks
            rtp[1] = rounds
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

    rt = Runtime(sequence, [x_ty, w_l3_ty, y_l3_ty, x_prod, w_prods, y_conses, rtps])
    return Program(dev, rt, workers=workers).resolve_program()


def compile_to(out_dir: Path, dev: str) -> None:
    device = from_name(dev, n_cols=None)
    iron.set_current_device(device)
    first_cores = None
    for dim, rows in SHAPES:
        name = sequence_name(dim, rows)
        # IRON reuses any kernel object already in the work dir, even when
        # its source changed, so every build starts from an empty one.
        work_dir = out_dir / f"{name}.work"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
        compile_mlir_module(
            gemv_program(device, dim, rows),
            device=device,
            work_dir=work_dir,
            xclbin_path=work_dir / f"{name}.xclbin",
            insts_path=out_dir / f"{name}.insts.bin",
            elf_path=out_dir / f"{name}.insts.elf",
        )
        # Every shape must build the same array configuration (core programs,
        # routing, locks, tile DMAs), or its sequence would not match the
        # shared xclbin.
        cores = sorted(
            p.relative_to(work_dir)
            for pattern in ("elfs_main_core_*/*.elf", "cdo_main/*.bin")
            for p in work_dir.glob(pattern)
        )
        if not cores:
            raise RuntimeError(f"{name}: no core ELFs or CDOs in {work_dir}")
        if first_cores is None:
            first_cores = (work_dir, cores)
            shutil.copyfile(work_dir / f"{name}.xclbin", out_dir / f"{XCLBIN}.xclbin")
        else:
            first_dir, first = first_cores
            if cores != first or not all(
                filecmp.cmp(first_dir / c, work_dir / c, shallow=False) for c in cores
            ):
                raise RuntimeError(f"{name}: core program differs between shapes")
