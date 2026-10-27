"""q4nx GEMV for decode on the whole array: y (n_out) = W (n_out x dim) x.

Every core (8 columns x 4 rows) owns M_TILE out-features per round and
accumulates one q4nx block per K_TILE in-features. Each column's weights come
in through its own shim DMA as one contiguous slice per round, so W must be
packed in this design's order (src/core/q4nx.hpp):

    out-feature n = ((round * COLS + col) * ROWS + row) * M_TILE + m
    block index   = ((round * COLS + col) * K_BLOCKS + k) * ROWS + row

One core program serves every call: each core reads its k block count,
round count and input/output stages from a runtime parameter that the
variant's instruction sequence writes before it streams. The build emits one
xclbin (q4nx_gemv.xclbin) and one instruction ELF per variant, all for the
same hardware context, so calls chain in one runlist.

The token buffer (X) is always 4 * MAX_DIM bytes; the tail past the input is
ignored. Input stages:
  plain    X holds dim fp32 values.
  act      X is a SwiGLU output (see below): the first 16 of every 32 fp32
           values, dim of them.
  resnorm  X holds [H | D | w]: dim fp32, dim fp32, dim bf16. x =
           rms_norm(H + D) * w, and H + D goes to R, the next residual: the
           four cores of column RES_COL each send a quarter of it down their
           output stream before their rows. dim must be HIDDEN.
Output stages:
  store    Y gets the rows' fp32 values.
  swiglu   each core's 32 rows are 16 gate rows then their 16 up rows; it
           writes silu(gate) * up to the first 16 of its 32 values.

Adapted from IRON's iron.operators.flm.lm_head design (Apache-2.0).
Runtime arguments: X, W (q4nx bytes as uint32), Y (fp32), R (HIDDEN fp32;
written only by resnorm).
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
HIDDEN = 2048  # residual width (resnorm)
RES_COL = 1  # the column whose cores hand out the residual
RES_PIECES = HIDDEN // ROWS // M_TILE  # output objects per core for it
RMS_EPS = 1e-6
# The sequence finishes each round's TaskGroup DEPTH rounds later, which frees
# its BDs for reuse (a shim tile holds 16).
DEPTH = 2
STACK_SIZE = 4 * 1024

KERNEL_SOURCE = Path(__file__).resolve().parent.parent / "aie" / "q4nx_gemv.cc"
XCLBIN = "q4nx_gemv"

bf16 = np.dtype[bfloat16]

PROLOGUES = {"plain": 0, "act": 0, "resnorm": 1}  # act = plain, strided X
EPILOGUES = {"store": 0, "swiglu": 1}

# Decode calls of Qwen3.5-2B as (dim, padded rows, input, output): mixer
# output, fused attention input (q + gate, k, v), fused DeltaNet input
# (qkv + z), fused gate + up with SwiGLU, down, and the LM head (248320 rows
# padded to 248832).
VARIANTS = [
    (2048, 2048, "plain", "store"),
    (2048, 5120, "resnorm", "store"),
    (2048, 8192, "resnorm", "store"),
    (2048, 12288, "resnorm", "swiglu"),
    (6144, 2048, "act", "store"),
    (2048, 248832, "resnorm", "store"),
]


def sequence_name(dim: int, rows: int, prologue: str, epilogue: str) -> str:
    return f"{XCLBIN}_k{dim}_n{rows}_{prologue}_{epilogue}"


# build.py checks these: the xclbin, then one instruction ELF per variant.
OUTPUTS = [f"{XCLBIN}.xclbin"] + [f"{sequence_name(*v)}.insts.elf" for v in VARIANTS]


def _kernels():
    x_ty = np.ndarray[(2 * MAX_DIM,), bf16]  # MAX_DIM fp32 of raw input
    sums_ty = np.ndarray[(2 * MAX_DIM // GROUP,), bf16]  # hi, then lo
    res_ty = np.ndarray[(HIDDEN,), np.dtype[np.float32]]
    y_ty_ = np.ndarray[(M_TILE,), np.dtype[np.float32]]
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
            f"-DGEMV_RMS_EPS={RMS_EPS}f",
            f"-DGEMV_HIDDEN={HIDDEN}",
            # Without it the bf16 mmul emulation runs about 8x slower.
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
    )
    bind = block.object_file.bind
    prep = bind("gemv_prep", [x_ty, sums_ty, np.int32, np.int32])
    residual = bind("gemv_residual", [x_ty, np.int32, np.int32])
    residual_out = bind("gemv_residual_out", [y_ty_, x_ty, np.int32, np.int32])
    zero = bind("gemv_zero", [y_acc_ty])
    store = bind("gemv_store", [y_ty, y_acc_ty, np.int32])
    kernels = dict(
        block=block,
        prep=prep,
        residual=residual,
        residual_out=residual_out,
        zero=zero,
        store=store,
    )
    return kernels, (x_ty, sums_ty, y_acc_ty, y_ty, w_blk_ty, res_ty)


def gemv_program(dev, dim: int, n_out: int, prologue: str, epilogue: str):
    """The design with the instruction sequence for one variant."""
    if dim % K_TILE or dim > MAX_DIM or n_out % OUT_PER_ROUND:
        raise ValueError(f"unsupported shape dim={dim} n_out={n_out}")
    if prologue == "resnorm" and dim != HIDDEN:
        raise ValueError("resnorm needs dim == HIDDEN")
    rounds = n_out // OUT_PER_ROUND
    k_blocks = dim // K_TILE
    k, types = _kernels()
    x_ty, sums_ty, y_acc_ty, y_blk_ty, w_blk_ty, res_ty = types
    (w_blk,) = w_blk_ty.__args__[0]
    (x_units,) = x_ty.__args__[0]
    # k blocks, rounds, input stage, output stage, residual pieces, core row.
    rtp_ty = np.ndarray[(6,), np.dtype[np.int32]]

    w_col_ty = np.ndarray[(ROWS, w_blk), bf16]
    y_col_ty = np.ndarray[(ROWS, M_TILE), np.dtype[np.float32]]
    shims = [Tile(j, 0) for j in range(COLS)]

    def body(x, w_in, y_out, k_zero, k_block, k_store, y_acc, sums, rtp):
        for _ in range_(rtp[1]):
            k_zero(y_acc)
            for kb in range_(rtp[0]):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, sums, kb)
                w_in.release(1)
            y = y_out.acquire(1)
            k_store(y, y_acc, rtp[3])
            y_out.release(1)

    # The sequence writes the RTP before it sends the token, so it is current
    # once the token arrives.
    def core_fn(
        x_in, w_in, y_out, k_residual, k_prep, k_zero, k_block, k_store, y_acc, sums, rtp
    ):
        x = x_in.acquire(1)
        k_residual(x, rtp[0], rtp[2])
        k_prep(x, sums, rtp[0], rtp[2])
        body(x, w_in, y_out, k_zero, k_block, k_store, y_acc, sums, rtp)
        x_in.release(1)

    # The residual column's cores first send their quarter of H + D (RTP word
    # 4 pieces; 0 when the input stage is not resnorm).
    def residual_core_fn(
        x_in,
        w_in,
        y_out,
        k_residual,
        k_residual_out,
        k_prep,
        k_zero,
        k_block,
        k_store,
        y_acc,
        sums,
        rtp,
    ):
        x = x_in.acquire(1)
        k_residual(x, rtp[0], rtp[2])
        for piece in range_(rtp[4]):
            y = y_out.acquire(1)
            k_residual_out(y, x, rtp[5], piece)
            y_out.release(1)
        k_prep(x, sums, rtp[0], rtp[2])
        body(x, w_in, y_out, k_zero, k_block, k_store, y_acc, sums, rtp)
        x_in.release(1)

    words_per_col = n_out * dim * 5 // 8 // 4 // (rounds * COLS)
    # act reads a SwiGLU output: 2 * dim fp32, as bf16 units.
    x_l3_units = 4 * dim if prologue == "act" else x_units
    x_l3_ty = np.ndarray[(x_l3_units,), bf16]
    w_l3_ty = np.ndarray[(n_out * dim * 5 // 8 // 4,), np.dtype[np.uint32]]
    y_l3_ty = np.ndarray[(n_out,), np.dtype[np.float32]]

    of_x = ObjectFifo(x_ty, name="x", depth=1)
    x_prod = of_x.prod(tile=shims[0])
    workers, w_prods, y_conses, rtps, rtp_rows = [], [], [], [], []
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
            rtp_rows.append(i)
            scratch = [
                Buffer(type=y_acc_ty, name=f"yacc_{i}_{j}"),
                Buffer(type=sums_ty, name=f"sums_{i}_{j}"),
                rtp,
            ]
            kernels = [k["prep"], k["zero"], k["block"], k["store"]]
            args = [of_x.cons(), w_cores[i].cons(), y_cores[i].prod(), k["residual"]]
            if j == RES_COL:
                fn = residual_core_fn
                args.append(k["residual_out"])
            else:
                fn = core_fn
            workers.append(Worker(fn, args + kernels + scratch, stack_size=STACK_SIZE))

    def row_taps(count, width):
        return [
            TensorAccessPattern((count, width), i * width, [1, 1, 1, width], [0, 0, 0, 1])
            for i in range(count)
        ]

    y_taps = row_taps(rounds * COLS, ROWS * M_TILE)
    w_taps = row_taps(rounds * COLS, words_per_col)
    if prologue == "act":
        # dim fp32 values: 16 of every 32, i.e. 32 of every 64 bf16 units.
        x_tap = TensorAccessPattern(
            (1, x_l3_units), 0, [1, 1, dim // 16, 32], [0, 0, 64, 1]
        )
    else:
        x_tap = TensorAccessPattern((1, x_units), 0, [1, 1, 1, x_units], [0, 0, 0, 1])
    if x_tap.sizes[-1] * x_tap.sizes[-2] != x_units:
        raise ValueError(f"{prologue} input does not fill the token buffer")

    # The residual: piece p of core row i lands at R[i * HIDDEN / 4 + p * M_TILE].
    r_tap = TensorAccessPattern(
        (1, HIDDEN),
        0,
        [1, RES_PIECES, ROWS, M_TILE],
        [0, M_TILE, HIDDEN // ROWS, 1],
    )

    def sequence(X, W, Y, R, x_prod, w_prods, y_conses, rtps):
        for rtp, row in zip(rtps, rtp_rows):
            rtp[0] = k_blocks
            rtp[1] = rounds
            rtp[2] = PROLOGUES[prologue]
            rtp[3] = EPILOGUES[epilogue]
            rtp[4] = RES_PIECES if prologue == "resnorm" else 0
            rtp[5] = row
        # One broadcast sends the token to every core; it has no completion
        # token, so its BD is freed at the end.
        x_task = x_prod.fill(X, x_tap, wait=False, managed=False)
        if prologue == "resnorm":
            # Queued first, so the residual column's first objects go to R.
            r_group = TaskGroup()
            y_conses[RES_COL].drain(R, r_tap, wait=True, group=r_group)
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
        if prologue == "resnorm":
            r_group.finish()
        x_task.free()

    rt = Runtime(
        sequence,
        [x_l3_ty, w_l3_ty, y_l3_ty, res_ty, x_prod, w_prods, y_conses, rtps],
    )
    return Program(dev, rt, workers=workers).resolve_program()


def compile_to(out_dir: Path, dev: str) -> None:
    device = from_name(dev, n_cols=None)
    iron.set_current_device(device)
    first_cores = None
    for variant in VARIANTS:
        name = sequence_name(*variant)
        # IRON reuses any kernel object already in the work dir, even when
        # its source changed, so every build starts from an empty one.
        work_dir = out_dir / f"{name}.work"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
        compile_mlir_module(
            gemv_program(device, *variant),
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
