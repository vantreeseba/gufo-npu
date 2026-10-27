"""q4nx GEMV for decode on the whole array: y (n_out) = W (n_out x dim) x, for
one or two tokens per weight read (two verify a speculative draft).

Every core (8 columns x 4 rows) owns M_TILE out-features per round and
accumulates one q4nx block per K_TILE in-features. Each column's weights come
in through its own shim DMA as one contiguous slice per round, so W must be
packed in this design's order (src/core/q4nx.hpp):

    out-feature n = ((round * COLS + col) * ROWS + row) * M_TILE + m
    block index   = ((round * COLS + col) * K_BLOCKS + k) * ROWS + row

One core program serves every call: each core reads its k block count, round
count, token count and input/output stages from a runtime parameter that the
variant's instruction sequence writes before it streams. The build emits one
xclbin (q4nx_gemv.xclbin) and one instruction ELF per variant, all for the
same hardware context, so calls chain in one runlist.

The token buffer X is always X_FLOATS fp32: TOKENS slots of SLOT fp32, then a
dim bf16 norm weight; the tail past the input is ignored. Input stages:
  plain    slot t: dim fp32.
  act      dim bf16 per token, back to back from the start of X (a compact
           SwiGLU output).
  resnorm  slot t: [H | D] (dim fp32 each), w after the slots. x =
           rms_norm(H + D) * w, and H + D goes to R (token t at t * SLOT), the
           next residual: the four cores of column RES_COL each send a quarter
           of it down their output stream before their rows. dim must be
           HIDDEN.
Output stages (token t at t * y_stride in Y):
  store    the rows' fp32 values.
  swiglu   each core's 32 rows are 16 gate rows then their 16 up rows; Y gets
           silu(gate) * up as bf16, compact (y_stride in fp32 words).

Adapted from IRON's iron.operators.flm.lm_head design (Apache-2.0).
Runtime arguments: X, W (q4nx bytes as uint32), Y (fp32), R (fp32).
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
# 4-bit codes, a uint8 sub-scale and int8 sub-minimum per row and 32 inputs,
# and a bf16 super-scale and super-minimum per row: 4.625 bits per weight.
BLOCK_BYTES = M_TILE * K_TILE // 2 + M_TILE * K_TILE // GROUP * 2 + M_TILE * 4
assert BLOCK_BYTES == 4736
COLS, ROWS = 8, 4
OUT_PER_ROUND = COLS * ROWS * M_TILE
MAX_DIM = 6144
HIDDEN = 2048  # residual width (resnorm)
TOKENS = 2  # tokens a call can carry
SLOT = 2 * HIDDEN  # fp32 per plain/resnorm token slot ([H | D])
X_FLOATS = TOKENS * SLOT + HIDDEN // 2  # slots, then the bf16 norm weight
RES_COL = 1  # the column whose cores hand out the residual
RES_PIECES = HIDDEN // ROWS // M_TILE  # output objects per core for it
RMS_EPS = 1e-6
# With per-round drains (SwiGLU), the sequence finishes each round's group
# DEPTH rounds later, which frees its BDs for reuse (a shim tile holds 16).
DEPTH = 2
STACK_SIZE = 8 * 1024

KERNEL_SOURCE = Path(__file__).resolve().parent.parent / "aie" / "q4nx_gemv.cc"
XCLBIN = "q4nx_gemv"

bf16 = np.dtype[bfloat16]

PROLOGUES = {"plain": 0, "resnorm": 1, "act": 2}
EPILOGUES = {"store": 0, "swiglu": 1}

# Decode calls of Qwen3.5-2B as (dim, padded rows, input, output, Y token
# stride): mixer output (into the next resnorm's D slots), the MTP head's
# eh_proj (into the residual H slots), fused attention
# input (q + gate, k, v), fused DeltaNet input (qkv + z), fused gate + up with
# SwiGLU, down (into the next D slots), and the LM head (248320 rows padded to
# 248832). Each is built for 1 and 2 tokens.
VARIANTS = [
    (2048, 2048, "plain", "store", SLOT),
    (4096, 2048, "plain", "store", SLOT),
    (2048, 5120, "resnorm", "store", 5120),
    (2048, 8192, "resnorm", "store", 8192),
    (2048, 12288, "resnorm", "swiglu", 12288 // 4),
    (6144, 2048, "act", "store", SLOT),
    (2048, 248832, "resnorm", "store", 248832),
    # The MTP draft's LM head over the draft vocabulary.
    (2048, 32768, "resnorm", "store", 32768),
]


def sequence_name(dim, rows, prologue, epilogue, tokens) -> str:
    return f"{XCLBIN}_k{dim}_n{rows}_{prologue}_{epilogue}_t{tokens}"


def _sequences():
    for dim, rows, prologue, epilogue, y_stride in VARIANTS:
        for tokens in range(1, TOKENS + 1):
            yield (dim, rows, prologue, epilogue, y_stride, tokens)


# build.py checks these: the xclbin, then one instruction ELF per variant.
OUTPUTS = [f"{XCLBIN}.xclbin"] + [
    f"{sequence_name(d, n, p, e, t)}.insts.elf" for d, n, p, e, _, t in _sequences()
]


def _kernels():
    x_ty = np.ndarray[(X_FLOATS,), np.dtype[np.float32]]
    sums_ty = np.ndarray[(TOKENS * 2 * MAX_DIM // GROUP,), bf16]  # hi, lo per token
    res_ty = np.ndarray[(SLOT + HIDDEN,), np.dtype[np.float32]]
    y_acc_ty = np.ndarray[(TOKENS * M_TILE,), np.dtype[np.float32]]
    y_ty = np.ndarray[(TOKENS * M_TILE,), np.dtype[np.float32]]
    w_blk_ty = np.ndarray[(BLOCK_BYTES // 2,), bf16]
    block = ExternalFunction(
        "gemv_block",
        source_file=str(KERNEL_SOURCE),
        arg_types=[w_blk_ty, x_ty, y_acc_ty, sums_ty, np.int32, np.int32, np.int32],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[
            f"-DQ4NX_M_TILE={M_TILE}",
            f"-DQ4NX_K_TILE={K_TILE}",
            f"-DQ4NX_GROUP={GROUP}",
            f"-DGEMV_MAX_DIM={MAX_DIM}",
            f"-DGEMV_RMS_EPS={RMS_EPS}f",
            f"-DGEMV_HIDDEN={HIDDEN}",
            f"-DGEMV_TOKENS={TOKENS}",
            f"-DGEMV_SLOT={SLOT}",
            # Without it the bf16 mmul emulation runs about 8x slower.
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
    )
    bind = block.object_file.bind
    prep = bind("gemv_prep", [x_ty, sums_ty, np.int32, np.int32, np.int32])
    residual = bind("gemv_residual", [x_ty, np.int32, np.int32, np.int32])
    residual_out = bind("gemv_residual_out", [y_ty, x_ty, np.int32, np.int32])
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


def gemv_program(dev, dim, n_out, prologue, epilogue, y_stride, tokens):
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
    # k blocks, rounds, input stage, output stage, residual pieces, core row,
    # tokens.
    rtp_ty = np.ndarray[(7,), np.dtype[np.int32]]

    w_col_ty = np.ndarray[(ROWS, w_blk), bf16]
    y_col_ty = np.ndarray[(ROWS, TOKENS * M_TILE), np.dtype[np.float32]]
    shims = [Tile(j, 0) for j in range(COLS)]

    def body(x, w_in, y_out, k_zero, k_block, k_store, y_acc, sums, rtp):
        for _ in range_(rtp[1]):
            k_zero(y_acc)
            for kb in range_(rtp[0]):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, sums, kb, rtp[2], rtp[6])
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
        k_residual(x, rtp[0], rtp[2], rtp[6])
        k_prep(x, sums, rtp[0], rtp[2], rtp[6])
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
        k_residual(x, rtp[0], rtp[2], rtp[6])
        for piece in range_(rtp[4]):
            y = y_out.acquire(1)
            k_residual_out(y, x, rtp[5], piece)
            y_out.release(1)
        k_prep(x, sums, rtp[0], rtp[2], rtp[6])
        body(x, w_in, y_out, k_zero, k_block, k_store, y_acc, sums, rtp)
        x_in.release(1)

    w_words = n_out // M_TILE * (dim // K_TILE) * BLOCK_BYTES // 4
    words_per_col = w_words // (rounds * COLS)
    w_l3_ty = np.ndarray[(w_words,), np.dtype[np.uint32]]
    if epilogue == "swiglu":
        # Token t's compact bf16 values at t * y_stride words; the padding of
        # each core's output (24 of every 32 words) goes past both tokens.
        pad_stride = TOKENS * y_stride
        y_l3_words = pad_stride * 4
    else:
        y_l3_words = (TOKENS - 1) * y_stride + n_out
    y_l3_ty = np.ndarray[(y_l3_words,), np.dtype[np.float32]]

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
            [TOKENS * M_TILE * i for i in range(ROWS)],
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

    # One weight task per column covers every round: column c's slice of
    # round r starts at (r * COLS + c) * words_per_col. The slice is split
    # into CHUNK-word runs to fit the BD's inner size field. A short
    # instruction stream also keeps the LM head's (243 rounds) small.
    chunk = 1024
    assert words_per_col % chunk == 0
    w_taps = [
        TensorAccessPattern(
            (1, rounds * COLS * words_per_col),
            col * words_per_col,
            [1, rounds, words_per_col // chunk, chunk],
            [0, COLS * words_per_col, chunk, 1],
        )
        for col in range(COLS)
    ]

    def y_tap(i):
        """Column object i (round * COLS + col): ROWS cores x TOKENS x M_TILE."""
        tile0 = i * ROWS
        # SwiGLU: per core and token, 8 words of bf16 results, 24 of padding.
        return TensorAccessPattern(
            (1, y_l3_words),
            tile0 * 8,
            [ROWS, TOKENS, 4, 8],
            [8, y_stride, pad_stride, 1],
        )

    def y_col_tap(col):
        """Every round of column `col` (store): ROWS cores x TOKENS x M_TILE."""
        return TensorAccessPattern(
            (1, y_l3_words),
            col * ROWS * M_TILE,
            [rounds, ROWS, TOKENS, M_TILE],
            [COLS * ROWS * M_TILE, M_TILE, y_stride, 1],
        )

    if epilogue == "swiglu":
        y_taps = [y_tap(i) for i in range(rounds * COLS)]
    else:
        y_taps = [y_col_tap(col) for col in range(COLS)]
    x_tap = TensorAccessPattern((1, X_FLOATS), 0, [1, 1, 1, X_FLOATS], [0, 0, 0, 1])
    # The residual: piece p of core row i, token t, lands at
    # R[t * SLOT + i * HIDDEN / 4 + p * M_TILE].
    r_tap = TensorAccessPattern(
        (1, SLOT + HIDDEN),
        0,
        [RES_PIECES, ROWS, TOKENS, M_TILE],
        [M_TILE, HIDDEN // ROWS, SLOT, 1],
    )

    def sequence(X, W, Y, R, x_prod, w_prods, y_conses, rtps):
        for rtp, row in zip(rtps, rtp_rows):
            rtp[0] = k_blocks
            rtp[1] = rounds
            rtp[2] = PROLOGUES[prologue]
            rtp[3] = EPILOGUES[epilogue]
            rtp[4] = RES_PIECES if prologue == "resnorm" else 0
            rtp[5] = row
            rtp[6] = tokens
        # One broadcast sends the token to every core; it has no completion
        # token, so its BD is freed at the end.
        x_task = x_prod.fill(X, x_tap, wait=False, managed=False)
        if prologue == "resnorm":
            # Queued first, so the residual column's first objects go to R.
            r_group = TaskGroup()
            y_conses[RES_COL].drain(R, r_tap, wait=True, group=r_group)
        tg = TaskGroup()
        for col in range(COLS):
            w_prods[col].fill(W, w_taps[col], wait=False, group=tg)
        if epilogue == "swiglu":
            # Five dimensions per column: one drain per round and column.
            window = []
            for rnd in range(rounds):
                dg = TaskGroup()
                for col in range(COLS):
                    y_conses[col].drain(Y, y_taps[rnd * COLS + col], wait=True, group=dg)
                window.append(dg)
                if len(window) > DEPTH:
                    window.pop(0).finish()
            for dg in window:
                dg.finish()
        else:
            for col in range(COLS):
                y_conses[col].drain(Y, y_taps[col], wait=True, group=tg)
        tg.finish()
        if prologue == "resnorm":
            r_group.finish()
        x_task.free()

    rt = Runtime(
        sequence,
        [x_ty, w_l3_ty, y_l3_ty, res_ty, x_prod, w_prods, y_conses, rtps],
    )
    return Program(dev, rt, workers=workers).resolve_program()


def compile_to(out_dir: Path, dev: str) -> None:
    device = from_name(dev, n_cols=None)
    iron.set_current_device(device)
    first_cores = None
    for variant in _sequences():
        dim, rows, prologue, epilogue, _, tokens = variant
        name = sequence_name(dim, rows, prologue, epilogue, tokens)
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
