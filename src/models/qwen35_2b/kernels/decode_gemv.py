"""Qwen3.5-2B's q4nx decode GEMV on the whole array: y (n_out) = W (n_out x
dim) x, for one or two tokens per weight read (two verify a speculative
draft).

Copied from the q4nx GEMV design the 2B shared with Qwen3.6-35B-A3B (now
src/models/qwen36_35b_a3b/kernels/decode_gemv.py) into the 2B's own xclbin, so changes here cannot reach
another model's NPU program. The 35B's shapes, MoE gathers and scaled SwiGLU
are left out; the geometry (decode_gemv_variants.py), tiling and core code
are the same.

Every core (8 columns x 4 rows) owns M_TILE out-features per round and
accumulates one q4nx block per K_TILE in-features. Each column's weights come
in through its own shim DMA as one contiguous slice per round, so W must be
packed in this design's order (src/core/q4nx.hpp):

    out-feature n = ((round * COLS + col) * ROWS + row) * M_TILE + m
    block index   = ((round * COLS + col) * K_BLOCKS + k) * ROWS + row

One core program serves every call: each core reads its k block count, round
count, token count and input/output stages from a runtime parameter that the
variant's instruction sequence writes before it streams. The build emits one
xclbin (qwen35_2b_decode.xclbin) and one instruction ELF per variant, all for
the same hardware context, so calls chain in one runlist. A chain sequence
(decode_gemv_variants.py CHAIN_NEXT) runs a layer's out, gate + up, down and
next call back to back in one instruction stream, so the decoder submits one
call per layer instead of four.

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

HERE = Path(__file__).resolve().parent

# Shapes, sequence names and outputs: pure data, also read by CMake.
import decode_gemv_variants as V  # noqa: E402 (sibling; build.py puts it on the path)

M_TILE, K_TILE, GROUP = 32, 256, 32
# 4-bit codes, a uint8 sub-scale and int8 sub-minimum per row and 32 inputs,
# and a bf16 super-scale and super-minimum per row: 4.625 bits per weight.
BLOCK_BYTES = M_TILE * K_TILE // 2 + M_TILE * K_TILE // GROUP * 2 + M_TILE * 4
assert BLOCK_BYTES == 4736
COLS, ROWS = 8, 4
OUT_PER_ROUND = COLS * ROWS * M_TILE
MAX_DIM = V.MAX_DIM
HIDDEN = V.HIDDEN  # residual width (resnorm)
TOKENS = V.TOKENS  # tokens a call can carry
SLOT = V.SLOT  # fp32 per plain/resnorm token slot ([H | D])
X_FLOATS = TOKENS * SLOT + HIDDEN // 2  # slots, then the bf16 norm weight
RES_COL = 1  # the column whose cores hand out the residual
RES_PIECES = HIDDEN // ROWS // M_TILE  # output objects per core for it
RMS_EPS = 1e-6
# With per-round drains (SwiGLU), the sequence finishes each round's group
# DEPTH rounds later, which frees its BDs for reuse (a shim tile holds 16).
DEPTH = 2
STACK_SIZE = 8 * 1024

KERNEL_SOURCE = HERE / "aie" / "decode_gemv.cc"
XCLBIN = V.NAME

bf16 = np.dtype[bfloat16]
# k blocks, rounds, input stage, output stage, residual pieces, core row,
# tokens.
RTP_TY = np.ndarray[(7,), np.dtype[np.int32]]

PROLOGUES = {"plain": 0, "resnorm": 1, "act": 2}
EPILOGUES = {"store": 0, "swiglu": 1}

VARIANTS = V.VARIANTS
sequence_name = V.sequence_name
sequences = V.sequences

# The xclbin, then one instruction ELF per variant.
OUTPUTS = V.OUTPUTS

def _chunk(words):
    """The longest run (at most 1024 words, the BD inner size) dividing words."""
    return max(c for c in range(1, 1025) if words % c == 0)


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


def _check_shape(dim, n_out, prologue):
    if dim % K_TILE or dim > MAX_DIM or n_out % OUT_PER_ROUND:
        raise ValueError(f"unsupported shape dim={dim} n_out={n_out}")
    if prologue == "resnorm" and dim != HIDDEN:
        raise ValueError("resnorm needs dim == HIDDEN")


def _matrix_words(dim, n_out):
    """A q4nx matrix's size in uint32 words."""
    return n_out // M_TILE * (dim // K_TILE) * BLOCK_BYTES // 4


class _Array:
    """The array every sequence runs on: the token broadcast, each column's
    weight and output object FIFOs and the 32 workers with their RTPs. None
    of it depends on the call's shape, which the RTPs carry."""

    def __init__(self):
        k, types = _kernels()
        x_ty, sums_ty, y_acc_ty, y_blk_ty, w_blk_ty, res_ty = types
        self.x_ty, self.res_ty = x_ty, res_ty
        (w_blk,) = w_blk_ty.__args__[0]
        rtp_ty = RTP_TY

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

        # The sequence writes the RTP before it sends the token, so it is
        # current once the token arrives.
        def core_fn(
            x_in, w_in, y_out, k_residual, k_prep, k_zero, k_block, k_store, y_acc, sums, rtp
        ):
            x = x_in.acquire(1)
            k_residual(x, rtp[0], rtp[2], rtp[6])
            k_prep(x, sums, rtp[0], rtp[2], rtp[6])
            body(x, w_in, y_out, k_zero, k_block, k_store, y_acc, sums, rtp)
            x_in.release(1)

        # The residual column's cores first send their quarter of H + D (RTP
        # word 4 pieces; 0 when the input stage is not resnorm).
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

        of_x = ObjectFifo(x_ty, name="x", depth=1)
        self.x_prod = of_x.prod(tile=shims[0])
        self.workers, self.w_prods, self.y_conses = [], [], []
        self.rtps, self.rtp_rows = [], []
        for j in range(COLS):
            of_w = ObjectFifo(w_col_ty, name=f"w{j}")
            self.w_prods.append(of_w.prod(tile=shims[j]))
            w_cores = of_w.cons().split(
                [w_blk * i for i in range(ROWS)],
                obj_types=[w_blk_ty] * ROWS,
                names=[f"w{j}_{i}" for i in range(ROWS)],
            )
            of_y = ObjectFifo(y_col_ty, name=f"y{j}")
            self.y_conses.append(of_y.cons(tile=shims[j]))
            y_cores = of_y.prod().join(
                [TOKENS * M_TILE * i for i in range(ROWS)],
                obj_types=[y_blk_ty] * ROWS,
                names=[f"y{j}_{i}" for i in range(ROWS)],
            )
            for i in range(ROWS):
                rtp = Buffer(type=rtp_ty, name=f"rtp_{i}_{j}", use_write_rtp=True)
                self.rtps.append(rtp)
                self.rtp_rows.append(i)
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
                self.workers.append(Worker(fn, args + kernels + scratch, stack_size=STACK_SIZE))

    def program(self, dev, sequence, arg_types):
        """The design with `sequence` (taking the arguments of `arg_types`
        and then this array's x_prod, w_prods, y_conses and rtps)."""
        rt = Runtime(
            sequence,
            arg_types + [self.x_prod, self.w_prods, self.y_conses, self.rtps],
        )
        return Program(dev, rt, workers=self.workers).resolve_program()


class _Region:
    """A runtime argument and where a call's operand starts in it: `words`
    is the argument's size and `offset` the operand's, both in 4-byte
    words."""

    def __init__(self, tensor, words, offset=0):
        self.tensor, self.words, self.offset = tensor, words, offset


def _emit_call(
    shape, tokens, X, W, Y, R, x_prod, w_prods, y_conses, rtps, rtp_rows, written
):
    """One GEMV call's part of a runtime sequence: X, W, Y and R are
    _Regions. `written` holds each core's RTP words as the sequence last
    wrote them (None before the first call); only the words that change are
    written, and `written` is updated.

    Every call ends with all its tasks awaited, so a call that follows in
    the same sequence starts on idle channels: its waits come in queue order
    per channel as a single call's do, and its RTP writes come after every
    core has sent its last output of this call, which is the last time a
    core reads its RTP before it takes the next token."""
    dim, n_out, prologue, epilogue, y_stride = shape
    _check_shape(dim, n_out, prologue)
    rounds = n_out // OUT_PER_ROUND
    k_blocks = dim // K_TILE
    w_words = _matrix_words(dim, n_out)
    words_per_col = w_words // (rounds * COLS)
    if epilogue == "swiglu":
        # Token t's compact bf16 values at t * y_stride words; the padding of
        # each core's output (24 of every 32 words) goes past both tokens.
        pad_stride = TOKENS * y_stride
    # One weight task per column covers every round: column c's slice of
    # round r starts at (r * COLS + c) * words_per_col. The slice is split
    # into CHUNK-word runs to fit the BD's inner size field. A short
    # instruction stream also keeps the LM head's (243 rounds) small.
    chunk = _chunk(words_per_col)
    w_taps = [
        TensorAccessPattern(
            (1, W.words),
            W.offset + col * words_per_col,
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
            (1, Y.words),
            Y.offset + tile0 * 8,
            [ROWS, TOKENS, 4, 8],
            [8, y_stride, pad_stride, 1],
        )

    def y_col_tap(col):
        """Every round of column `col` (store): ROWS cores x TOKENS x M_TILE."""
        return TensorAccessPattern(
            (1, Y.words),
            Y.offset + col * ROWS * M_TILE,
            [rounds, ROWS, TOKENS, M_TILE],
            [COLS * ROWS * M_TILE, M_TILE, y_stride, 1],
        )

    if epilogue == "swiglu":
        y_taps = [y_tap(i) for i in range(rounds * COLS)]
    else:
        y_taps = [y_col_tap(col) for col in range(COLS)]
    x_tap = TensorAccessPattern(
        (1, X.words), X.offset, [1, 1, 1, X_FLOATS], [0, 0, 0, 1]
    )
    # The residual: piece p of core row i, token t, lands at
    # R[t * SLOT + i * HIDDEN / 4 + p * M_TILE].
    r_tap = TensorAccessPattern(
        (1, R.words),
        R.offset,
        [RES_PIECES, ROWS, TOKENS, M_TILE],
        [M_TILE, HIDDEN // ROWS, SLOT, 1],
    )

    shape_words = (
        k_blocks,
        rounds,
        PROLOGUES[prologue],
        EPILOGUES[epilogue],
        RES_PIECES if prologue == "resnorm" else 0,
    )
    for c, (rtp, row) in enumerate(zip(rtps, rtp_rows)):
        words = shape_words + (row, tokens)
        for word, value in enumerate(words):
            if written[c] is None or written[c][word] != value:
                rtp[word] = value
        written[c] = words
    # One broadcast sends the token to every core; it has no completion
    # token, so its BD is freed at the end.
    x_task = x_prod.fill(X.tensor, x_tap, wait=False, managed=False)
    # A wait takes the next completion token of its channel, whichever
    # task sent it, so waits on a channel must come in the order its
    # tasks were queued. The residual drain is queued first on RES_COL's
    # channel, so it is awaited before any Y drain there: awaited later,
    # its token satisfied the wait for round 0's Y drain, whose BD the
    # SwiGLU window then reused while round 0 could still be in flight
    # (about 1 call in 30,000 put round 0's values into round 3 or 4).
    r_group = None
    if prologue == "resnorm":
        # Queued first, so the residual column's first objects go to R.
        r_group = TaskGroup()
        y_conses[RES_COL].drain(R.tensor, r_tap, wait=True, group=r_group)

    def finish_residual():
        nonlocal r_group
        if r_group is not None:
            r_group.finish()
            r_group = None

    tg = TaskGroup()
    for col in range(COLS):
        w_prods[col].fill(W.tensor, w_taps[col], wait=False, group=tg)
    if epilogue == "swiglu":
        # Five dimensions per column: one drain per round and column.
        window = []
        for rnd in range(rounds):
            dg = TaskGroup()
            for col in range(COLS):
                y_conses[col].drain(Y.tensor, y_taps[rnd * COLS + col], wait=True, group=dg)
            window.append(dg)
            if len(window) > DEPTH:
                finish_residual()
                window.pop(0).finish()
        finish_residual()
        for dg in window:
            dg.finish()
    else:
        for col in range(COLS):
            y_conses[col].drain(Y.tensor, y_taps[col], wait=True, group=tg)
        finish_residual()
    tg.finish()
    x_task.free()


def _y_words(shape):
    """A call's Y span in words (Q4nxGemv::y_bytes / 4)."""
    _, n_out, _, epilogue, y_stride = shape
    if epilogue == "swiglu":
        return TOKENS * y_stride * 4
    return (TOKENS - 1) * y_stride + n_out


def gemv_program(dev, dim, n_out, prologue, epilogue, y_stride, tokens):
    """The design with the instruction sequence for one variant.
    Runtime arguments: X, W (q4nx bytes as uint32), Y (fp32), R (fp32)."""
    shape = (dim, n_out, prologue, epilogue, y_stride)
    _check_shape(dim, n_out, prologue)
    arr = _Array()
    w_words = _matrix_words(dim, n_out)
    y_words = _y_words(shape)
    r_words = SLOT + HIDDEN
    w_l3_ty = np.ndarray[(w_words,), np.dtype[np.uint32]]
    y_l3_ty = np.ndarray[(y_words,), np.dtype[np.float32]]

    def sequence(X, W, Y, R, x_prod, w_prods, y_conses, rtps):
        _emit_call(
            shape,
            tokens,
            _Region(X, X_FLOATS),
            _Region(W, w_words),
            _Region(Y, y_words),
            _Region(R, r_words),
            x_prod,
            w_prods,
            y_conses,
            rtps,
            arr.rtp_rows,
            [None] * len(rtps),
        )

    return arr.program(dev, sequence, [arr.x_ty, w_l3_ty, y_l3_ty, arr.res_ty])


def chain_program(dev, next_name, tokens):
    """The design with one chain's instruction sequence (V.CHAIN_CALLS, then
    the call named by V.CHAIN_NEXT) on the activation window layout of
    decode_gemv_variants.py. Runtime arguments: A (the window, fp32), WL
    (the layer's out, gate + up and down matrices back to back, uint32), WN
    (the next call's matrix, uint32), S (scratch for the residual argument
    of the calls that write none, fp32)."""
    arr = _Array()
    out, ffn, down = V.CHAIN_CALLS
    a_words = V.chain_window_bytes(next_name) // 4
    calls_words = [_matrix_words(d, n) for d, n, *_ in V.CHAIN_CALLS]
    wl_words = sum(calls_words)
    next_rows = V.CHAIN_NEXT[next_name]
    next_shape = None
    wn_words = 1024  # unused by a chain without a next call
    if next_rows is not None:
        next_shape = (HIDDEN, next_rows, "resnorm", "store", next_rows)
        wn_words = _matrix_words(HIDDEN, next_rows)
    s_words = SLOT + HIDDEN

    def word(offset_bytes):
        assert offset_bytes % 4 == 0
        return offset_bytes // 4

    def sequence(A, WL, WN, S, x_prod, w_prods, y_conses, rtps):
        written = [None] * len(rtps)

        def a(offset_bytes):
            return _Region(A, a_words, word(offset_bytes))

        def call(shape, X, W, Y, R):
            _emit_call(
                shape, tokens, X, W, Y, R, x_prod, w_prods, y_conses, rtps,
                arr.rtp_rows, written,
            )

        scratch = _Region(S, s_words)
        # out: the mixer output -> the FFN input's D slots.
        call(
            out,
            a(V.WINDOW_OUT_X),
            _Region(WL, wl_words, 0),
            a(V.WINDOW_FFN_X + V.D_OFFSET),
            scratch,
        )
        # gate + up with SwiGLU; its residual H + D -> the next input's H.
        call(
            ffn,
            a(V.WINDOW_FFN_X),
            _Region(WL, wl_words, calls_words[0]),
            a(V.WINDOW_FFN_Y),
            a(V.WINDOW_NEXT),
        )
        # down -> the next input's D slots.
        call(
            down,
            a(V.WINDOW_FFN_Y),
            _Region(WL, wl_words, calls_words[0] + calls_words[1]),
            a(V.WINDOW_NEXT + V.D_OFFSET),
            scratch,
        )
        if next_shape is not None:
            call(
                next_shape,
                a(V.WINDOW_NEXT),
                _Region(WN, wn_words),
                a(V.WINDOW_NEXT_Y),
                a(V.WINDOW_NEXT_R),
            )

    arg_types = [
        np.ndarray[(a_words,), np.dtype[np.float32]],
        np.ndarray[(wl_words,), np.dtype[np.uint32]],
        np.ndarray[(wn_words,), np.dtype[np.uint32]],
        arr.res_ty,
    ]
    return arr.program(dev, sequence, arg_types)


def compile_to(out_dir: Path, dev: str) -> None:
    device = from_name(dev, n_cols=None)
    iron.set_current_device(device)
    first_cores = None
    builds = [
        (sequence_name(d, n, p, e, t), lambda v=(d, n, p, e, s, t): gemv_program(device, *v))
        for d, n, p, e, s, t in sequences()
    ] + [
        (V.chain_name(n, t), lambda n=n, t=t: chain_program(device, n, t))
        for n, t in V.chains()
    ]
    for name, build in builds:
        # IRON reuses any kernel object already in the work dir, even when
        # its source changed, so every build starts from an empty one.
        work_dir = out_dir / f"{name}.work"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
        compile_mlir_module(
            build(),
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
