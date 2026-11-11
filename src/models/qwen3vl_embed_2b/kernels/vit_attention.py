"""Qwen3-VL-Embedding-2B's ViT attention on the whole array: bidirectional
softmax(Q K^T / 8) V for the 16 heads (dimension 64) of one image of n
patches, n padded to a multiple of 256 (vit_attention_variants.py).

Each core (8 columns x 4 rows) runs flash attention for a 64-row query block
of one head over all its keys (aie/vit_attention.cc: fp32 softmax state and
output, 3-term bfp16 splits on the mmul). Column c computes heads 2c and
2c + 1: for each head and group g of 4 query blocks, core row r takes block
4g + r. Per column:
  - Q comes through one shim channel, 4 query blocks per memtile object,
    split to the 4 cores (one object per core and unit).
  - K and V come through the other, one object per key block, which the
    memtile broadcasts to the 4 cores; the head's key blocks are sent once per
    group (a repeat in the shim's access pattern).
  - The cores' outputs are joined in the memtile, a quarter of each core's
    block per object, and drained through one shim channel.

Each core reads its unit count (minus one) and key block count from a runtime
parameter after its first query block arrives, so one core program serves
every length. The build emits one xclbin (qwen3vl_embed_2b_vit_attn.xclbin)
and one instruction ELF per length, all for the same hardware context.

Runtime arguments (uint32 words), with G = n / 256 groups and NB = n / 32 key
blocks:
  Q   [head][group][row] query blocks of Q_BYTES (split tiles, see the core
      source).
  KV  [head][key block] objects of KV_BYTES (split k tiles, biases, split v^T
      tiles).
  O   [head][group][quarter][row] objects of O_BYTES (bf16 tiles).
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

import vit_attention_variants as V  # noqa: E402 (sibling; build.py puts it on the path)

HERE = Path(__file__).resolve().parent
KERNEL_SOURCE = HERE / "aie" / "vit_attention.cc"
XCLBIN = V.NAME
OUTPUTS = V.OUTPUTS

COLS, ROWS = 8, 4
HEADS_PER_COL = V.HEADS // COLS
BQ, BK, D = 64, 32, V.HEAD_DIM
QT, KT, DT = BQ // 8, BK // 8, D // 8
BV = 72  # one 8x8 tile as bfp16ebs8 (8 blocks of 9 bytes)
Q_BYTES = QT * DT * 2 * BV
K_BYTES = DT * KT * 2 * BV + BK * 4
V_BYTES = DT * KT * 2 * BV
KV_BYTES = K_BYTES + V_BYTES
O_BYTES = 2 * DT * 64 * 2  # two query tiles, bf16
# m, d (int32 per row), the moved flags, l (row pattern per query tile), the
# bias pattern (per key tile) and S - m / P (fp32 [qt][kt] tiles).
SCRATCH = BQ * 4 * 2 + 64 + QT * 64 * 4 + KT * 64 * 4 + QT * KT * 64 * 4
# The kernel's frames take about 2.5 KB.
STACK_SIZE = 3584
assert V.BUCKET == ROWS * BQ and V.BUCKET % BK == 0

u8 = np.dtype[np.uint8]
bf16 = np.dtype[bfloat16]


def _words(nbytes):
    assert nbytes % 4 == 0
    return nbytes // 4


def attention_program(dev, n):
    if n % V.BUCKET or not 0 < n:
        raise ValueError(f"unsupported length {n}")
    groups = n // V.BUCKET
    blocks = n // BK
    units = HEADS_PER_COL * groups  # per core

    q_ty = np.ndarray[(Q_BYTES,), u8]
    q_col_ty = np.ndarray[(ROWS * Q_BYTES,), u8]
    kv_ty = np.ndarray[(KV_BYTES,), u8]
    o_ty = np.ndarray[(O_BYTES // 2,), bf16]
    o_col_ty = np.ndarray[(ROWS * O_BYTES // 2,), bf16]
    acc_ty = np.ndarray[(BQ * D,), np.dtype[np.float32]]
    scratch_ty = np.ndarray[(SCRATCH,), u8]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]  # units - 1, key blocks

    block = ExternalFunction(
        "attn_block",
        source_file=str(KERNEL_SOURCE),
        arg_types=[q_ty, kv_ty, acc_ty, scratch_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[
            f"-DVIT_ATTN_SCRATCH={SCRATCH}",
            f"-DVIT_ATTN_Q_BYTES={Q_BYTES}",
            f"-DVIT_ATTN_KV_BYTES={KV_BYTES}",
        ],
    )
    bind = block.object_file.bind
    begin = bind("attn_begin", [acc_ty, scratch_ty])
    out = bind("attn_out", [o_ty, acc_ty, scratch_ty, np.int32])

    def core_fn(q_in, kv_in, o_out, k_begin, k_block, k_out, acc, scratch, rtp):
        def unit():
            q = q_in.acquire(1)
            k_begin(acc, scratch)
            for _ in range_(rtp[1]):
                kv = kv_in.acquire(1)
                k_block(q, kv, acc, scratch)
                kv_in.release(1)
            for c in range(ROWS):
                o = o_out.acquire(1)
                k_out(o, acc, scratch, c)
                o_out.release(1)
            q_in.release(1)

        # The sequence writes the RTP before it sends Q, so it is current once
        # the first query block has arrived.
        unit()
        for _ in range_(rtp[0]):
            unit()

    shims = [Tile(c, 0) for c in range(COLS)]
    workers, q_prods, kv_prods, o_conses, rtps = [], [], [], [], []
    for col in range(COLS):
        memtile = Tile(col, 1)
        of_q = ObjectFifo(q_col_ty, name=f"q{col}", depth=2)
        q_prods.append(of_q.prod(tile=shims[col]))
        q_cores = of_q.cons().split(
            [Q_BYTES * r for r in range(ROWS)],
            tile=memtile,
            obj_types=[q_ty] * ROWS,
            depths=[1] * ROWS,
            names=[f"q{col}_{r}" for r in range(ROWS)],
        )
        of_kv = ObjectFifo(kv_ty, name=f"kv{col}", depth=2)
        kv_prods.append(of_kv.prod(tile=shims[col]))
        of_kv_l1 = of_kv.cons().forward(tile=memtile, obj_type=kv_ty, depth=2, name=f"kv{col}_l1")
        of_o = ObjectFifo(o_col_ty, name=f"o{col}", depth=2)
        o_conses.append(of_o.cons(tile=shims[col]))
        o_cores = of_o.prod().join(
            [O_BYTES // 2 * r for r in range(ROWS)],
            tile=memtile,
            obj_types=[o_ty] * ROWS,
            depths=[1] * ROWS,
            names=[f"o{col}_{r}" for r in range(ROWS)],
        )
        for row in range(ROWS):
            rtp = Buffer(type=rtp_ty, name=f"rtp_{col}_{row}", use_write_rtp=True)
            rtps.append(rtp)
            args = [
                q_cores[row].cons(),
                of_kv_l1.cons(),
                o_cores[row].prod(),
                begin,
                block,
                out,
                Buffer(type=acc_ty, name=f"acc_{col}_{row}"),
                Buffer(type=scratch_ty, name=f"scratch_{col}_{row}"),
                rtp,
            ]
            workers.append(
                Worker(core_fn, args, stack_size=STACK_SIZE, tile=Tile(col, 2 + row))
            )

    q_words = _words(V.HEADS * groups * ROWS * Q_BYTES)
    kv_head_words = _words(blocks * KV_BYTES)
    kv_words = V.HEADS * kv_head_words
    o_unit_words = _words(ROWS * ROWS * O_BYTES)
    o_words = V.HEADS * groups * o_unit_words
    q_l3_ty = np.ndarray[(q_words,), np.dtype[np.uint32]]
    kv_l3_ty = np.ndarray[(kv_words,), np.dtype[np.uint32]]
    o_l3_ty = np.ndarray[(o_words,), np.dtype[np.uint32]]

    def linear(total, offset, words):
        """`words` contiguous words at `offset`, in runs of at most 1024."""
        chunk = max(c for c in range(1, 1025) if words % c == 0)
        return TensorAccessPattern((1, total), offset, [1, 1, words // chunk, chunk], [0, 0, chunk, 1])

    col_q_words = q_words // COLS
    col_o_words = o_words // COLS
    q_taps = [linear(q_words, col * col_q_words, col_q_words) for col in range(COLS)]
    o_taps = [linear(o_words, col * col_o_words, col_o_words) for col in range(COLS)]
    # A head's key blocks, once per group: a zero-stride outer dimension.
    kv_chunk = max(c for c in range(1, 1025) if _words(KV_BYTES) % c == 0)
    kv_taps = [
        TensorAccessPattern(
            (1, kv_words),
            head * kv_head_words,
            [groups, 1, kv_head_words // kv_chunk, kv_chunk],
            [0, 0, kv_chunk, 1],
        )
        for head in range(V.HEADS)
    ]

    def sequence(Q, KV, O, q_prods, kv_prods, o_conses, rtps):
        for rtp in rtps:
            rtp[0] = units - 1
            rtp[1] = blocks
        tg = TaskGroup()
        for col in range(COLS):
            q_prods[col].fill(Q, q_taps[col], wait=False, group=tg)
            for h in range(HEADS_PER_COL):
                kv_prods[col].fill(KV, kv_taps[col * HEADS_PER_COL + h], wait=False, group=tg)
        for col in range(COLS):
            o_conses[col].drain(O, o_taps[col], wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [q_l3_ty, kv_l3_ty, o_l3_ty, q_prods, kv_prods, o_conses, rtps],
    )
    return Program(dev, rt, workers=workers).resolve_program()


def _core_files(work_dir: Path):
    return sorted(
        p.relative_to(work_dir)
        for pattern in ("elfs_main_core_*/*.elf", "cdo_main/*.bin")
        for p in work_dir.glob(pattern)
    )


def build(out_dir: Path, dev: str, lengths) -> None:
    device = from_name(dev, n_cols=None)
    iron.set_current_device(device)
    first = None
    for n in lengths:
        name = V.sequence_name(n)
        # IRON reuses any kernel object already in the work dir, even when
        # its source changed, so every build starts from an empty one.
        work_dir = out_dir / f"{name}.work"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
        compile_mlir_module(
            attention_program(device, n),
            device=device,
            work_dir=work_dir,
            xclbin_path=work_dir / f"{name}.xclbin",
            insts_path=out_dir / f"{name}.insts.bin",
            elf_path=out_dir / f"{name}.insts.elf",
        )
        # Every length must build the same array configuration (core
        # programs, routing, locks, tile DMAs), or its sequence would not
        # match the shared xclbin.
        cores = _core_files(work_dir)
        if not cores:
            raise RuntimeError(f"{name}: no core ELFs or CDOs in {work_dir}")
        if first is None:
            first = (work_dir, cores)
            shutil.copyfile(work_dir / f"{name}.xclbin", out_dir / f"{XCLBIN}.xclbin")
        else:
            first_dir, first_cores = first
            if cores != first_cores or not all(
                filecmp.cmp(first_dir / c, work_dir / c, shallow=False) for c in cores
            ):
                raise RuntimeError(f"{name}: core program differs between lengths")


def compile_to(out_dir: Path, dev: str) -> None:
    build(out_dir, dev, V.LENGTHS)

