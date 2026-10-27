"""Qwen3.5-4B's prefill GEMM, a copy of kernels/designs/bfp16_gemm.py (the
2B's) built with the 4B's shapes (prefill_gemm_variants.py) into its own xclbin
(qwen35_4b_prefill). The core program is the same; only the names and the
instruction sequences differ.

Prefill GEMM on the whole array: C (M x N, bf16) = A (M x K, bf16) x B, with
B (K x N) the weights as bfp16ebs8 (9 bytes per 8 values), packed on the host
in the order the cores read it (src/core/bfp16.hpp).

This is IRON's flm.GEMM design (iron.operators.flm.gemm.design and the
mlir-aie fused_mm kernel, Apache-2.0) at one fixed configuration: 4 rows x 8
columns of cores, a 64 x 64 output tile per core, k tiles of 512 held 128 at
a time, A broadcast along each core row from a memtile, B down each column
through a memtile slot pool, C joined in the memtile. Each core converts A to
bfp16 and runs the 8x8x8 bfp16 mmul with an fp32 accumulator; the output is
rounded to bf16 (round to nearest even).

M, K and N reach the cores as runtime parameters, so every shape shares one
xclbin (qwen35_4b_prefill.xclbin, in its own hardware context) and differs
only in its instruction sequence, one ELF per (M, K, N). M is the prompt
chunk (a multiple of 256 tokens); K and N are the 4B's projections.

Runtime arguments: A (bf16, M x K row-major), B (packed bfp16), C (bf16,
M x N row-major).
"""

import filecmp
from pathlib import Path
import shutil

import aie.iron as iron
from aie.iron.device import from_name
from aie.iron.kernels import fused_mm
from aie.utils.compile.utils import (
    compile_external_kernels,
    compile_mlir_module,
    resolve_target_arch,
)
from iron.operators.flm.gemm import design as flm

import prefill_gemm_variants as V  # noqa: E402 (sibling; build.py puts it on the path)

XCLBIN = V.PREFILL_NAME
TILE_N = 64
K_TILE = flm.K_TILE
CT_K = flm.CT_MAX_K_FOR_N[TILE_N]
M_CHUNK = 1
# bfp16ebs8: 9 bytes per 8 values.
B_BYTES_PER_VALUE = flm.BFP16_GROUP_BYTES / flm.BFP16_GROUP

CHUNKS = V.PREFILL_CHUNKS
SHAPES = V.PREFILL_SHAPES
sequence_name = V.prefill_sequence_name
_sequences = V.prefill_sequences
OUTPUTS = V.OUTPUTS


def _kernel(tile_ma):
    return fused_mm(
        dim_m=flm.M_TILE,
        dim_k=K_TILE,
        dim_n=TILE_N,
        band_m=tile_ma,
        chunk_k=CT_K,
        out_chunk=flm.CT_OUT_LEN,
        c_depth=flm.C_DEPTH,
        mmul_shape=(flm.R, flm.S, flm.T),
        bfp16_b=True,
        epilogue_modes=("none",),
        rounding="conv_even",
        gelu="fp32",
    )


def _program(device, m, k, n, tile_ma, kernel):
    return flm.gemm(
        device,
        m,
        k,
        n,
        epilogue=flm.Epilogue.NONE,
        clamp=None,
        tile_n=TILE_N,
        k_tile=K_TILE,
        m_chunk=M_CHUNK,
        tile_ma=tile_ma,
        kernel=kernel,
        trace_size=0,
    )


def _core_files(work_dir: Path):
    return sorted(
        p.relative_to(work_dir)
        for pattern in ("elfs_main_core_*/*.elf", "cdo_main/*.bin")
        for p in work_dir.glob(pattern)
    )


def compile_to(out_dir: Path, dev: str) -> None:
    device = from_name(dev, n_cols=None)
    iron.set_current_device(device)
    tile_ma, _ = flm._default_l1(
        TILE_N, CT_K, B_BYTES_PER_VALUE, flm.l1_budget(device), M_CHUNK
    )
    kernel = _kernel(tile_ma)
    first = None
    for m, k, n in _sequences():
        name = sequence_name(m, k, n)
        # IRON reuses any kernel object already in the work dir, even when
        # its source changed, so every build starts from an empty one.
        work_dir = out_dir / f"{name}.work"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
        # The cores call the kernel's entry points under bound names, which
        # compile_mlir_module's automatic kernel build does not match to the
        # kernel's own source, so the object is built here.
        compile_external_kernels([kernel], str(work_dir), resolve_target_arch(device))
        compile_mlir_module(
            _program(device, m, k, n, tile_ma, kernel),
            device=device,
            work_dir=work_dir,
            xclbin_path=work_dir / f"{name}.xclbin",
            insts_path=out_dir / f"{name}.insts.bin",
            elf_path=out_dir / f"{name}.insts.elf",
            # A long leg's pieces outnumber a shim's BD ids; the compiler
            # recycles finished tasks' ids (as flm.GEMM builds it).
            options=["--reclaim-runtime-bds"],
        )
        # Every shape must build the same array configuration, or its
        # sequence would not match the shared xclbin.
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
                raise RuntimeError(f"{name}: core program differs between shapes")

