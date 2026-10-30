"""Qwen3-VL-Embedding-2B's GEMM shapes (embed_gemm.py): pure data, no IRON
imports, so kernels/build.py --list-outputs can read OUTPUTS at configure
time. Must match src/models/qwen3vl_embed_2b/npu/gemm.hpp.
"""

HIDDEN = 2048
FFN = 6144
# q (16 heads) | k (8) | v (8), head size 128.
IN_ROWS = 4096

GEMM_NAME = "qwen3vl_embed_2b_gemm"
# Tokens per call (rows of A), as npu::kGemmChunks.
GEMM_CHUNKS = [256, 512, 1024]
# The projections as (K, N): attention input (q, k, v), attention output,
# gate + up, down.
GEMM_SHAPES = [
    (HIDDEN, IN_ROWS),
    (HIDDEN, HIDDEN),
    (HIDDEN, 2 * FFN),
    (FFN, HIDDEN),
]


def gemm_sequence_name(m, k, n) -> str:
    """Matches npu::Bfp16GemmSequence with GEMM_NAME as the prefix."""
    return f"{GEMM_NAME}_m{m}_k{k}_n{n}"


def gemm_sequences():
    for m in GEMM_CHUNKS:
        for k, n in GEMM_SHAPES:
            yield m, k, n


OUTPUTS = [f"{GEMM_NAME}.xclbin"] + [
    f"{gemm_sequence_name(m, k, n)}.insts.elf" for m, k, n in gemm_sequences()
]
