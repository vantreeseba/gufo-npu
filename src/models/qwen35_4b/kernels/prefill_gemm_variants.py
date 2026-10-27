"""Qwen3.5-4B's prefill GEMM shapes (prefill_gemm.py): pure data, no IRON
imports, so kernels/build.py --list-outputs can read OUTPUTS at configure
time.
"""

HIDDEN = 2560
FFN = 9216

# Prefill GEMM (prefill_gemm.py): prompt tokens per call, and the
# projections as (K, N): DeltaNet input (qkv + z), attention input (q + gate,
# k, v), mixer output (ssm_out / attn_output), gate + up, down, and the MTP
# head's eh_proj.
PREFILL_NAME = "qwen35_4b_prefill"
PREFILL_CHUNKS = [256, 512, 1024]
PREFILL_SHAPES = [
    (HIDDEN, 12288),
    (HIDDEN, 10240),
    (4096, HIDDEN),
    (HIDDEN, 2 * FFN),
    (FFN, HIDDEN),
    (2 * HIDDEN, HIDDEN),
]


def prefill_sequence_name(m, k, n) -> str:
    """Matches npu::Bfp16GemmSequence with PREFILL_NAME as the prefix."""
    return f"{PREFILL_NAME}_m{m}_k{k}_n{n}"


def prefill_sequences():
    for m in PREFILL_CHUNKS:
        for k, n in PREFILL_SHAPES:
            yield m, k, n


OUTPUTS = [f"{PREFILL_NAME}.xclbin"] + [
    f"{prefill_sequence_name(m, k, n)}.insts.elf" for m, k, n in prefill_sequences()
]
