"""Qwen3.6-35B-A3B's prefill GEMM shapes (prefill_gemm.py): pure data, no
IRON imports, so kernels/build.py --list-outputs can read OUTPUTS at
configure time.
"""

import prefill_dequant_variants as D  # noqa: E402 (sibling)

PREFILL_NAME = "qwen36_35b_a3b_prefill"
# Prompt rows per call (rows of A), each a multiple of 256.
PREFILL_CHUNKS = [256, 512, 1024]
# (K, N): every matrix the dequant converts (prefill_dequant_variants.py).
PREFILL_SHAPES = list(D.DEQUANT_SHAPES)


def prefill_sequence_name(m, k, n) -> str:
    return f"{PREFILL_NAME}_m{m}_k{k}_n{n}"


def prefill_sequences():
    for m in PREFILL_CHUNKS:
        for k, n in PREFILL_SHAPES:
            yield m, k, n


OUTPUTS = [f"{PREFILL_NAME}.xclbin"] + [
    f"{prefill_sequence_name(m, k, n)}.insts.elf" for m, k, n in prefill_sequences()
]
