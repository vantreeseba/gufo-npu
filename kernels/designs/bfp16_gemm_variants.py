"""Shape variants of the bfp16 GEMM design (bfp16_gemm.py): pure data, no
IRON imports, so the build can list the design's outputs at configure time
(build.py --list-outputs).
"""

XCLBIN = "bfp16_gemm"

# Prompt tokens per call (rows of A), each a multiple of 256.
CHUNKS = [256, 512, 1024]
# Qwen3.5-2B prefill projections as (K, N): DeltaNet input (qkv + z),
# attention input (q + gate, k, v), mixer output (ssm_out / attn_output),
# gate + up, down, and the MTP head's eh_proj.
SHAPES = [
    (2048, 8192),
    (2048, 5120),
    (2048, 2048),
    (2048, 12288),
    (6144, 2048),
    (4096, 2048),
]


def sequence_name(m, k, n) -> str:
    return f"{XCLBIN}_m{m}_k{k}_n{n}"


def sequences():
    for m in CHUNKS:
        for k, n in SHAPES:
            yield m, k, n


# build.py checks these: the xclbin, then one instruction ELF per shape.
OUTPUTS = [f"{XCLBIN}.xclbin"] + [
    f"{sequence_name(m, k, n)}.insts.elf" for m, k, n in sequences()
]
