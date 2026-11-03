"""Qwen3.5-2B's decode GEMV shapes (decode_gemv.py): pure data, no IRON
imports, so kernels/build.py --list-outputs can read OUTPUTS at configure
time. The host mirrors the geometry in src/models/qwen35_2b/npu/gemv.hpp.
"""

NAME = "qwen35_2b_decode"
HIDDEN = 2048  # residual width (resnorm)
FFN = 6144
# The token buffer's per-token activation width, in values. The 2B's longest
# input is FFN down's 6,144; this keeps the 8,704 of the program it was
# copied from (sized for the 35B's MoE gathers), so the fork moves no buffer,
# stride or core program size.
MAX_DIM = 8704
TOKENS = 2  # tokens a call can carry (two verify an MTP draft)
SLOT = 2 * HIDDEN  # fp32 per plain/resnorm token slot ([H | D])

# Decode calls as (dim, padded rows, input, output, Y token stride in fp32
# words). Each is built for 1 and 2 tokens.
VARIANTS = [
    # Mixer output: ssm_out and attn_output (2048 -> 2048) into the next
    # resnorm's D slots.
    (HIDDEN, HIDDEN, "plain", "store", SLOT),
    # The MTP head's eh_proj (4096 -> 2048) into the residual H slots.
    (2 * HIDDEN, HIDDEN, "plain", "store", SLOT),
    # Fused attention input: q + gate (4096), k (512), v (512).
    (HIDDEN, 5120, "resnorm", "store", 5120),
    # Fused DeltaNet input: qkv (6144) + z (2048).
    (HIDDEN, 8192, "resnorm", "store", 8192),
    # Fused gate + up with SwiGLU (2 x 6144).
    (HIDDEN, 2 * FFN, "resnorm", "swiglu", MAX_DIM // 2),
    # FFN down (6144 -> 2048) into the next D slots.
    (FFN, HIDDEN, "act", "store", SLOT),
    # The LM head (tied embeddings): 248,320 rows padded to 248,832.
    (HIDDEN, 248832, "resnorm", "store", 248832),
    # The MTP draft's LM head over the draft vocabulary.
    (HIDDEN, 32768, "resnorm", "store", 32768),
]


def sequence_name(dim, padded_rows, prologue, epilogue, tokens) -> str:
    """Matches npu::Q4nxGemvSequence with NAME as the design prefix."""
    return f"{NAME}_k{dim}_n{padded_rows}_{prologue}_{epilogue}_t{tokens}"


def sequences():
    for dim, rows, prologue, epilogue, y_stride in VARIANTS:
        for tokens in range(1, TOKENS + 1):
            yield (dim, rows, prologue, epilogue, y_stride, tokens)


# The xclbin, then one instruction ELF per variant (the ELF flow only).
OUTPUTS = [f"{NAME}.xclbin"] + [
    f"{sequence_name(d, n, p, e, t)}.insts.elf" for d, n, p, e, _, t in sequences()
]
