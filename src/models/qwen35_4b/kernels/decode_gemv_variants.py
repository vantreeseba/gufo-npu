"""Qwen3.5-4B's decode GEMV shapes (decode_gemv.py): pure data, no IRON
imports, so kernels/build.py --list-outputs can read OUTPUTS at configure
time. The host mirrors the geometry in src/models/qwen35_4b/npu/gemv.hpp.
"""

NAME = "qwen35_4b_decode"
HIDDEN = 2560  # residual width (resnorm)
FFN = 9216
MAX_DIM = FFN  # the longest input: FFN down's
TOKENS = 2  # tokens a call can carry (two verify an MTP draft)
SLOT = 2 * HIDDEN  # fp32 per plain/resnorm token slot ([H | D])
OUT_PER_ROUND = 8 * 4 * 32  # columns x rows x rows per core
LM_ROWS = 248320  # vocabulary
LM_PADDED = -(-LM_ROWS // OUT_PER_ROUND) * OUT_PER_ROUND
HIDDEN_PADDED = -(-HIDDEN // OUT_PER_ROUND) * OUT_PER_ROUND  # q4nx padding

# (dim, padded rows, rows, input, output, Y token stride in fp32 words).
# Padded rows are the q4nx matrix's (a multiple of OUT_PER_ROUND); only
# `rows` are computed and stored: a hidden-sized output (2560 rows, 2.5
# rounds) runs a third round on the first four columns only.
VARIANTS = [
    # Mixer output: ssm_out and attn_output (4096 -> 2560) into the next
    # resnorm's D slots.
    (4096, HIDDEN_PADDED, HIDDEN, "plain", "store", SLOT),
    # The MTP head's eh_proj (5120 -> 2560) into the residual H slots.
    (2 * HIDDEN, HIDDEN_PADDED, HIDDEN, "plain", "store", SLOT),
    # Fused attention input: q + gate (8192), k (1024), v (1024).
    (HIDDEN, 10240, 10240, "resnorm", "store", 10240),
    # Fused DeltaNet input: qkv (8192) + z (4096).
    (HIDDEN, 12288, 12288, "resnorm", "store", 12288),
    # Fused gate + up with SwiGLU (2 x 9216).
    (HIDDEN, 2 * FFN, 2 * FFN, "resnorm", "swiglu", 2 * FFN // 4),
    # FFN down (9216 -> 2560) into the next D slots.
    (FFN, HIDDEN_PADDED, HIDDEN, "act", "store", SLOT),
    # The LM head (tied embeddings), every padded row.
    (HIDDEN, LM_PADDED, LM_PADDED, "resnorm", "store", LM_PADDED),
    # The MTP draft's LM head over the draft vocabulary.
    (HIDDEN, 32768, 32768, "resnorm", "store", 32768),
]


def sequence_name(dim, padded_rows, prologue, epilogue, tokens) -> str:
    """Matches npu::Q4nxGemvSequence with NAME as the design prefix."""
    return f"{NAME}_k{dim}_n{padded_rows}_{prologue}_{epilogue}_t{tokens}"


def sequences():
    for dim, padded, rows, prologue, epilogue, y_stride in VARIANTS:
        for tokens in range(1, TOKENS + 1):
            yield (dim, padded, rows, prologue, epilogue, y_stride, tokens)


OUTPUTS = [f"{NAME}.xclbin"] + [
    f"{sequence_name(d, n, p, e, t)}.insts.elf" for d, n, _, p, e, _, t in sequences()
]
