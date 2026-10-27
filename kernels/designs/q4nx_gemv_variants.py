"""Shape variants of the q4nx GEMV design (q4nx_gemv.py): pure data, no
IRON imports, so the build can list the design's outputs at configure time
(build.py --list-outputs).
"""

XCLBIN = "q4nx_gemv"
MAX_DIM = 6144  # longest input a call takes
HIDDEN = 2048  # residual width (resnorm)
TOKENS = 2  # tokens a call can carry
SLOT = 2 * HIDDEN  # fp32 per plain/resnorm token slot ([H | D])
EXPERT_FFN = 512  # an MoE expert's width (Qwen3.6-35B-A3B)

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
    # Qwen3.6-35B-A3B MoE, one expert per call: gate + up and down.
    (2048, 1024, "resnorm", "swiglu", MAX_DIM // 2),
    (512, 2048, "act", "store", SLOT),
    # Qwen3.6-35B-A3B input projections: DeltaNet (qkv + z), attention
    # (q + gate, k, v). Its mixer outputs (4096 -> 2048) and LM head
    # (248,320 rows) use the shapes above.
    (2048, 12288, "resnorm", "store", 12288),
    (2048, 9216, "resnorm", "store", 9216),
]

# MoE gathers as (dim, rows, input, output, Y token stride, gather, experts):
# 9 experts (8 routed and the shared one) per call.
MOE_VARIANTS = [
    (2048, 9 * 1024, "resnorm", "swiglu_scaled", MAX_DIM // 2, "rounds", 9),
    (9 * EXPERT_FFN, 2048, "act", "store", SLOT, "k", 9),
]


def sequence_name(dim, rows, prologue, epilogue, tokens, gather="", experts=0):
    moe = f"_{gather}{experts}" if gather else ""
    return f"{XCLBIN}_k{dim}_n{rows}_{prologue}_{epilogue}{moe}_t{tokens}"


def sequences():
    for variant in VARIANTS + MOE_VARIANTS:
        dim, rows, prologue, epilogue, y_stride, *moe = variant
        gather, experts = moe if moe else ("", 0)
        for tokens in range(1, TOKENS + 1):
            yield (dim, rows, prologue, epilogue, y_stride, tokens, gather, experts)


# build.py checks these: the xclbin, then per variant an instruction ELF and
# the raw instruction words (the host's instruction-buffer flow).
OUTPUTS = [f"{XCLBIN}.xclbin"] + [
    f"{sequence_name(d, n, p, e, t, g, x)}.insts.{suffix}"
    for d, n, p, e, _, t, g, x in sequences()
    for suffix in ("elf", "bin")
]
