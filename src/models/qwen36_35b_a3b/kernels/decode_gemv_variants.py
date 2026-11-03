"""Qwen3.6-35B-A3B's decode GEMV shapes (decode_gemv.py): pure data, no
IRON imports, so kernels/build.py --list-outputs can read OUTPUTS at
configure time. The host mirrors the geometry in
src/models/qwen36_35b_a3b/npu/gemv.hpp.
"""

XCLBIN = "qwen36_35b_a3b_decode"
# The longest input a call takes: a down gather over 17 experts (two tokens'
# 16 routed experts and the shared one, 17 x 512).
MAX_DIM = 8704
HIDDEN = 2048  # residual width (resnorm)
TOKENS = 2  # tokens a call can carry
SLOT = 2 * HIDDEN  # fp32 per plain/resnorm token slot ([H | D])
EXPERT_FFN = 512  # an MoE expert's width (Qwen3.6-35B-A3B)
MAX_EXPERTS = MAX_DIM // EXPERT_FFN  # experts a down gather can concatenate

# Dense decode calls as (dim, padded rows, input, output, Y token stride),
# each built for 1 and 2 tokens.
VARIANTS = [
    # Mixer outputs (ssm_out, attn_output: 4096 -> 2048) into the next
    # resnorm's D slots, and the MTP head's eh_proj (4096 -> 2048) into the
    # residual H slots.
    (4096, 2048, "plain", "store", SLOT),
    # Fused DeltaNet input: qkv (8192) + z (4096).
    (2048, 12288, "resnorm", "store", 12288),
    # Fused attention input: q + gate (8192), k (512), v (512).
    (2048, 9216, "resnorm", "store", 9216),
    # The LM head (248,320 rows padded to 248,832).
    (2048, 248832, "resnorm", "store", 248832),
    # The MTP draft's LM head over the draft vocabulary.
    (2048, 32768, "resnorm", "store", 32768),
    # One expert per call, gate + up and down: only tools/bench/npu_moe_spike
    # (the per-expert baseline the gathers replaced).
    (2048, 1024, "resnorm", "swiglu", MAX_DIM // 2),
    (512, 2048, "act", "store", SLOT),
]

# MoE gathers as (dim, rows, input, output, Y token stride, gather, experts,
# token counts). One token reads its 8 routed experts and the shared one.
# Two tokens (a speculative verify, or two prompt tokens) read the union of
# their routed experts (8 to 16) and the shared one: a variant per count, so
# a pair pays for the experts it uses.
MOE_COUNTS = range(9, MAX_EXPERTS + 1)
MOE_VARIANTS = [
    (
        2048,
        n * 1024,
        "resnorm",
        "swiglu_scaled",
        MAX_DIM // 2,
        "rounds",
        n,
        (1, 2) if n == 9 else (2,),
    )
    for n in MOE_COUNTS
] + [
    (n * EXPERT_FFN, 2048, "act", "store", SLOT, "k", n, (1, 2) if n == 9 else (2,))
    for n in MOE_COUNTS
]


def sequence_name(dim, rows, prologue, epilogue, tokens, gather="", experts=0):
    moe = f"_{gather}{experts}" if gather else ""
    return f"{XCLBIN}_k{dim}_n{rows}_{prologue}_{epilogue}{moe}_t{tokens}"


def sequences():
    for variant in VARIANTS + MOE_VARIANTS:
        dim, rows, prologue, epilogue, y_stride, *moe = variant
        gather, experts, token_counts = moe if moe else ("", 0, range(1, TOKENS + 1))
        for tokens in token_counts:
            yield (dim, rows, prologue, epilogue, y_stride, tokens, gather, experts)


# build.py checks these: the xclbin, then per variant an instruction ELF and
# the raw instruction words (the host's instruction-buffer flow).
OUTPUTS = [f"{XCLBIN}.xclbin"] + [
    f"{sequence_name(d, n, p, e, t, g, x)}.insts.{suffix}"
    for d, n, p, e, _, t, g, x in sequences()
    for suffix in ("elf", "bin")
]
