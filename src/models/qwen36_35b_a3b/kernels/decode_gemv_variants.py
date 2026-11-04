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


# Chains: one instruction sequence runs a MoE layer's calls after the CPU
# router (the gate + up gather, the down gather, then the next call), so the
# submission is one call instead of three. The calls read and write regions
# of one activation buffer, laid out per layer in blocks
# (src/models/qwen36_35b_a3b/npu/gemv.hpp mirrors this layout):
#   block: in_x | ffn_x | in_y | out_x | ffn_y | scratch
#   head:  x | r | y      (the LM heads' input, residual and logits)
# A chain's window starts at its block's ffn_x and runs into the next block
# (the next layer's, or a head), whose x, r and y sit where a block's in_x,
# ffn_x and in_y do. Bytes throughout. Runtime arguments: the window, the
# layer's expert arena twice (the gate + up gather patches only the second
# argument, the down gather only the third), the next call's weights.
INPUT_BYTES = (TOKENS * SLOT + HIDDEN // 2) * 4  # an input region (X_FLOATS)
IN_Y_BYTES = TOKENS * 12288 * 4  # the longest input projection's y (DeltaNet)
FFN_Y_BYTES = 8 * (MAX_DIM // 2) * 4  # a gate + up gather's y (y_bytes)
SCRATCH_BYTES = (SLOT + HIDDEN) * 4  # a residual argument nobody reads
BLOCK_IN_X = 0
BLOCK_FFN_X = INPUT_BYTES
BLOCK_IN_Y = 2 * INPUT_BYTES
BLOCK_OUT_X = 2 * INPUT_BYTES + IN_Y_BYTES
BLOCK_FFN_Y = 3 * INPUT_BYTES + IN_Y_BYTES
BLOCK_SCRATCH = 3 * INPUT_BYTES + IN_Y_BYTES + FFN_Y_BYTES
BLOCK_BYTES = BLOCK_SCRATCH + SCRATCH_BYTES
# Offsets inside a chain's window.
WINDOW_FFN_X = 0
WINDOW_FFN_Y = BLOCK_FFN_Y - BLOCK_FFN_X
WINDOW_SCRATCH = BLOCK_SCRATCH - BLOCK_FFN_X
WINDOW_NEXT = BLOCK_BYTES - BLOCK_FFN_X  # the next block's (or head's) x
WINDOW_NEXT_R = WINDOW_NEXT + BLOCK_FFN_X
WINDOW_NEXT_Y = WINDOW_NEXT + BLOCK_IN_Y
D_OFFSET = HIDDEN * 4  # the D half of a slot

# The call after a layer's FFN, by name: the rows of a resnorm/store call on
# the hidden vector, or None for a chain that stops after the FFN (the last
# layer of a prompt pass whose logits nobody reads).
CHAIN_NEXT = {
    "deltanet": 12288,  # the next layer's DeltaNet input
    "attention": 9216,  # the next layer's attention input
    "lm": 248832,  # the LM head
    "draft": 32768,  # the MTP draft's LM head
    "bare": None,
}


def chain_name(next_name, experts, tokens) -> str:
    """Matches qwen36_35b_a3b::npu::ChainSequence."""
    return f"{XCLBIN}_chain_{next_name}_e{experts}_t{tokens}"


def chain_window_bytes(next_name) -> int:
    """The window a chain reads and writes (its first argument)."""
    rows = CHAIN_NEXT[next_name]
    if rows is None:
        return WINDOW_NEXT + INPUT_BYTES
    return WINDOW_NEXT_Y + TOKENS * rows * 4


def chains():
    """(next, experts, tokens): one token reads 9 experts, two read 9 to 17,
    as the gathers do."""
    for next_name in CHAIN_NEXT:
        for experts in MOE_COUNTS:
            for tokens in (1, 2) if experts == 9 else (2,):
                yield (next_name, experts, tokens)


assert all(
    offset % 64 == 0
    for offset in (BLOCK_FFN_X, BLOCK_IN_Y, BLOCK_OUT_X, BLOCK_FFN_Y, BLOCK_SCRATCH, BLOCK_BYTES)
)

# build.py checks these: the xclbin, then per variant an instruction ELF and
# the raw instruction words (the host's instruction-buffer flow), then per
# chain its raw instruction words (chains run only in that flow).
OUTPUTS = (
    [f"{XCLBIN}.xclbin"]
    + [
        f"{sequence_name(d, n, p, e, t, g, x)}.insts.{suffix}"
        for d, n, p, e, _, t, g, x in sequences()
        for suffix in ("elf", "bin")
    ]
    + [f"{chain_name(n, x, t)}.insts.bin" for n, x, t in chains()]
)
