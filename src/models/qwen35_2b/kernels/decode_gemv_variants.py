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


# Chains: one instruction sequence runs a layer's calls from its mixer output
# to the next projection (out, gate + up with SwiGLU, down, then the next
# call), so a submission is one call instead of four. The calls read and
# write regions of one activation buffer, laid out per layer in blocks
# (src/models/qwen35_2b/npu/gemv.hpp mirrors this layout):
#   block: in_x | ffn_x | in_y | out_x | ffn_y
#   head:  x | r | y      (the LM heads' input, residual and logits)
# A chain's window starts at its block's ffn_x and runs into the next block
# (the next layer's, or a head), whose x, r and y sit where a block's in_x,
# ffn_x and in_y do. Bytes throughout.
INPUT_BYTES = TOKENS * SLOT * 4 + HIDDEN * 2  # an input region (kInputBytes)
IN_Y_BYTES = TOKENS * 8192 * 4  # the longest input projection's y (DeltaNet)
FFN_Y_BYTES = 8 * (MAX_DIM // 2) * 4  # a SwiGLU y (Q4nxGemv::y_bytes)
BLOCK_IN_X = 0
BLOCK_FFN_X = INPUT_BYTES
BLOCK_IN_Y = 2 * INPUT_BYTES
BLOCK_OUT_X = 2 * INPUT_BYTES + IN_Y_BYTES
BLOCK_FFN_Y = 3 * INPUT_BYTES + IN_Y_BYTES
BLOCK_BYTES = 3 * INPUT_BYTES + IN_Y_BYTES + FFN_Y_BYTES
# Offsets inside a chain's window.
WINDOW_FFN_X = 0
WINDOW_OUT_X = BLOCK_OUT_X - BLOCK_FFN_X
WINDOW_FFN_Y = BLOCK_FFN_Y - BLOCK_FFN_X
WINDOW_NEXT = BLOCK_BYTES - BLOCK_FFN_X  # the next block's (or head's) x
WINDOW_NEXT_R = WINDOW_NEXT + BLOCK_FFN_X
WINDOW_NEXT_Y = WINDOW_NEXT + BLOCK_IN_Y
D_OFFSET = HIDDEN * 4  # the D half of a slot

# The call after a layer's FFN, by name: (rows) of a resnorm/store call on
# the hidden vector, or None for a chain that stops after the FFN (the last
# layer of a prompt pass whose logits nobody reads).
CHAIN_NEXT = {
    "deltanet": 8192,  # the next layer's DeltaNet input
    "attention": 5120,  # the next layer's attention input
    "lm": 248832,  # the LM head
    "draft": 32768,  # the MTP draft's LM head
    "bare": None,
}
# A chain's calls before the next one: out, gate + up, down, as (dim, rows,
# input, output, Y token stride) of VARIANTS.
CHAIN_CALLS = [
    (HIDDEN, HIDDEN, "plain", "store", SLOT),
    (HIDDEN, 2 * FFN, "resnorm", "swiglu", MAX_DIM // 2),
    (FFN, HIDDEN, "act", "store", SLOT),
]


def chain_name(next_name, tokens) -> str:
    """Matches qwen35_2b::npu::ChainSequence."""
    return f"{NAME}_chain_{next_name}_t{tokens}"


def chain_window_bytes(next_name) -> int:
    """The window a chain reads and writes (its activation argument)."""
    rows = CHAIN_NEXT[next_name]
    if rows is None:
        return WINDOW_NEXT + INPUT_BYTES
    return WINDOW_NEXT_Y + ((TOKENS - 1) * rows + rows) * 4


def chains():
    for next_name in CHAIN_NEXT:
        for tokens in range(1, TOKENS + 1):
            yield (next_name, tokens)


# The xclbin, then one instruction ELF per variant and per chain (the ELF
# flow only).
OUTPUTS = (
    [f"{NAME}.xclbin"]
    + [f"{sequence_name(d, n, p, e, t)}.insts.elf" for d, n, p, e, _, t in sequences()]
    + [f"{chain_name(n, t)}.insts.elf" for n, t in chains()]
)
