"""Qwen3.6-35B-A3B's prefill dequant shapes (prefill_dequant.py): pure data,
no IRON imports, so kernels/build.py --list-outputs can read OUTPUTS at
configure time.
"""

HIDDEN = 2048
EXPERT_FFN = 512

DEQUANT_NAME = "qwen36_35b_a3b_dequant"
# (K, N) of the matrices converted per prompt, read straight from the
# decoder's q4nx buffers: the DeltaNet input (qkv + z), the attention input
# (q + gate, k, v; also the MTP block's), the mixer output (ssm_out /
# attn_output; also eh_proj), and one expert's (or the shared expert's)
# gate + up and down.
DEQUANT_SHAPES = [
    (HIDDEN, 12288),
    (HIDDEN, 9216),
    (4096, HIDDEN),
    (HIDDEN, 2 * EXPERT_FFN),
    (EXPERT_FFN, HIDDEN),
]


def dequant_sequence_name(k, n) -> str:
    return f"{DEQUANT_NAME}_k{k}_n{n}"


OUTPUTS = [f"{DEQUANT_NAME}.xclbin"] + [
    f"{dequant_sequence_name(k, n)}.insts.elf" for k, n in DEQUANT_SHAPES
]
