"""Qwen3-VL-Embedding-2B's ViT attention lengths (vit_attention.py): pure
data, no IRON imports, so kernels/build.py --list-outputs can read OUTPUTS at
configure time. Must match src/models/qwen3vl_embed_2b/npu/vit_attention.hpp.
"""

NAME = "qwen3vl_embed_2b_vit_attn"
HEADS = 16
HEAD_DIM = 64
# Patches per image are padded up to a multiple of BUCKET: 4 query blocks of
# 64 (one per core of a column) and 8 key blocks of 32.
BUCKET = 256
# The embedder's image cap: 1800 merged tokens of 2 x 2 patches
# (kImageMaxPixels in vision.hpp).
MAX_PATCHES = 1800 * 4
LENGTHS = list(range(BUCKET, -(-MAX_PATCHES // BUCKET) * BUCKET + 1, BUCKET))


def sequence_name(n: int) -> str:
    return f"{NAME}_n{n}"


OUTPUTS = [f"{NAME}.xclbin"] + [f"{sequence_name(n)}.insts.elf" for n in LENGTHS]
