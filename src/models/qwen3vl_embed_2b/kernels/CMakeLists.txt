# Qwen3-VL-Embedding-2B's NPU programs, registered from
# kernels/CMakeLists.txt: the bfp16 GEMM with this model's shapes
# (embed_gemm_variants.py) and the ViT attention (vit_attention_variants.py).

gufo_npu_add_kernel(NAME qwen3vl_embed_2b_gemm DESIGN embed_gemm.py
                    SOURCES embed_gemm_variants.py
                    MODEL qwen3vl_embed_2b)

# The ViT's attention (VISION.md stage 3): flash attention on the whole
# array, one instruction sequence per padded patch count.
gufo_npu_add_kernel(NAME qwen3vl_embed_2b_vit_attn DESIGN vit_attention.py
                    SOURCES aie/vit_attention.cc vit_attention_variants.py
                    MODEL qwen3vl_embed_2b)
