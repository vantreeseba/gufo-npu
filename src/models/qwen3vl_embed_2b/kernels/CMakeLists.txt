# Qwen3-VL-Embedding-2B's NPU program: the bfp16 GEMM with this model's
# shapes (embed_gemm_variants.py), registered from kernels/CMakeLists.txt.

gufo_npu_add_kernel(NAME qwen3vl_embed_2b_gemm DESIGN embed_gemm.py
                    SOURCES embed_gemm_variants.py
                    MODEL qwen3vl_embed_2b)
