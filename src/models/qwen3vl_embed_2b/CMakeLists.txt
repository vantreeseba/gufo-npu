# Qwen3-VL-Embedding-2B (GGUF arch qwen3vl), text embeddings: input
# formatting, the CPU reference and the NPU embedder over the model's own
# GEMM program (kernels/, registered from kernels/CMakeLists.txt). Nothing
# here is shared with the Qwen3.5 family.

add_library(gufo_npu_qwen3vl_embed_2b STATIC config.cpp input.cpp ops.cpp
                                             reference.cpp vision.cpp)
target_link_libraries(gufo_npu_qwen3vl_embed_2b PUBLIC gufo_npu_core)

add_library(gufo_npu_qwen3vl_embed_2b_npu STATIC npu_embedder.cpp)
target_link_libraries(gufo_npu_qwen3vl_embed_2b_npu PUBLIC
                      gufo_npu_qwen3vl_embed_2b gufo_npu_runtime)

# The CPU side vectorizes with AVX2 + FMA, as the prefills do.
set_source_files_properties(ops.cpp reference.cpp vision.cpp npu_embedder.cpp
  PROPERTIES
  COMPILE_OPTIONS "-march=x86-64-v3;-O3")
