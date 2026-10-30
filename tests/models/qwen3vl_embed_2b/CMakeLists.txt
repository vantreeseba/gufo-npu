# Qwen3-VL-Embedding-2B (arch qwen3vl), text embeddings.
#   GUFO_NPU_QWEN3VL_EMBED_2B_GGUF    its BF16 GGUF (llama.cpp's converter)
#   GUFO_NPU_QWEN3VL_EMBED_2B_ORACLE  the HF oracle's output
#                                     (tools/reference/qwen3vl_embedding_oracle.py)
# Model tests exit 77 (skip) when these are unset. The tokenizer fixtures
# test (qwen3vl_tokenizer_test) is registered with the core tests.

# The template, packing, attention, pooling: no model, hosted CI.
add_executable(qwen3vl_embed_2b_ops_test ops_test.cpp)
target_link_libraries(qwen3vl_embed_2b_ops_test PRIVATE
                      gufo_npu_qwen3vl_embed_2b)
target_compile_options(qwen3vl_embed_2b_ops_test PRIVATE -march=x86-64-v3)
add_test(NAME qwen3vl_embed_2b_ops_test COMMAND qwen3vl_embed_2b_ops_test)
set_tests_properties(qwen3vl_embed_2b_ops_test PROPERTIES
                     LABELS qwen3vl_embed_2b)

add_executable(qwen3vl_embed_2b_reference_oracle_test reference_oracle_test.cpp)
target_link_libraries(qwen3vl_embed_2b_reference_oracle_test PRIVATE
                      gufo_npu_qwen3vl_embed_2b nlohmann_json)
add_test(NAME qwen3vl_embed_2b_reference_oracle_test
         COMMAND qwen3vl_embed_2b_reference_oracle_test)
set_tests_properties(qwen3vl_embed_2b_reference_oracle_test PROPERTIES
                     SKIP_RETURN_CODE 77 LABELS "qwen3vl_embed_2b;model"
                     TIMEOUT 1800)

if(GUFO_NPU_ENABLE_XRT AND GUFO_NPU_BUILD_KERNELS)
  set(kernels "${CMAKE_BINARY_DIR}/kernels")

  add_executable(qwen3vl_embed_2b_gemm_test gemm_test.cpp)
  target_link_libraries(qwen3vl_embed_2b_gemm_test PRIVATE gufo_npu_runtime)
  add_test(NAME qwen3vl_embed_2b_gemm_test
           COMMAND qwen3vl_embed_2b_gemm_test "${kernels}")
  set_tests_properties(qwen3vl_embed_2b_gemm_test PROPERTIES
                       LABELS "qwen3vl_embed_2b;npu;kernels"
                       SKIP_RETURN_CODE 77)

  add_executable(qwen3vl_embed_2b_npu_embed_test npu_embed_test.cpp)
  target_link_libraries(qwen3vl_embed_2b_npu_embed_test PRIVATE
                        gufo_npu_qwen3vl_embed_2b_npu nlohmann_json)
  add_test(NAME qwen3vl_embed_2b_npu_embed_test
           COMMAND qwen3vl_embed_2b_npu_embed_test "${kernels}")
  set_tests_properties(qwen3vl_embed_2b_npu_embed_test PROPERTIES
                       LABELS "qwen3vl_embed_2b;npu;model" SKIP_RETURN_CODE 77
                       TIMEOUT 1800)
endif()
