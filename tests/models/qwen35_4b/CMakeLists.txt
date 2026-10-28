# Qwen3.5-4B (arch qwen35, dense, tied embeddings).
#   GUFO_NPU_QWEN35_4B_GGUF    its UD-Q4_K_XL GGUF
#   GUFO_NPU_QWEN35_4B_ORACLE  the llama.cpp oracle's output
#                              (tools/reference/qwen36_llama_oracle.py)
# Model tests exit 77 (skip) when these are unset.

add_executable(qwen35_4b_reference_llama_test reference_llama_test.cpp)
target_link_libraries(qwen35_4b_reference_llama_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen35_4b_reference_llama_test
         COMMAND qwen35_4b_reference_llama_test)
set_tests_properties(qwen35_4b_reference_llama_test PROPERTIES
                     SKIP_RETURN_CODE 77 LABELS "qwen35_4b;model" TIMEOUT 1800)

add_executable(qwen35_4b_decoder_test decoder_test.cpp)
target_link_libraries(qwen35_4b_decoder_test PRIVATE gufo_npu_qwen35_npu)
add_test(NAME qwen35_4b_decoder_test COMMAND qwen35_4b_decoder_test)
set_tests_properties(qwen35_4b_decoder_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS "qwen35_4b;model")

if(GUFO_NPU_ENABLE_XRT AND GUFO_NPU_BUILD_KERNELS)
  add_executable(qwen35_4b_decode_gemv_test decode_gemv_test.cpp)
  target_link_libraries(qwen35_4b_decode_gemv_test PRIVATE gufo_npu_runtime)
  add_test(NAME qwen35_4b_decode_gemv_test
           COMMAND qwen35_4b_decode_gemv_test "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(qwen35_4b_decode_gemv_test PROPERTIES
                       LABELS "qwen35_4b;npu;kernels" SKIP_RETURN_CODE 77)

  add_executable(qwen35_4b_npu_decode_test npu_decode_test.cpp)
  target_link_libraries(qwen35_4b_npu_decode_test PRIVATE gufo_npu_qwen35_4b)
  add_test(NAME qwen35_4b_npu_decode_test
           COMMAND qwen35_4b_npu_decode_test "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(qwen35_4b_npu_decode_test PROPERTIES
                       LABELS "qwen35_4b;npu;model" SKIP_RETURN_CODE 77)

  # Speculative sampling: MTP drafts keep the sampled distribution.
  add_executable(qwen35_4b_npu_sampling_test npu_sampling_test.cpp)
  target_link_libraries(qwen35_4b_npu_sampling_test PRIVATE gufo_npu_qwen35_4b)
  add_test(NAME qwen35_4b_npu_sampling_test COMMAND qwen35_4b_npu_sampling_test "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(qwen35_4b_npu_sampling_test PROPERTIES
                       LABELS "qwen35_4b;npu;model" SKIP_RETURN_CODE 77 TIMEOUT 3600)

  add_executable(qwen35_4b_npu_prefill_test npu_prefill_test.cpp)
  target_link_libraries(qwen35_4b_npu_prefill_test PRIVATE gufo_npu_qwen35_4b)
  add_test(NAME qwen35_4b_npu_prefill_test
           COMMAND qwen35_4b_npu_prefill_test "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(qwen35_4b_npu_prefill_test PROPERTIES
                       LABELS "qwen35_4b;npu;model" SKIP_RETURN_CODE 77)

  add_executable(qwen35_4b_prompt_cache_test prompt_cache_test.cpp)
  target_link_libraries(qwen35_4b_prompt_cache_test PRIVATE gufo_npu_qwen35_4b)
  add_test(NAME qwen35_4b_prompt_cache_test
           COMMAND qwen35_4b_prompt_cache_test "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(qwen35_4b_prompt_cache_test PROPERTIES
                       LABELS "qwen35_4b;npu;model" SKIP_RETURN_CODE 77
                       TIMEOUT 1800)
endif()
