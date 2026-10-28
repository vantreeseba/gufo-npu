# Qwen3.5-2B (docs/models/qwen3.5-2b). Model tests skip (77) without
# GUFO_NPU_QWEN35_2B_GGUF (BF16, for the HF oracle checks),
# GUFO_NPU_QWEN35_2B_Q4_GGUF (UD-Q4_K_XL) and GUFO_NPU_QWEN35_2B_ORACLE.

add_executable(qwen35_gguf_load_test gguf_load_test.cpp)
target_link_libraries(qwen35_gguf_load_test PRIVATE gufo_npu_core)
add_test(NAME qwen35_gguf_load_test COMMAND qwen35_gguf_load_test)
set_tests_properties(qwen35_gguf_load_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS "qwen35_2b;model")

add_executable(qwen35_reference_oracle_test reference_oracle_test.cpp)
target_link_libraries(qwen35_reference_oracle_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen35_reference_oracle_test COMMAND qwen35_reference_oracle_test)
set_tests_properties(qwen35_reference_oracle_test PROPERTIES
                     SKIP_RETURN_CODE 77 LABELS "qwen35_2b;model")

add_executable(qwen35_generate_oracle_test generate_oracle_test.cpp)
target_link_libraries(qwen35_generate_oracle_test PRIVATE gufo_npu_qwen35
                      nlohmann_json)
add_test(NAME qwen35_generate_oracle_test COMMAND qwen35_generate_oracle_test)
set_tests_properties(qwen35_generate_oracle_test PROPERTIES
                     SKIP_RETURN_CODE 77 LABELS "qwen35_2b;model")

add_executable(qwen35_decoder_test decoder_test.cpp)
target_link_libraries(qwen35_decoder_test PRIVATE gufo_npu_qwen35_npu)
add_test(NAME qwen35_decoder_test COMMAND qwen35_decoder_test)
set_tests_properties(qwen35_decoder_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS "qwen35_2b;model")

if(GUFO_NPU_ENABLE_XRT AND GUFO_NPU_BUILD_KERNELS)
  set(kernels "${CMAKE_BINARY_DIR}/kernels")

  add_executable(qwen35_npu_decode_test npu_decode_test.cpp)
  target_link_libraries(qwen35_npu_decode_test PRIVATE gufo_npu_qwen35_npu)
  add_test(NAME qwen35_npu_decode_test
           COMMAND qwen35_npu_decode_test "${kernels}")
  set_tests_properties(qwen35_npu_decode_test PROPERTIES
                       LABELS "qwen35_2b;npu;model" SKIP_RETURN_CODE 77)

  # Speculative sampling: MTP drafts keep the sampled distribution.
  add_executable(qwen35_npu_sampling_test npu_sampling_test.cpp)
  target_link_libraries(qwen35_npu_sampling_test PRIVATE gufo_npu_qwen35_npu)
  add_test(NAME qwen35_npu_sampling_test COMMAND qwen35_npu_sampling_test "${kernels}")
  set_tests_properties(qwen35_npu_sampling_test PROPERTIES
                       LABELS "qwen35_2b;npu;model" SKIP_RETURN_CODE 77 TIMEOUT 3600)

  add_executable(qwen35_npu_prefill_test npu_prefill_test.cpp)
  target_link_libraries(qwen35_npu_prefill_test PRIVATE gufo_npu_qwen35_npu)
  add_test(NAME qwen35_npu_prefill_test
           COMMAND qwen35_npu_prefill_test "${kernels}")
  set_tests_properties(qwen35_npu_prefill_test PROPERTIES
                       LABELS "qwen35_2b;npu;model" SKIP_RETURN_CODE 77)

  # Op tests of the kernels this model runs.
  add_executable(q4nx_gemv_test q4nx_gemv_test.cpp)
  target_link_libraries(q4nx_gemv_test PRIVATE gufo_npu_runtime)
  add_test(NAME q4nx_gemv_test COMMAND q4nx_gemv_test "${kernels}")
  set_tests_properties(q4nx_gemv_test PROPERTIES
                       LABELS "qwen35_2b;npu;kernels" SKIP_RETURN_CODE 77)

  add_executable(q4nx_gemv_chain_test q4nx_gemv_chain_test.cpp)
  target_link_libraries(q4nx_gemv_chain_test PRIVATE gufo_npu_runtime)
  add_test(NAME q4nx_gemv_chain_test COMMAND q4nx_gemv_chain_test "${kernels}")
  set_tests_properties(q4nx_gemv_chain_test PROPERTIES
                       LABELS "qwen35_2b;npu;kernels" SKIP_RETURN_CODE 77
                       TIMEOUT 600)

  add_executable(bfp16_gemm_test bfp16_gemm_test.cpp)
  target_link_libraries(bfp16_gemm_test PRIVATE gufo_npu_runtime)
  add_test(NAME bfp16_gemm_test COMMAND bfp16_gemm_test "${kernels}")
  set_tests_properties(bfp16_gemm_test PROPERTIES
                       LABELS "qwen35_2b;npu;kernels" SKIP_RETURN_CODE 77)
endif()
