# Qwen3.6-35B-A3B (arch qwen35moe; docs/models/qwen3.6-35b-a3b). Model
# tests skip (77) without GUFO_NPU_QWEN36_MODEL (the UD-Q4_K_XL GGUF) and,
# for the oracle checks, GUFO_NPU_QWEN36_ORACLE.

add_executable(qwen36_gguf_load_test gguf_load_test.cpp)
target_link_libraries(qwen36_gguf_load_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen36_gguf_load_test COMMAND qwen36_gguf_load_test)
set_tests_properties(qwen36_gguf_load_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS "qwen36_35b_a3b;model")

add_executable(qwen36_reference_llama_test reference_llama_test.cpp)
target_link_libraries(qwen36_reference_llama_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen36_reference_llama_test COMMAND qwen36_reference_llama_test)
set_tests_properties(qwen36_reference_llama_test PROPERTIES
                     SKIP_RETURN_CODE 77 LABELS "qwen36_35b_a3b;model"
                     TIMEOUT 3600)

add_executable(qwen36_mtp_test mtp_test.cpp)
target_link_libraries(qwen36_mtp_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen36_mtp_test COMMAND qwen36_mtp_test)
set_tests_properties(qwen36_mtp_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS "qwen36_35b_a3b;model" TIMEOUT 3600)

add_executable(qwen36_q4nx_requant_test q4nx_requant_test.cpp)
target_link_libraries(qwen36_q4nx_requant_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen36_q4nx_requant_test COMMAND qwen36_q4nx_requant_test)
set_tests_properties(qwen36_q4nx_requant_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS "qwen36_35b_a3b;model" TIMEOUT 7200)

# Held-out text (tools/reference/fetch_text.py) in GUFO_NPU_QWEN36_TEXT.
add_executable(qwen36_q4nx_quality_test q4nx_quality_test.cpp)
target_link_libraries(qwen36_q4nx_quality_test PRIVATE gufo_npu_qwen35)
target_compile_options(qwen36_q4nx_quality_test PRIVATE -march=x86-64-v3)
add_test(NAME qwen36_q4nx_quality_test COMMAND qwen36_q4nx_quality_test)
set_tests_properties(qwen36_q4nx_quality_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS "qwen36_35b_a3b;model" TIMEOUT 7200)

add_executable(qwen36_decoder_test decoder_test.cpp)
target_link_libraries(qwen36_decoder_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen36_decoder_test COMMAND qwen36_decoder_test)
set_tests_properties(qwen36_decoder_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS "qwen36_35b_a3b;model" TIMEOUT 3600)

if(GUFO_NPU_ENABLE_XRT AND GUFO_NPU_BUILD_KERNELS)
  set(kernels "${CMAKE_BINARY_DIR}/kernels")

  add_executable(qwen36_npu_decode_test npu_decode_test.cpp)
  target_link_libraries(qwen36_npu_decode_test PRIVATE gufo_npu_qwen35_npu)
  add_test(NAME qwen36_npu_decode_test
           COMMAND qwen36_npu_decode_test "${kernels}")
  set_tests_properties(qwen36_npu_decode_test PROPERTIES
                       LABELS "qwen36_35b_a3b;npu;model" SKIP_RETURN_CODE 77
                       TIMEOUT 3600)

  # Op tests of the MoE expert gathers.
  add_executable(q4nx_moe_gather_test q4nx_moe_gather_test.cpp)
  target_link_libraries(q4nx_moe_gather_test PRIVATE gufo_npu_runtime)
  add_test(NAME q4nx_moe_gather_test COMMAND q4nx_moe_gather_test "${kernels}")
  set_tests_properties(q4nx_moe_gather_test PROPERTIES
                       LABELS "qwen36_35b_a3b;npu;kernels" SKIP_RETURN_CODE 77)

  add_executable(q4nx_moe_chain_test q4nx_moe_chain_test.cpp)
  target_link_libraries(q4nx_moe_chain_test PRIVATE gufo_npu_runtime)
  add_test(NAME q4nx_moe_chain_test COMMAND q4nx_moe_chain_test "${kernels}")
  set_tests_properties(q4nx_moe_chain_test PROPERTIES
                       LABELS "qwen36_35b_a3b;npu;kernels" SKIP_RETURN_CODE 77
                       TIMEOUT 600)
endif()
