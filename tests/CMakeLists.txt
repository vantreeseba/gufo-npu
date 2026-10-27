# Tests labelled "npu" need /dev/accel/accel0; everything else runs on hosted
# CI. NPU tests exit 77 (skip) when the device is missing, which is not a pass.

add_executable(diagnose_test core/diagnose_test.cpp)
target_link_libraries(diagnose_test PRIVATE gufo_npu_core)
add_test(NAME diagnose_test COMMAND diagnose_test)

add_executable(gguf_test core/gguf_test.cpp)
target_link_libraries(gguf_test PRIVATE gufo_npu_core)
add_test(NAME gguf_test COMMAND gguf_test)

add_executable(quant_test core/quant_test.cpp)
target_link_libraries(quant_test PRIVATE gufo_npu_core)
add_test(NAME quant_test
         COMMAND quant_test "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/quant")

add_executable(q4nx_test core/q4nx_test.cpp)
target_link_libraries(q4nx_test PRIVATE gufo_npu_core)
add_test(NAME q4nx_test COMMAND q4nx_test)

add_executable(tokenizer_test core/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE gufo_npu_core nlohmann_json)
add_test(NAME tokenizer_test
         COMMAND tokenizer_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer/qwen35.jsonl")
set_tests_properties(tokenizer_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS model)

add_executable(qwen35_gguf_load_test models/qwen35/gguf_load_test.cpp)
target_link_libraries(qwen35_gguf_load_test PRIVATE gufo_npu_core)
add_test(NAME qwen35_gguf_load_test COMMAND qwen35_gguf_load_test)
set_tests_properties(qwen35_gguf_load_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS model)

add_executable(qwen35_reference_oracle_test
               models/qwen35/reference_oracle_test.cpp)
target_link_libraries(qwen35_reference_oracle_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen35_reference_oracle_test COMMAND qwen35_reference_oracle_test)
set_tests_properties(qwen35_reference_oracle_test PROPERTIES
                     SKIP_RETURN_CODE 77 LABELS model)

add_executable(qwen35_generate_oracle_test
               models/qwen35/generate_oracle_test.cpp)
target_link_libraries(qwen35_generate_oracle_test PRIVATE gufo_npu_qwen35
                      nlohmann_json)
add_test(NAME qwen35_generate_oracle_test COMMAND qwen35_generate_oracle_test)
set_tests_properties(qwen35_generate_oracle_test PROPERTIES
                     SKIP_RETURN_CODE 77 LABELS model)

add_executable(qwen35_decoder_test models/qwen35/decoder_test.cpp)
target_link_libraries(qwen35_decoder_test PRIVATE gufo_npu_qwen35_npu)
add_test(NAME qwen35_decoder_test COMMAND qwen35_decoder_test)
set_tests_properties(qwen35_decoder_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS model)

add_test(NAME cli_version COMMAND gufo-npu version)
set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION
                     "^gufo-npu ${GUFO_NPU_VERSION}")

if(GUFO_NPU_ENABLE_XRT AND GUFO_NPU_BUILD_KERNELS)
  add_test(NAME npu_smoke
           COMMAND gufo-npu smoke --iters 200
                   --kernels "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(npu_smoke PROPERTIES LABELS npu)

  add_executable(qwen35_npu_decode_test models/qwen35/npu_decode_test.cpp)
  target_link_libraries(qwen35_npu_decode_test PRIVATE gufo_npu_qwen35_npu)
  add_test(NAME qwen35_npu_decode_test
           COMMAND qwen35_npu_decode_test "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(qwen35_npu_decode_test PROPERTIES LABELS "npu;model"
                       SKIP_RETURN_CODE 77)

  add_executable(q4nx_gemv_test npu/q4nx_gemv_test.cpp)
  target_link_libraries(q4nx_gemv_test PRIVATE gufo_npu_runtime)
  add_test(NAME q4nx_gemv_test
           COMMAND q4nx_gemv_test "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(q4nx_gemv_test PROPERTIES LABELS npu
                       SKIP_RETURN_CODE 77)
endif()
