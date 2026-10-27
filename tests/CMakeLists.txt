# Tests labelled "npu" need /dev/accel/accel0; everything else runs on hosted
# CI. NPU tests exit 77 (skip) when the device is missing, which is not a pass.
# Model tests live with their model (models/<model>/CMakeLists.txt) and carry
# its label (qwen35_2b, qwen36_35b_a3b), plus "model" when they need weights
# and "kernels" for NPU op tests. Shared code is tested here and in
# models/qwen35 (the Qwen3.5 family's CPU code, label qwen35_family).

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

# The same source twice: with the decoders' flags (F16C, AVX2, FMA paths)
# and with the default flags (portable fallbacks).
add_executable(vec_test core/vec_test.cpp)
target_link_libraries(vec_test PRIVATE gufo_npu_core)
target_compile_options(vec_test PRIVATE -march=x86-64-v3)
add_test(NAME vec_test COMMAND vec_test)
add_executable(vec_portable_test core/vec_test.cpp)
target_link_libraries(vec_portable_test PRIVATE gufo_npu_core)
add_test(NAME vec_portable_test COMMAND vec_portable_test)

add_executable(burst_pool_test core/burst_pool_test.cpp)
target_link_libraries(burst_pool_test PRIVATE gufo_npu_core)
add_test(NAME burst_pool_test COMMAND burst_pool_test)

add_executable(simd_test core/simd_test.cpp)
target_link_libraries(simd_test PRIVATE gufo_npu_core)
# The vector paths, as the prefill builds them.
target_compile_options(simd_test PRIVATE -march=x86-64-v3)
add_test(NAME simd_test COMMAND simd_test)

add_executable(bfp16_test core/bfp16_test.cpp)
target_link_libraries(bfp16_test PRIVATE gufo_npu_core)
add_test(NAME bfp16_test COMMAND bfp16_test)

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

add_test(NAME cli_version COMMAND gufo-npu version)
set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION
                     "^gufo-npu ${GUFO_NPU_VERSION}")

if(GUFO_NPU_ENABLE_XRT AND GUFO_NPU_BUILD_KERNELS)
  add_test(NAME npu_smoke
           COMMAND gufo-npu smoke --iters 200
                   --kernels "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(npu_smoke PROPERTIES LABELS npu)
endif()

add_subdirectory(models/qwen35)
add_subdirectory(models/qwen35_2b)
add_subdirectory(models/qwen36_35b_a3b)
