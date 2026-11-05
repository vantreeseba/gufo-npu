# The Qwen3.5 family's shared CPU code (config, reference, decoder, MTP,
# routing), on synthetic inputs; each model's own checks are in its folder.

add_executable(qwen35_config_test config_test.cpp)
target_link_libraries(qwen35_config_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen35_config_test COMMAND qwen35_config_test)
set_tests_properties(qwen35_config_test PROPERTIES LABELS qwen35_family)

add_executable(qwen35_prompt_cache_logic_test prompt_cache_test.cpp)
target_link_libraries(qwen35_prompt_cache_logic_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen35_prompt_cache_logic_test COMMAND qwen35_prompt_cache_logic_test)
set_tests_properties(qwen35_prompt_cache_logic_test PROPERTIES LABELS qwen35_family)

# The fused GGUF row dots twice: with the decoders' flags (AVX2, FMA, F16C
# paths) and with the default flags (portable fallbacks).
add_executable(qwen35_gguf_dot_test gguf_dot_test.cpp)
target_link_libraries(qwen35_gguf_dot_test PRIVATE gufo_npu_qwen35)
target_compile_options(qwen35_gguf_dot_test PRIVATE -march=x86-64-v3)
add_test(NAME qwen35_gguf_dot_test COMMAND qwen35_gguf_dot_test)
set_tests_properties(qwen35_gguf_dot_test PROPERTIES LABELS qwen35_family)
add_executable(qwen35_gguf_dot_portable_test gguf_dot_test.cpp)
target_link_libraries(qwen35_gguf_dot_portable_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen35_gguf_dot_portable_test COMMAND qwen35_gguf_dot_portable_test)
set_tests_properties(qwen35_gguf_dot_portable_test PROPERTIES LABELS qwen35_family)
