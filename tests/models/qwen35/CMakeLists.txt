# The Qwen3.5 family's shared CPU code (config, reference, decoder, MTP,
# routing), on synthetic inputs; each model's own checks are in its folder.

add_executable(qwen35_config_test config_test.cpp)
target_link_libraries(qwen35_config_test PRIVATE gufo_npu_qwen35)
add_test(NAME qwen35_config_test COMMAND qwen35_config_test)
set_tests_properties(qwen35_config_test PROPERTIES LABELS qwen35)
