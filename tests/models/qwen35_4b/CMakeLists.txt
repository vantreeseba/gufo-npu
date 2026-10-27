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
