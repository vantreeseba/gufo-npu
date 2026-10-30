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

add_executable(thread_pool_test core/thread_pool_test.cpp)
target_link_libraries(thread_pool_test PRIVATE gufo_npu_core)
add_test(NAME thread_pool_test COMMAND thread_pool_test)
set_tests_properties(thread_pool_test PROPERTIES TIMEOUT 300)

add_executable(sampling_test core/sampling_test.cpp)
target_link_libraries(sampling_test PRIVATE gufo_npu_core)
add_test(NAME sampling_test COMMAND sampling_test)

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

# The OpenAI-compatible server over a scripted engine (no model, no NPU).
add_executable(serve_text_test serve/text_test.cpp)
target_link_libraries(serve_text_test PRIVATE gufo_npu_serve)
add_test(NAME serve_text_test COMMAND serve_text_test)
add_executable(serve_server_test serve/server_test.cpp)
target_link_libraries(serve_server_test PRIVATE gufo_npu_serve cpp_httplib)
add_test(NAME serve_server_test COMMAND serve_server_test)
# The same with the real Qwen3.5 chat format: reasoning and tool calls end
# to end.
add_executable(serve_chat_test serve/server_chat_test.cpp)
target_link_libraries(serve_chat_test PRIVATE gufo_npu_serve
                      gufo_npu_serve_chat cpp_httplib)
add_test(NAME serve_chat_test COMMAND serve_chat_test)
add_executable(serve_engine_pool_test serve/engine_pool_test.cpp)
target_link_libraries(serve_engine_pool_test PRIVATE gufo_npu_serve cpp_httplib)
add_test(NAME serve_engine_pool_test COMMAND serve_engine_pool_test)
# logprobs, n, logit_bias, parallel_tool_calls and /v1/completions.
add_executable(serve_extras_test serve/server_extras_test.cpp)
target_link_libraries(serve_extras_test PRIVATE gufo_npu_serve cpp_httplib)
add_test(NAME serve_extras_test COMMAND serve_extras_test)
# Request, generation and model-load log lines in gufo's format.
add_executable(serve_logging_test serve/logging_test.cpp)
target_link_libraries(serve_logging_test PRIVATE gufo_npu_serve cpp_httplib)
add_test(NAME serve_logging_test COMMAND serve_logging_test)
# POST /v1/embeddings over a scripted embedding engine.
add_executable(serve_embeddings_test serve/server_embeddings_test.cpp)
target_link_libraries(serve_embeddings_test PRIVATE gufo_npu_serve cpp_httplib)
add_test(NAME serve_embeddings_test COMMAND serve_embeddings_test)
set_tests_properties(serve_text_test serve_server_test serve_chat_test
                     serve_extras_test serve_engine_pool_test serve_logging_test
                     serve_embeddings_test
                     PROPERTIES LABELS serve TIMEOUT 120)
# Structured outputs: grammar masks against token-by-token simulation,
# constrained generations validated against their schema, forced tool calls,
# the server end to end, and the mask's cost on a 248k-token vocabulary.
add_executable(serve_structured_test serve/structured_test.cpp)
target_link_libraries(serve_structured_test PRIVATE gufo_npu_serve
                      gufo_npu_serve_chat cpp_httplib)
add_test(NAME serve_structured_test COMMAND serve_structured_test)
set_tests_properties(serve_structured_test PROPERTIES LABELS serve
                     TIMEOUT 600)

add_executable(tokenizer_test core/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE gufo_npu_core nlohmann_json)
add_test(NAME tokenizer_test
         COMMAND tokenizer_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer/qwen35.jsonl")
set_tests_properties(tokenizer_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS model)
# The Qwen2/Qwen3 pre-tokenizer on Qwen3-VL-Embedding-2B's vocabulary.
add_test(NAME qwen3vl_tokenizer_test
         COMMAND tokenizer_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer/qwen3vl.jsonl"
                 GUFO_NPU_QWEN3VL_EMBED_2B_GGUF)
set_tests_properties(qwen3vl_tokenizer_test PROPERTIES SKIP_RETURN_CODE 77
                     LABELS "qwen3vl_embed_2b;model")

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
add_subdirectory(models/qwen35_4b)
add_subdirectory(models/qwen3vl_embed_2b)

# The server's chat format against the GGUF chat template (fixtures rendered
# by tools/reference/qwen35_chat_template.py); no model needed.
add_executable(chat_format_test serve/chat_format_test.cpp)
target_link_libraries(chat_format_test PRIVATE gufo_npu_serve_chat
                      gufo_npu_qwen35)
add_test(NAME chat_format_test
         COMMAND chat_format_test "${CMAKE_CURRENT_SOURCE_DIR}/serve/fixtures")
set_tests_properties(chat_format_test PROPERTIES LABELS serve)
add_subdirectory(serve)
