# The server's model side: the model registry and the Engine over each model
# package (engine.hpp). Builds without XRT too (the stub runtime reports the
# NPU as absent at load time).

add_library(gufo_npu_serve_engine STATIC engine.cpp model_registry.cpp)
target_link_libraries(gufo_npu_serve_engine PUBLIC gufo_npu_qwen35_npu
                      gufo_npu_qwen35_4b gufo_npu_qwen36_35b_a3b)

# Load once, generate many: reference prompts and random mixed-length
# requests through one engine, with per-request timing and memory.
add_executable(gufo_npu_engine_soak
               "${PROJECT_SOURCE_DIR}/tools/serve/engine_soak.cpp")
target_link_libraries(gufo_npu_engine_soak PRIVATE gufo_npu_serve_engine)
