cmake_minimum_required(VERSION 3.25)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" GUFO_NPU_VERSION)
string(STRIP "${GUFO_NPU_VERSION}" GUFO_NPU_VERSION)

project(gufo_npu VERSION ${GUFO_NPU_VERSION} LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  message(FATAL_ERROR "gufo-npu targets Linux x86-64 (AMD Strix Halo) only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(GUFO_NPU_ENABLE_XRT "Build the XRT (NPU) runtime path" ON)
option(GUFO_NPU_BUILD_KERNELS "Compile NPU kernels with mlir-aie/IRON" ON)
option(BUILD_TESTING "Build tests" ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

include(GNUInstallDirs)
set(GUFO_NPU_KERNEL_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/gufo-npu/kernels")

configure_file(src/core/build_info.hpp.in
               "${CMAKE_CURRENT_BINARY_DIR}/generated/build_info.hpp" @ONLY)

find_package(ICU REQUIRED COMPONENTS uc i18n)

# Vendored header-only dependencies (see NOTICE).
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
add_library(cpp_httplib INTERFACE)
target_include_directories(cpp_httplib SYSTEM INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
find_package(Threads REQUIRED)
target_link_libraries(cpp_httplib INTERFACE Threads::Threads)

# CPU-only core: no XRT, safe for hosted CI and unit tests.
add_library(gufo_npu_core STATIC
  src/core/bfp16.cpp
  src/core/diagnose.cpp
  src/core/gguf.cpp
  src/core/q4nx.cpp
  src/core/quant.cpp
  src/core/sampling.cpp
  src/core/tokenizer.cpp
)
target_link_libraries(gufo_npu_core PUBLIC ICU::uc ICU::i18n)
target_include_directories(gufo_npu_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")

# The server's chat format: OpenAI messages to the Qwen3.5 / Qwen3.6 chat
# template and the model's output back to content, reasoning and tool calls.
add_library(gufo_npu_serve_chat STATIC src/serve/chat_format_qwen35.cpp)
target_link_libraries(gufo_npu_serve_chat PUBLIC gufo_npu_core nlohmann_json)

# Qwen3.5 / Qwen3.6 model code (CPU reference).
add_library(gufo_npu_qwen35 STATIC
  src/models/qwen35/chat.cpp
  src/models/qwen35/config.cpp
  src/models/qwen35/decoder.cpp
  src/models/qwen35/draft_vocab.cpp
  src/models/qwen35/generate.cpp
  src/models/qwen35/gguf_projections.cpp
  src/models/qwen35/moe.cpp
  src/models/qwen35/mtp.cpp
  src/models/qwen35/prompt_cache.cpp
  src/models/qwen35/reference.cpp
)
target_link_libraries(gufo_npu_qwen35 PUBLIC gufo_npu_core)
# The decode steps' CPU math vectorizes with AVX2 + FMA (Zen 5 has more; the
# hosted CI runners have at least this).
set_source_files_properties(src/models/qwen35/decoder.cpp
  src/models/qwen35/gguf_projections.cpp
  src/models/qwen35/npu_decoder.cpp PROPERTIES
  COMPILE_OPTIONS "-march=x86-64-v3;-O3")
# The q4nx packer's fit vectorizes too; no fused multiply-adds, so Unpack and
# Gemv round exactly as written (tests compare unpacked weights bit for bit).
# Sampling weighs the whole vocabulary once per token; its exp loop
# vectorizes once compares may not trap (nothing here reads FP exceptions).
set_source_files_properties(src/core/sampling.cpp PROPERTIES
  COMPILE_OPTIONS "-march=x86-64-v3;-O3;-fno-trapping-math")
set_source_files_properties(src/core/q4nx.cpp PROPERTIES
  COMPILE_OPTIONS "-march=x86-64-v3;-O3;-ffp-contract=off")

# NPU runtime: XRT when enabled, otherwise a stub that reports it is absent.
if(GUFO_NPU_ENABLE_XRT)
  find_package(XRT REQUIRED)
  add_library(gufo_npu_runtime STATIC src/npu/device_xrt.cpp
                                     src/npu/gemm_xrt.cpp
                                     src/npu/gemv_xrt.cpp
                                     src/npu/program_xrt.cpp)
  target_link_libraries(gufo_npu_runtime PUBLIC gufo_npu_core XRT::xrt_coreutil)
else()
  add_library(gufo_npu_runtime STATIC src/npu/device_stub.cpp
                                     src/npu/gemm_stub.cpp
                                     src/npu/gemv_stub.cpp
                                     src/npu/program_stub.cpp)
  target_link_libraries(gufo_npu_runtime PUBLIC gufo_npu_core)
endif()

# Qwen3.5 decode on the NPU, and the same q4nx weights on the CPU for tests.
add_library(gufo_npu_qwen35_npu STATIC src/models/qwen35/npu_decoder.cpp
                                       src/models/qwen35/q4nx_projections.cpp)
target_link_libraries(gufo_npu_qwen35_npu PUBLIC gufo_npu_qwen35
                      gufo_npu_runtime)
# Prompt processing on the NPU (GEMMs) with the mixers batched on the CPU
# (prefill_mixer: the family's, shared with the 35B's prefill).
target_sources(gufo_npu_qwen35_npu PRIVATE src/models/qwen35/npu_prefill.cpp
                                           src/models/qwen35/prefill_mixer.cpp)
set_source_files_properties(src/models/qwen35/npu_prefill.cpp
  src/models/qwen35/prefill_mixer.cpp PROPERTIES
  COMPILE_OPTIONS "-march=x86-64-v3;-O3")

if(GUFO_NPU_BUILD_KERNELS)
  add_subdirectory(kernels)
endif()

# Per-model packages: their own NPU programs and host wiring.
add_subdirectory(src/models/qwen35_4b)
add_subdirectory(src/models/qwen36_35b_a3b)
# The OpenAI-compatible server's model side (src/serve).
add_subdirectory(src/serve)

# OpenAI-compatible server (docs/SERVING.md): HTTP, OpenAI JSON, queue and
# metrics over the Engine and ChatFormat interfaces. CPU-only.
add_library(gufo_npu_serve STATIC
  src/serve/engine_pool.cpp
  src/serve/metrics.cpp
  src/serve/openai.cpp
  src/serve/reply.cpp
  src/serve/server.cpp
  src/serve/text.cpp
)
target_link_libraries(gufo_npu_serve PUBLIC gufo_npu_core nlohmann_json
                      PRIVATE cpp_httplib)

add_executable(gufo-npu src/cli/main.cpp)
target_link_libraries(gufo-npu PRIVATE gufo_npu_qwen35_npu gufo_npu_qwen35_4b
                      gufo_npu_qwen36_35b_a3b gufo_npu_serve
                      gufo_npu_serve_chat gufo_npu_serve_engine)
install(TARGETS gufo-npu RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
