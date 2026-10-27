# Qwen3.5-4B on the NPU: its own decode program (kernels/) and the host code
# wired to it. The CPU side (config, reference, Mixer) is the family's shared
# gufo_npu_qwen35.

add_library(gufo_npu_qwen35_4b STATIC npu_decoder.cpp npu_prefill.cpp)
target_link_libraries(gufo_npu_qwen35_4b PUBLIC gufo_npu_qwen35_npu)
set_source_files_properties(npu_decoder.cpp npu_prefill.cpp PROPERTIES
  COMPILE_OPTIONS "-march=x86-64-v3;-O3")
# Its NPU programs (kernels/) are registered from kernels/CMakeLists.txt.
