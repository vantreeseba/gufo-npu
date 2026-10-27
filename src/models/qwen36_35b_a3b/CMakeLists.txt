# Qwen3.6-35B-A3B on the NPU: its prompt processing (npu_prefill) and its
# own NPU programs for it (kernels/, registered from kernels/CMakeLists.txt).
# Its decode and the CPU side (config, reference, Mixer, routing, the
# prefill mixers) are the family's shared gufo_npu_qwen35(_npu).

add_library(gufo_npu_qwen36_35b_a3b STATIC npu_prefill.cpp)
target_link_libraries(gufo_npu_qwen36_35b_a3b PUBLIC gufo_npu_qwen35_npu)
set_source_files_properties(npu_prefill.cpp PROPERTIES
  COMPILE_OPTIONS "-march=x86-64-v3;-O3")
