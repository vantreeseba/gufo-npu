# Qwen3.6-35B-A3B on the NPU: its prompt processing (npu_prefill) and its
# own NPU programs for it (kernels/, registered from kernels/CMakeLists.txt).
# Its decode and the CPU side (config, reference, Mixer, routing, the
# prefill mixers) are the family's shared gufo_npu_qwen35(_npu).

add_library(gufo_npu_qwen36_35b_a3b STATIC npu_prefill.cpp npu_experts.cpp)
target_link_libraries(gufo_npu_qwen36_35b_a3b PUBLIC gufo_npu_qwen35_npu)
set_source_files_properties(npu_prefill.cpp npu_experts.cpp PROPERTIES
  COMPILE_OPTIONS "-march=x86-64-v3;-O3")

# Its measurement tools (tools/qwen36_35b_a3b; docs/research/NPU-MOE-PREFILL.md):
# the prefill's NPU parts on random weights, and real routing over a prompt.
if(GUFO_NPU_ENABLE_XRT)
  add_executable(qwen36_prefill_spike
                 "${PROJECT_SOURCE_DIR}/tools/qwen36_35b_a3b/prefill_spike.cpp")
  target_link_libraries(qwen36_prefill_spike PRIVATE gufo_npu_runtime)
  target_compile_options(qwen36_prefill_spike PRIVATE -O2)
  add_executable(qwen36_route_dump
                 "${PROJECT_SOURCE_DIR}/tools/qwen36_35b_a3b/route_dump.cpp")
  target_link_libraries(qwen36_route_dump PRIVATE gufo_npu_qwen35_npu)
endif()
