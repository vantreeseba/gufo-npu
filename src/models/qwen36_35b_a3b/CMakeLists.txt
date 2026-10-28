# Qwen3.6-35B-A3B on the NPU: its prompt processing (npu_prefill) and its
# own NPU programs for it (kernels/, registered from kernels/CMakeLists.txt).
# Its decode and the CPU side (config, reference, Mixer, routing, the
# prefill mixers) are the family's shared gufo_npu_qwen35(_npu).

add_library(gufo_npu_qwen36_35b_a3b STATIC npu_prefill.cpp npu_experts.cpp
            cpu_experts.cpp)
target_link_libraries(gufo_npu_qwen36_35b_a3b PUBLIC gufo_npu_qwen35_npu)
# cpu_experts.cpp's AVX-512 BF16 path is chosen at run time (target
# attributes); the rest of each unit is x86-64-v3.
set_source_files_properties(npu_prefill.cpp npu_experts.cpp cpu_experts.cpp
  PROPERTIES COMPILE_OPTIONS "-march=x86-64-v3;-O3")

# Its measurement tools (tools/qwen36_35b_a3b; docs/research/NPU-MOE-PREFILL.md):
# the prefill's NPU parts on random weights, real routing over a prompt, and
# the CPU expert step on random experts (no NPU).
add_executable(qwen36_cpu_experts_bench
               "${PROJECT_SOURCE_DIR}/tools/qwen36_35b_a3b/cpu_experts_bench.cpp")
target_link_libraries(qwen36_cpu_experts_bench PRIVATE gufo_npu_qwen36_35b_a3b)
target_compile_options(qwen36_cpu_experts_bench PRIVATE -O2)
if(GUFO_NPU_ENABLE_XRT)
  add_executable(qwen36_prefill_spike
                 "${PROJECT_SOURCE_DIR}/tools/qwen36_35b_a3b/prefill_spike.cpp")
  target_link_libraries(qwen36_prefill_spike PRIVATE gufo_npu_runtime)
  target_compile_options(qwen36_prefill_spike PRIVATE -O2)
  add_executable(qwen36_route_dump
                 "${PROJECT_SOURCE_DIR}/tools/qwen36_35b_a3b/route_dump.cpp")
  target_link_libraries(qwen36_route_dump PRIVATE gufo_npu_qwen35_npu)
endif()
