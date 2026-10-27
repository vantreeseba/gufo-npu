# NPU kernels: IRON / mlir-aie designs compiled to .xclbin + instruction
# sequences. Compiling targets the npu2 (XDNA2) family and needs no NPU.
# Each design is registered with gufo_npu_add_kernel
# (cmake/GufoNpuKernels.cmake), naming the models it serves.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GUFO_NPU_KERNEL_DEVICE "npu2" CACHE STRING "mlir-aie device family")
set(GUFO_NPU_KERNEL_OUT "${CMAKE_BINARY_DIR}/kernels")
set(GUFO_NPU_KERNEL_BUILD_PY "${CMAKE_CURRENT_SOURCE_DIR}/build.py")

add_custom_target(gufo_npu_kernels ALL)
include(GufoNpuKernels)

# Diagnostics: the smoke kernel and the bandwidth probes.
gufo_npu_add_kernel(NAME smoke_add_one DESIGN designs/smoke_add_one.py MODEL common)
gufo_npu_add_kernel(NAME bw_memcpy DESIGN designs/bw_memcpy.py MODEL common)
gufo_npu_add_kernel(NAME bw_read DESIGN designs/bw_read.py MODEL common)

# Decode GEMVs: one core program and xclbin for the 2B's and the 35B's calls,
# including the 35B's MoE gathers (designs/q4nx_gemv_variants.py lists the
# shapes; each sequence as an instruction ELF and as raw instruction words).
gufo_npu_add_kernel(NAME q4nx_gemv DESIGN designs/q4nx_gemv.py
                    SOURCES aie/q4nx_gemv.cc
                    MODEL qwen35_2b qwen36_35b_a3b)

# Prefill GEMMs (Qwen3.5-2B; designs/bfp16_gemm_variants.py).
gufo_npu_add_kernel(NAME bfp16_gemm DESIGN designs/bfp16_gemm.py
                    MODEL qwen35_2b)
