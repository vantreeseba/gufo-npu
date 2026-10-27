# NPU kernels: IRON / mlir-aie designs compiled to .xclbin + instruction ELF.
# Compiling targets the npu2 (XDNA2) family and needs no NPU attached.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GUFO_NPU_KERNEL_DEVICE "npu2" CACHE STRING "mlir-aie device family")
set(GUFO_NPU_KERNEL_OUT "${CMAKE_BINARY_DIR}/kernels")

# One entry per design in designs/<name>.py.
set(GUFO_NPU_KERNEL_DESIGNS smoke_add_one bw_memcpy bw_read q4nx_gemv)
# Designs that write other files than <name>.xclbin + <name>.insts.elf list
# them here (must match OUTPUTS in the design); <name>_SOURCES are extra
# inputs.
set(q4nx_gemv_OUTPUTS
  q4nx_gemv.xclbin
  q4nx_gemv_k2048_n2048_plain_store_t1.insts.elf
  q4nx_gemv_k2048_n2048_plain_store_t2.insts.elf
  q4nx_gemv_k4096_n2048_plain_store_t1.insts.elf
  q4nx_gemv_k4096_n2048_plain_store_t2.insts.elf
  q4nx_gemv_k2048_n5120_resnorm_store_t1.insts.elf
  q4nx_gemv_k2048_n5120_resnorm_store_t2.insts.elf
  q4nx_gemv_k2048_n8192_resnorm_store_t1.insts.elf
  q4nx_gemv_k2048_n8192_resnorm_store_t2.insts.elf
  q4nx_gemv_k2048_n12288_resnorm_swiglu_t1.insts.elf
  q4nx_gemv_k2048_n12288_resnorm_swiglu_t2.insts.elf
  q4nx_gemv_k6144_n2048_act_store_t1.insts.elf
  q4nx_gemv_k6144_n2048_act_store_t2.insts.elf
  q4nx_gemv_k2048_n248832_resnorm_store_t1.insts.elf
  q4nx_gemv_k2048_n248832_resnorm_store_t2.insts.elf
  q4nx_gemv_k2048_n32768_resnorm_store_t1.insts.elf
  q4nx_gemv_k2048_n32768_resnorm_store_t2.insts.elf)
set(q4nx_gemv_SOURCES aie/q4nx_gemv.cc)

set(kernel_outputs)
foreach(design IN LISTS GUFO_NPU_KERNEL_DESIGNS)
  set(files ${design}.xclbin ${design}.insts.elf)
  if(DEFINED ${design}_OUTPUTS)
    set(files ${${design}_OUTPUTS})
  endif()
  set(outputs)
  foreach(file IN LISTS files)
    list(APPEND outputs "${GUFO_NPU_KERNEL_OUT}/${file}")
  endforeach()
  add_custom_command(
    OUTPUT ${outputs}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/build.py"
            --dev "${GUFO_NPU_KERNEL_DEVICE}"
            --out "${GUFO_NPU_KERNEL_OUT}" ${design}
    DEPENDS build.py "designs/${design}.py" ${${design}_SOURCES}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Compiling NPU kernel ${design}"
    VERBATIM)
  list(APPEND kernel_outputs ${outputs})
endforeach()

add_custom_target(gufo_npu_kernels ALL DEPENDS ${kernel_outputs})
install(FILES ${kernel_outputs} DESTINATION "${GUFO_NPU_KERNEL_INSTALL_DIR}")
