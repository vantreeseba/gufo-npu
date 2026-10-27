# NPU kernels: IRON / mlir-aie designs compiled to .xclbin + instruction ELF.
# Compiling targets the npu2 (XDNA2) family and needs no NPU attached.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GUFO_NPU_KERNEL_DEVICE "npu2" CACHE STRING "mlir-aie device family")
set(GUFO_NPU_KERNEL_OUT "${CMAKE_BINARY_DIR}/kernels")

# One entry per design in designs/<name>.py.
set(GUFO_NPU_KERNEL_DESIGNS smoke_add_one bw_memcpy q4nx_gemv)
# Designs that emit several shapes list them (must match their ARTIFACTS).
set(q4nx_gemv_ARTIFACTS
  q4nx_gemv_k2048_n2048
  q4nx_gemv_k2048_n6144
  q4nx_gemv_k6144_n2048
  q4nx_gemv_k2048_n248832_norm)
set(q4nx_gemv_SOURCES aie/q4nx_gemv.cc)

set(kernel_outputs)
foreach(design IN LISTS GUFO_NPU_KERNEL_DESIGNS)
  set(artifacts ${design})
  if(DEFINED ${design}_ARTIFACTS)
    set(artifacts ${${design}_ARTIFACTS})
  endif()
  set(outputs)
  foreach(artifact IN LISTS artifacts)
    list(APPEND outputs
      "${GUFO_NPU_KERNEL_OUT}/${artifact}.xclbin"
      "${GUFO_NPU_KERNEL_OUT}/${artifact}.insts.elf")
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
