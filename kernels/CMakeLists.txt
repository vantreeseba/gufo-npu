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
# q4nx_gemv: one sequence per variant (VARIANTS and MOE_VARIANTS in the
# design) and token count, each as an instruction ELF (the ELF flow) and as
# raw instruction words (the instruction-buffer flow, src/npu/gemv.hpp).
set(q4nx_gemv_SEQUENCES
  q4nx_gemv_k2048_n2048_plain_store
  q4nx_gemv_k4096_n2048_plain_store
  q4nx_gemv_k2048_n5120_resnorm_store
  q4nx_gemv_k2048_n8192_resnorm_store
  q4nx_gemv_k2048_n12288_resnorm_swiglu
  q4nx_gemv_k6144_n2048_act_store
  q4nx_gemv_k2048_n248832_resnorm_store
  q4nx_gemv_k2048_n32768_resnorm_store
  q4nx_gemv_k2048_n1024_resnorm_swiglu
  q4nx_gemv_k512_n2048_act_store
  q4nx_gemv_k2048_n9216_resnorm_swiglu_scaled_rounds9
  q4nx_gemv_k4608_n2048_act_store_k9
  q4nx_gemv_k2048_n12288_resnorm_store
  q4nx_gemv_k2048_n9216_resnorm_store)
set(q4nx_gemv_OUTPUTS q4nx_gemv.xclbin)
foreach(sequence IN LISTS q4nx_gemv_SEQUENCES)
  foreach(tokens 1 2)
    list(APPEND q4nx_gemv_OUTPUTS ${sequence}_t${tokens}.insts.elf
         ${sequence}_t${tokens}.insts.bin)
  endforeach()
endforeach()
# Two-token MoE gathers over 10 to 17 experts (a pair's expert union).
foreach(experts RANGE 10 17)
  math(EXPR rows "${experts} * 1024")
  math(EXPR dim "${experts} * 512")
  foreach(sequence
          q4nx_gemv_k2048_n${rows}_resnorm_swiglu_scaled_rounds${experts}_t2
          q4nx_gemv_k${dim}_n2048_act_store_k${experts}_t2)
    list(APPEND q4nx_gemv_OUTPUTS ${sequence}.insts.elf ${sequence}.insts.bin)
  endforeach()
endforeach()
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
