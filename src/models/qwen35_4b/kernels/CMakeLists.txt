# Qwen3.5-4B's NPU programs, compiled from this package's own design copies
# (no NPU needed) into ${CMAKE_BINARY_DIR}/kernels/qwen35_4b and installed
# under <kernels>/qwen35_4b.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(QWEN35_4B_KERNEL_OUT "${CMAKE_BINARY_DIR}/kernels/qwen35_4b")

# Decode GEMV: the outputs come from variants.py (pure data).
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/variants.py"
          --outputs decode
  OUTPUT_VARIABLE qwen35_4b_decode_files
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_SOURCE_DIR}/variants.py")
set(qwen35_4b_decode_outputs)
foreach(file IN LISTS qwen35_4b_decode_files)
  list(APPEND qwen35_4b_decode_outputs "${QWEN35_4B_KERNEL_OUT}/${file}")
endforeach()
add_custom_command(
  OUTPUT ${qwen35_4b_decode_outputs}
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/decode_gemv.py"
          --dev "${GUFO_NPU_KERNEL_DEVICE}" --out "${QWEN35_4B_KERNEL_OUT}"
  DEPENDS decode_gemv.py variants.py aie/decode_gemv.cc
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMENT "Compiling NPU kernel qwen35_4b_decode"
  VERBATIM)

# Prefill GEMM.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/variants.py"
          --outputs prefill
  OUTPUT_VARIABLE qwen35_4b_prefill_files
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
set(qwen35_4b_prefill_outputs)
foreach(file IN LISTS qwen35_4b_prefill_files)
  list(APPEND qwen35_4b_prefill_outputs "${QWEN35_4B_KERNEL_OUT}/${file}")
endforeach()
add_custom_command(
  OUTPUT ${qwen35_4b_prefill_outputs}
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prefill_gemm.py"
          --dev "${GUFO_NPU_KERNEL_DEVICE}" --out "${QWEN35_4B_KERNEL_OUT}"
  DEPENDS prefill_gemm.py variants.py
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMENT "Compiling NPU kernel qwen35_4b_prefill"
  VERBATIM)

add_custom_target(qwen35_4b_kernels ALL
                  DEPENDS ${qwen35_4b_decode_outputs} ${qwen35_4b_prefill_outputs})
install(FILES ${qwen35_4b_decode_outputs} ${qwen35_4b_prefill_outputs}
        DESTINATION "${GUFO_NPU_KERNEL_INSTALL_DIR}/qwen35_4b")
