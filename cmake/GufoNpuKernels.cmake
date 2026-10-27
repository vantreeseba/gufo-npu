# gufo_npu_add_kernel: compile one IRON / mlir-aie design to an xclbin plus
# instruction sequences at build time (kernels/build.py; needs no NPU).
#
#   gufo_npu_add_kernel(NAME <name> DESIGN <design.py>
#                       [SOURCES <core sources>...] MODEL <model>...)
#
# The outputs come from `build.py --list-outputs`, which reads the design's
# pure-data <design>_variants.py when it has one. MODEL names the models the
# kernel serves (qwen35_2b, qwen36_35b_a3b, ...), or `common` for
# diagnostics; with GUFO_NPU_MODELS set to a list of models, only kernels
# serving one of them (and `common` ones) are built. Outputs go to
# GUFO_NPU_KERNEL_OUT and are
# installed to GUFO_NPU_KERNEL_INSTALL_DIR; each kernel has a target
# gufo_npu_kernel_<name>, and gufo_npu_kernels builds them all.

set(GUFO_NPU_MODELS "all" CACHE STRING
    "Models whose NPU kernels to build (list), or all")

function(gufo_npu_add_kernel)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "NAME;DESIGN" "SOURCES;MODEL")
  if(NOT arg_NAME OR NOT arg_DESIGN OR NOT arg_MODEL)
    message(FATAL_ERROR "gufo_npu_add_kernel: NAME, DESIGN and MODEL are required")
  endif()
  if(NOT GUFO_NPU_MODELS STREQUAL "all" AND NOT "common" IN_LIST arg_MODEL)
    set(wanted FALSE)
    foreach(model IN LISTS arg_MODEL)
      if(model IN_LIST GUFO_NPU_MODELS)
        set(wanted TRUE)
      endif()
    endforeach()
    if(NOT wanted)
      return()
    endif()
  endif()

  cmake_path(ABSOLUTE_PATH arg_DESIGN BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
             OUTPUT_VARIABLE design)
  cmake_path(GET design STEM stem)
  cmake_path(REPLACE_FILENAME design "${stem}_variants.py" OUTPUT_VARIABLE variants)
  set(build_py "${GUFO_NPU_KERNEL_BUILD_PY}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${build_py}")
  if(EXISTS "${variants}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${variants}")
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${build_py}" --list-outputs --design "${design}"
    OUTPUT_VARIABLE listed
    RESULT_VARIABLE status
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT status EQUAL 0 OR listed STREQUAL "")
    message(FATAL_ERROR "gufo_npu_add_kernel(${arg_NAME}): build.py --list-outputs failed")
  endif()
  string(REPLACE "\n" ";" files "${listed}")

  set(outputs)
  foreach(file IN LISTS files)
    list(APPEND outputs "${GUFO_NPU_KERNEL_OUT}/${file}")
  endforeach()
  set(sources)
  foreach(source IN LISTS arg_SOURCES)
    cmake_path(ABSOLUTE_PATH source BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
               OUTPUT_VARIABLE source)
    list(APPEND sources "${source}")
  endforeach()
  set(depends "${build_py}" "${design}" ${sources})
  if(EXISTS "${variants}")
    list(APPEND depends "${variants}")
  endif()
  add_custom_command(
    OUTPUT ${outputs}
    COMMAND "${Python3_EXECUTABLE}" "${build_py}" --dev "${GUFO_NPU_KERNEL_DEVICE}"
            --out "${GUFO_NPU_KERNEL_OUT}" --design "${design}"
    DEPENDS ${depends}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Compiling NPU kernel ${arg_NAME} (${arg_MODEL})"
    VERBATIM)
  add_custom_target(gufo_npu_kernel_${arg_NAME} DEPENDS ${outputs})
  add_dependencies(gufo_npu_kernels gufo_npu_kernel_${arg_NAME})
  install(FILES ${outputs} DESTINATION "${GUFO_NPU_KERNEL_INSTALL_DIR}")
endfunction()
