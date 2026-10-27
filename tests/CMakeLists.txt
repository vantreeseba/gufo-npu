# Tests labelled "npu" need /dev/accel/accel0; everything else runs on hosted
# CI. NPU tests exit 77 (skip) when the device is missing, which is not a pass.

add_executable(diagnose_test core/diagnose_test.cpp)
target_link_libraries(diagnose_test PRIVATE gufo_npu_core)
add_test(NAME diagnose_test COMMAND diagnose_test)

add_test(NAME cli_version COMMAND gufo-npu version)
set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION
                     "^gufo-npu ${GUFO_NPU_VERSION}")

if(GUFO_NPU_ENABLE_XRT AND GUFO_NPU_BUILD_KERNELS)
  add_test(NAME npu_smoke
           COMMAND gufo-npu smoke --iters 200
                   --kernels "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(npu_smoke PROPERTIES LABELS npu)
endif()
