# The server's model side (src/serve).

add_executable(serve_model_registry_test model_registry_test.cpp)
target_link_libraries(serve_model_registry_test PRIVATE gufo_npu_serve_engine)
add_test(NAME serve_model_registry_test COMMAND serve_model_registry_test)
set_tests_properties(serve_model_registry_test PROPERTIES LABELS serve)

if(GUFO_NPU_ENABLE_XRT AND GUFO_NPU_BUILD_KERNELS)
  add_executable(serve_engine_test engine_test.cpp)
  target_link_libraries(serve_engine_test PRIVATE gufo_npu_serve_engine)
  add_test(NAME serve_engine_test
           COMMAND serve_engine_test "${CMAKE_BINARY_DIR}/kernels")
  set_tests_properties(serve_engine_test PROPERTIES
                       LABELS "serve;qwen35_2b;npu;model" SKIP_RETURN_CODE 77)
endif()
