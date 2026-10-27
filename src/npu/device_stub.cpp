#include "npu/device.hpp"

namespace gufo_npu::npu {

bool RuntimeAvailable() {
  return false;
}

std::vector<diagnose::Check> RuntimeChecks() {
  return {{"XRT", diagnose::Status::kFail,
           "this build has no XRT support (GUFO_NPU_ENABLE_XRT=OFF)"}};
}

SmokeResult RunSmoke(const std::filesystem::path& /*kernel_dir*/,
                     int /*iterations*/) {
  return {.error = "this build has no XRT support"};
}

DispatchBenchResult BenchDispatch(const std::filesystem::path& /*kernel_dir*/,
                                  int /*rounds*/,
                                  const std::vector<int>& /*depths*/) {
  return {.error = "this build has no XRT support"};
}

}  // namespace gufo_npu::npu
