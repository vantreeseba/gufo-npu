#include <stdexcept>

#include "npu/gemv.hpp"

namespace gufo_npu::npu {

std::string Q4nxGemvSequence(std::size_t dim, std::size_t padded_rows) {
  return "q4nx_gemv_k" + std::to_string(dim) + "_n" +
         std::to_string(padded_rows);
}

struct GemvRuntime::Impl {};

GemvRuntime::GemvRuntime(const std::filesystem::path&) {
  throw std::runtime_error("gufo-npu was built without XRT");
}

GemvRuntime::~GemvRuntime() = default;

struct Q4nxGemv::Impl {};

Q4nxGemv::Q4nxGemv(GemvRuntime&, const q4nx::Matrix&) {
  throw std::runtime_error("gufo-npu was built without XRT");
}

Q4nxGemv::~Q4nxGemv() = default;

void Q4nxGemv::Run(const std::uint16_t*, float*) {}
void Q4nxGemv::Run(const float*, float*) {}

Q4nxGemv::Timing Q4nxGemv::Time(int) {
  return {};
}

std::size_t Q4nxGemv::weight_bytes() const {
  return 0;
}

}  // namespace gufo_npu::npu
