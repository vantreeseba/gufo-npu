#include <stdexcept>

#include "npu/gemm.hpp"

namespace gufo_npu::npu {

namespace {
[[noreturn]] void NoXrt() {
  throw std::runtime_error("gufo-npu was built without XRT");
}
}  // namespace

std::string Bfp16GemmSequence(std::size_t m, std::size_t k, std::size_t n,
                              const std::string& design) {
  return design + "_m" + std::to_string(m) + "_k" + std::to_string(k) + "_n" +
         std::to_string(n);
}

struct GemmRuntime::Impl {};
GemmRuntime::GemmRuntime(const std::filesystem::path&, const std::string&) {
  NoXrt();
}
GemmRuntime::~GemmRuntime() = default;

struct GemmBuffer::Impl {};
GemmBuffer::GemmBuffer(GemmRuntime&, std::size_t) {
  NoXrt();
}
GemmBuffer::~GemmBuffer() = default;
void* GemmBuffer::data() {
  NoXrt();
}
void GemmBuffer::ToDevice(std::size_t) {
  NoXrt();
}
void GemmBuffer::FromDevice(std::size_t) {
  NoXrt();
}

struct Bfp16Gemm::Impl {};
Bfp16Gemm::Bfp16Gemm(GemmRuntime&, const float*, std::size_t, std::size_t) {
  NoXrt();
}
Bfp16Gemm::~Bfp16Gemm() = default;
std::size_t Bfp16Gemm::weight_bytes() const {
  return 0;
}
void Bfp16Gemm::Run(GemmBuffer&, GemmBuffer&, std::size_t) {
  NoXrt();
}

struct GemmCall::Impl {};
GemmCall::GemmCall() = default;
GemmCall::GemmCall(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GemmCall::~GemmCall() = default;
GemmCall::GemmCall(GemmCall&&) noexcept = default;
GemmCall& GemmCall::operator=(GemmCall&&) noexcept = default;
void GemmCall::Wait() {}
GemmCall Bfp16Gemm::Start(GemmBuffer&, GemmBuffer&, std::size_t) {
  NoXrt();
}
Bfp16Gemm::Timing Bfp16Gemm::Time(int) {
  NoXrt();
}

}  // namespace gufo_npu::npu
