#include <stdexcept>

#include "npu/gemv.hpp"

namespace gufo_npu::npu {

namespace {
[[noreturn]] void NoXrt() {
  throw std::runtime_error("gufo-npu was built without XRT");
}
}  // namespace

std::string Q4nxGemvSequence(std::size_t dim, std::size_t padded_rows, Prologue,
                             Epilogue, std::size_t tokens,
                             const std::string& design) {
  return design + "_k" + std::to_string(dim) + "_n" +
         std::to_string(padded_rows) + "_t" + std::to_string(tokens);
}

struct GemvRuntime::Impl {};
GemvRuntime::GemvRuntime(const std::filesystem::path&, Flow, GemvDesign design)
    : design_(std::move(design)) {
  NoXrt();
}
GemvRuntime::~GemvRuntime() = default;
Flow GemvRuntime::flow() const {
  return Flow::kElf;
}

struct Buffer::Impl {};
Buffer::Buffer(GemvRuntime&, std::size_t) {
  NoXrt();
}
Buffer::Buffer(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
Buffer::~Buffer() = default;
Buffer::Buffer(const Buffer&) = default;
Buffer& Buffer::operator=(const Buffer&) = default;
Buffer Buffer::Sub(std::size_t, std::size_t) const {
  NoXrt();
}
void* Buffer::data() {
  NoXrt();
}
std::size_t Buffer::size() const {
  return 0;
}
void Buffer::ToDevice() {}
void Buffer::FromDevice() {}

struct Q4nxGemv::Impl {};
Q4nxGemv::Q4nxGemv(GemvRuntime&, const q4nx::Matrix&, Prologue, Epilogue) {
  NoXrt();
}
Q4nxGemv::~Q4nxGemv() = default;
void Q4nxGemv::Run(Buffer&, Buffer&, Buffer&, std::size_t) {}
void Q4nxGemv::Run(const float*, float*) {}
void Q4nxGemv::Run(const std::uint16_t*, float*) {}
Q4nxGemv::Timing Q4nxGemv::Time(int) {
  return {};
}
std::size_t Q4nxGemv::weight_bytes() const {
  return 0;
}
std::size_t Q4nxGemv::y_bytes() const {
  return 0;
}

struct Q4nxGather::Impl {};
Q4nxGather::Q4nxGather(GemvRuntime&, Kind, const ExpertArena&, std::size_t,
                       std::size_t) {
  NoXrt();
}
Q4nxGather::~Q4nxGather() = default;
void Q4nxGather::Select(const int*, const float*) {}
std::size_t Q4nxGather::y_bytes() const {
  return 0;
}
std::size_t Q4nxGather::experts() const {
  return 0;
}
std::size_t Q4nxGather::tokens() const {
  return 0;
}

struct Chain::Impl {};
Chain::Chain(GemvRuntime&) {
  NoXrt();
}
Chain::~Chain() = default;
void Chain::Add(Q4nxGemv&, Buffer&, Buffer&, Buffer&, std::size_t) {}
void Chain::Add(Q4nxGather&, Buffer&, Buffer&, Buffer&, Buffer&) {}
void Chain::Run() {}

}  // namespace gufo_npu::npu
