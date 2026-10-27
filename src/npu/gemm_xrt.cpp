#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <vector>

#include "core/bfp16.hpp"
#include "npu/gemm.hpp"
#include "npu/xrt_kernel.hpp"

namespace gufo_npu::npu {

std::string Bfp16GemmSequence(std::size_t m, std::size_t k, std::size_t n,
                              const std::string& design) {
  return design + "_m" + std::to_string(m) + "_k" + std::to_string(k) + "_n" +
         std::to_string(n);
}

struct GemmRuntime::Impl {
  std::filesystem::path dir;
  std::string design;
  HwContext hw;
  std::map<std::string, std::unique_ptr<Sequence>> sequences;

  Impl(const std::filesystem::path& d, const std::string& name)
      : dir(d), design(name), hw(d, name) {}

  Sequence& Get(const std::string& name) {
    auto& slot = sequences[name];
    if (!slot) {
      if (!std::filesystem::exists(dir / (name + ".insts.elf"))) {
        throw std::runtime_error("no bfp16 GEMM sequence " + name + " in " +
                                 dir.string());
      }
      slot = std::make_unique<Sequence>(hw, dir, name);
    }
    return *slot;
  }
};

GemmRuntime::GemmRuntime(const std::filesystem::path& kernel_dir,
                         const std::string& design)
    : impl_(std::make_unique<Impl>(kernel_dir, design)) {}

GemmRuntime::~GemmRuntime() = default;

struct GemmBuffer::Impl {
  xrt::bo bo;
};

GemmBuffer::GemmBuffer(GemmRuntime& runtime, std::size_t bytes)
    : impl_(std::make_unique<Impl>(
          Impl{runtime.impl().hw.Buffer(std::max<std::size_t>(bytes, 64))})),
      bytes_(bytes) {
  // Zero it and push the zeros out of the CPU cache, so no dirty line is
  // written back over what the NPU writes later.
  std::memset(data(), 0, bytes);
  ToDevice();
}

GemmBuffer::~GemmBuffer() = default;

void* GemmBuffer::data() {
  return impl_->bo.map<void*>();
}

void GemmBuffer::ToDevice(std::size_t bytes) {
  impl_->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes == 0 ? bytes_ : bytes, 0);
}

void GemmBuffer::FromDevice(std::size_t bytes) {
  impl_->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes == 0 ? bytes_ : bytes, 0);
}

struct Bfp16Gemm::Impl {
  std::map<std::size_t, Sequence*> sequences;  // by m
  std::size_t w_bytes;
  xrt::bo w;
  std::optional<xrt::run> run;

  Impl(GemmRuntime::Impl& rt, const std::vector<std::uint8_t>& packed,
       std::size_t dim, std::size_t rows)
      : w_bytes(packed.size()), w(rt.hw.Buffer(w_bytes)) {
    for (const std::size_t m : kGemmChunks) {
      sequences[m] = &rt.Get(Bfp16GemmSequence(m, dim, rows, rt.design));
    }
    std::memcpy(w.map<void*>(), packed.data(), w_bytes);
    w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  xrt::run Bind(GemmBuffer& a, GemmBuffer& c, std::size_t m) {
    const auto found = sequences.find(m);
    if (found == sequences.end()) {
      throw std::invalid_argument("bfp16 GEMM: no sequence for " +
                                  std::to_string(m) + " rows");
    }
    return found->second->Run({&a.impl().bo, &w, &c.impl().bo});
  }
};

Bfp16Gemm::Bfp16Gemm(GemmRuntime& runtime, const float* w, std::size_t rows,
                     std::size_t dim)
    : dim_(dim), rows_(rows) {
  impl_ = std::make_unique<Impl>(runtime.impl(), bfp16::PackGemmB(w, rows, dim),
                                 dim, rows);
}

Bfp16Gemm::~Bfp16Gemm() = default;

std::size_t Bfp16Gemm::weight_bytes() const {
  return impl_->w_bytes;
}

void Bfp16Gemm::Run(GemmBuffer& a, GemmBuffer& c, std::size_t m) {
  if (a.size() < a_bytes(m) || c.size() < c_bytes(m)) {
    throw std::invalid_argument("bfp16 GEMM: operand buffer too small");
  }
  impl_->run.emplace(impl_->Bind(a, c, m));
  impl_->run->start();
  impl_->run->wait();
}

struct GemmCall::Impl {
  xrt::run run;
};

GemmCall::GemmCall() = default;
GemmCall::GemmCall(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GemmCall::~GemmCall() {
  Wait();
}
GemmCall::GemmCall(GemmCall&&) noexcept = default;
GemmCall& GemmCall::operator=(GemmCall&& other) noexcept {
  if (this != &other) {
    Wait();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

void GemmCall::Wait() {
  if (impl_) {
    impl_->run.wait();
    impl_.reset();
  }
}

GemmCall Bfp16Gemm::Start(GemmBuffer& a, GemmBuffer& c, std::size_t m) {
  if (a.size() < a_bytes(m) || c.size() < c_bytes(m)) {
    throw std::invalid_argument("bfp16 GEMM: operand buffer too small");
  }
  auto call =
      std::make_unique<GemmCall::Impl>(GemmCall::Impl{impl_->Bind(a, c, m)});
  call->run.start();
  return GemmCall(std::move(call));
}

Bfp16Gemm::Timing Bfp16Gemm::Time(int iterations) {
  if (!impl_->run) {
    throw std::logic_error("bfp16 GEMM: Time() before Run()");
  }
  iterations = std::max(iterations, 1);
  Timing t;
  t.best_us = std::numeric_limits<double>::max();
  double total = 0;
  for (int i = 0; i < iterations; ++i) {
    const auto start = Clock::now();
    impl_->run->start();
    impl_->run->wait();
    const double us = MicrosSince(start);
    total += us;
    t.best_us = std::min(t.best_us, us);
  }
  t.mean_us = total / iterations;
  return t;
}

}  // namespace gufo_npu::npu
