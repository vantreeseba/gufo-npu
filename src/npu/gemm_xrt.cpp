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

std::string Bfp16GemmSequence(std::size_t m, std::size_t k, std::size_t n) {
  return "bfp16_gemm_m" + std::to_string(m) + "_k" + std::to_string(k) +
         "_n" + std::to_string(n);
}

struct GemmRuntime::Impl {
  std::filesystem::path dir;
  HwContext hw;
  std::map<std::string, std::unique_ptr<Sequence>> sequences;

  explicit Impl(const std::filesystem::path& d) : dir(d), hw(d, "bfp16_gemm") {}

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

GemmRuntime::GemmRuntime(const std::filesystem::path& kernel_dir)
    : impl_(std::make_unique<Impl>(kernel_dir)) {}

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
  Sequence* sequence;
  std::size_t w_bytes;
  xrt::bo w;
  std::optional<xrt::run> run;

  Impl(GemmRuntime::Impl& rt, const std::vector<std::uint8_t>& packed,
       std::size_t m, std::size_t dim, std::size_t rows)
      : sequence(&rt.Get(Bfp16GemmSequence(m, dim, rows))),
        w_bytes(packed.size()),
        w(rt.hw.Buffer(w_bytes)) {
    std::memcpy(w.map<void*>(), packed.data(), w_bytes);
    w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
};

Bfp16Gemm::Bfp16Gemm(GemmRuntime& runtime, const float* w, std::size_t rows,
                     std::size_t dim, std::size_t m)
    : m_(m), dim_(dim), rows_(rows) {
  impl_ = std::make_unique<Impl>(runtime.impl(),
                                 bfp16::PackGemmB(w, rows, dim), m, dim, rows);
}

Bfp16Gemm::~Bfp16Gemm() = default;

std::size_t Bfp16Gemm::weight_bytes() const {
  return impl_->w_bytes;
}

void Bfp16Gemm::Run(GemmBuffer& a, GemmBuffer& c) {
  if (a.size() < a_bytes() || c.size() < c_bytes()) {
    throw std::invalid_argument("bfp16 GEMM: operand buffer too small");
  }
  impl_->run.emplace(
      impl_->sequence->Run({&a.impl().bo, &impl_->w, &c.impl().bo}));
  impl_->run->start();
  impl_->run->wait();
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
