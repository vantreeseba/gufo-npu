#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <vector>

#include "core/quant.hpp"
#include "npu/gemv.hpp"
#include "npu/xrt_kernel.hpp"

namespace gufo_npu::npu {

std::string Q4nxGemvSequence(std::size_t dim, std::size_t padded_rows) {
  return "q4nx_gemv_k" + std::to_string(dim) + "_n" +
         std::to_string(padded_rows);
}

struct GemvRuntime::Impl {
  std::filesystem::path dir;
  HwContext hw;
  std::map<std::string, std::unique_ptr<Sequence>> sequences;

  explicit Impl(const std::filesystem::path& d) : dir(d), hw(d, "q4nx_gemv") {}

  Sequence& Get(const std::string& name) {
    auto& slot = sequences[name];
    if (!slot) {
      if (!std::filesystem::exists(dir / (name + ".insts.elf"))) {
        throw std::runtime_error("no q4nx GEMV sequence " + name + " in " +
                                 dir.string());
      }
      slot = std::make_unique<Sequence>(hw, dir, name);
    }
    return *slot;
  }
};

GemvRuntime::GemvRuntime(const std::filesystem::path& kernel_dir)
    : impl_(std::make_unique<Impl>(kernel_dir)) {}

GemvRuntime::~GemvRuntime() = default;

struct Q4nxGemv::Impl {
  std::size_t y_bytes;
  std::size_t w_bytes;
  xrt::bo x;
  xrt::bo w;
  xrt::bo y;
  std::optional<xrt::run> run;

  Impl(GemvRuntime::Impl& rt, const q4nx::Matrix& m)
      : y_bytes(m.padded_rows * sizeof(float)),
        w_bytes(m.bytes.size()),
        x(rt.hw.Buffer(kGemvMaxDim * sizeof(std::uint16_t))),
        w(rt.hw.Buffer(w_bytes)),
        y(rt.hw.Buffer(y_bytes)) {
    auto& sequence = rt.Get(Q4nxGemvSequence(m.dim, m.padded_rows));
    std::memset(x.map<void*>(), 0, kGemvMaxDim * sizeof(std::uint16_t));
    std::memcpy(w.map<void*>(), m.bytes.data(), w_bytes);
    w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    run.emplace(sequence.Run({&x, &w, &y}));
  }
};

Q4nxGemv::Q4nxGemv(GemvRuntime& runtime, const q4nx::Matrix& weights)
    : impl_(std::make_unique<Impl>(runtime.impl(), weights)),
      dim_(weights.dim),
      rows_(weights.rows) {
  if (weights.dim > kGemvMaxDim) {
    throw std::invalid_argument("q4nx GEMV: dim above the design's maximum");
  }
}

Q4nxGemv::~Q4nxGemv() = default;

void Q4nxGemv::Run(const std::uint16_t* x, float* y) {
  std::memcpy(impl_->x.map<void*>(), x, dim_ * sizeof(std::uint16_t));
  impl_->x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  impl_->run->start();
  impl_->run->wait();
  impl_->y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::memcpy(y, impl_->y.map<const void*>(), impl_->y_bytes);
}

void Q4nxGemv::Run(const float* x, float* y) {
  auto* xb = impl_->x.map<std::uint16_t*>();
  for (std::size_t i = 0; i < dim_; ++i) {
    xb[i] = quant::FloatToBf16(x[i]);
  }
  impl_->x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  impl_->run->start();
  impl_->run->wait();
  impl_->y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::memcpy(y, impl_->y.map<const void*>(), rows_ * sizeof(float));
}

Q4nxGemv::Timing Q4nxGemv::Time(int iterations) {
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

std::size_t Q4nxGemv::weight_bytes() const {
  return impl_->w_bytes;
}

}  // namespace gufo_npu::npu
