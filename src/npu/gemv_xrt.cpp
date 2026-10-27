#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>

#include "npu/gemv.hpp"
#include "npu/xrt_kernel.hpp"

namespace gufo_npu::npu {

std::string Q4nxGemvDesign(std::size_t dim, std::size_t padded_rows,
                           bool norm) {
  return "q4nx_gemv_k" + std::to_string(dim) + "_n" +
         std::to_string(padded_rows) + (norm ? "_norm" : "");
}

struct Q4nxGemv::Impl {
  LoadedKernel kernel;
  std::size_t x_bytes;
  std::size_t y_bytes;
  std::size_t w_bytes;
  xrt::bo x;
  xrt::bo w;
  xrt::bo y;
  std::optional<xrt::run> run;

  Impl(const std::filesystem::path& dir, const q4nx::Matrix& m, bool norm)
      : kernel(dir, Q4nxGemvDesign(m.dim, m.padded_rows, norm)),
        x_bytes((norm ? 2 : 1) * m.dim * sizeof(std::uint16_t)),
        y_bytes(m.padded_rows * sizeof(std::uint16_t)),
        w_bytes(m.bytes.size()),
        x(kernel.Buffer(x_bytes)),
        w(kernel.Buffer(w_bytes)),
        y(kernel.Buffer(y_bytes)) {
    std::memcpy(w.map<void*>(), m.bytes.data(), w_bytes);
    w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    run.emplace(kernel.Run({&x, &w, &y}));
  }
};

Q4nxGemv::Q4nxGemv(const std::filesystem::path& kernel_dir,
                   const q4nx::Matrix& weights, bool norm)
    : impl_(std::make_unique<Impl>(kernel_dir, weights, norm)) {}

Q4nxGemv::~Q4nxGemv() = default;

void Q4nxGemv::Run(const std::uint16_t* x, std::uint16_t* y) {
  std::memcpy(impl_->x.map<void*>(), x, impl_->x_bytes);
  impl_->x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  impl_->run->start();
  impl_->run->wait();
  impl_->y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::memcpy(y, impl_->y.map<const void*>(), impl_->y_bytes);
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
