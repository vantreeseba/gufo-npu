#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "core/q4nx.hpp"

// The q4nx GEMV design (kernels/designs/q4nx_gemv.py) with its weights
// resident in a device buffer: one call multiplies one token.
namespace gufo_npu::npu {

// The compiled artifact name for a shape, e.g. "q4nx_gemv_k2048_n6144".
std::string Q4nxGemvDesign(std::size_t dim, std::size_t padded_rows, bool norm);

class Q4nxGemv {
public:
  // With `norm`, each token arrives followed by its RMS norm weight and the
  // kernel normalizes it first (eps 1e-6). Throws when the design is missing
  // or the NPU cannot be opened.
  Q4nxGemv(const std::filesystem::path& kernel_dir, const q4nx::Matrix& weights,
           bool norm);
  ~Q4nxGemv();
  Q4nxGemv(const Q4nxGemv&) = delete;
  Q4nxGemv& operator=(const Q4nxGemv&) = delete;

  // x: dim bf16 bits (2 * dim with norm). y: padded_rows bf16 bits.
  void Run(const std::uint16_t* x, std::uint16_t* y);

  // Mean and best wall time of start + wait, in microseconds, over
  // `iterations` warm dispatches on the token from the last Run.
  struct Timing {
    double mean_us = 0;
    double best_us = 0;
  };
  Timing Time(int iterations);

  std::size_t weight_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::npu
