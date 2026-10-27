#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "core/q4nx.hpp"

// The q4nx GEMV design (kernels/designs/q4nx_gemv.py): one xclbin in one
// hardware context, one instruction sequence per compiled shape. Each
// Q4nxGemv keeps one matrix resident in a device buffer and multiplies one
// token per call.
namespace gufo_npu::npu {

// The longest token the design accepts; shorter ones are zero-padded.
inline constexpr std::size_t kGemvMaxDim = 6144;

// The instruction sequence name for a shape, e.g. "q4nx_gemv_k2048_n8192".
std::string Q4nxGemvSequence(std::size_t dim, std::size_t padded_rows);

class GemvRuntime {
public:
  // Loads q4nx_gemv.xclbin from `kernel_dir`. Throws when it is missing or
  // the NPU cannot be opened.
  explicit GemvRuntime(const std::filesystem::path& kernel_dir);
  ~GemvRuntime();
  GemvRuntime(const GemvRuntime&) = delete;
  GemvRuntime& operator=(const GemvRuntime&) = delete;

  struct Impl;
  Impl& impl() { return *impl_; }

private:
  std::unique_ptr<Impl> impl_;
};

class Q4nxGemv {
public:
  // Uploads `weights`. Throws when no sequence was compiled for its shape.
  Q4nxGemv(GemvRuntime& runtime, const q4nx::Matrix& weights);
  ~Q4nxGemv();
  Q4nxGemv(const Q4nxGemv&) = delete;
  Q4nxGemv& operator=(const Q4nxGemv&) = delete;

  // x: dim bf16 bits. y: padded_rows fp32 values.
  void Run(const std::uint16_t* x, float* y);
  // x as fp32 (rounded to bf16 on the way in); y gets `rows` values.
  void Run(const float* x, float* y);

  // Mean and best wall time of start + wait, in microseconds, over
  // `iterations` warm dispatches on the token from the last Run.
  struct Timing {
    double mean_us = 0;
    double best_us = 0;
  };
  Timing Time(int iterations);

  std::size_t dim() const { return dim_; }
  std::size_t rows() const { return rows_; }
  std::size_t weight_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::size_t dim_ = 0;
  std::size_t rows_ = 0;
};

}  // namespace gufo_npu::npu
