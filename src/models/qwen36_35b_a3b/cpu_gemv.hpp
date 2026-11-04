#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/thread_pool.hpp"

// y = W x on the CPU for 1 or 2 tokens from a q4nx matrix in host memory
// (read in place, e.g. the mapped NPU weight buffer), shaped like the NPU
// core's GEMV (kernels/aie/decode_gemv.cc) rather than a dequantize-then-dot:
//   - x rounded to bf16 (nearest even), as the core's input stage does;
//   - per row and 32 inputs, the integer codes times x summed in fp32
//     (codes are exact in bf16: AVX-512 BF16 dot products), then times the
//     group's sub-scale sc, and the group's sum of x times its sub-minimum
//     mn;
//   - per 256-input block the row's d and dmin multiply those two totals.
// The core splits its fp32 partial sums into bf16 halves (hi + lo) before
// each scale multiply; here they stay fp32, so the two agree to about 1e-5
// relative, not bit for bit.
//
// The fast (AVX-512 BF16, VBMI) and portable paths run the same operations
// in the same order and give the same bits for finite, normal inputs (the
// BF16 dot product flushes denormals).
namespace gufo_npu::qwen36_35b_a3b {

class CpuGemv {
public:
  static constexpr int kMaxTokens = 2;

  // `q4nx`: the matrix's packed bytes (q4nx::Matrix layout, padded rows a
  // multiple of 32, `dim` a multiple of 256), which must outlive this.
  // `rows`: outputs written. `portable` forces the portable path (tests);
  // otherwise AVX-512 BF16 runs when the CPU has it.
  CpuGemv(SpinPool& pool, const std::uint8_t* q4nx, std::size_t rows,
          std::size_t dim, bool portable = false);

  // x: token t at x + t * x_stride (fp32, dim values); y: token t at
  // y + t * y_stride (fp32, rows values). 1 or 2 tokens.
  void Run(const float* x, std::size_t x_stride, float* y, std::size_t y_stride,
           int tokens);

  // Whether the AVX-512 BF16 path runs.
  bool fast() const { return fast_; }

private:
  SpinPool& pool_;
  const std::uint8_t* matrix_;
  std::size_t rows_;
  std::size_t dim_;
  std::size_t k_blocks_;
  std::size_t tiles_;
  bool fast_;
  // Per token: x as bf16, and the fp32 sum of each 32 of them.
  std::vector<std::uint16_t> xb_;
  std::vector<float> sums_;
};

}  // namespace gufo_npu::qwen36_35b_a3b
