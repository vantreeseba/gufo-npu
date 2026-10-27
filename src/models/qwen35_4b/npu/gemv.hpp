#pragma once

#include <cstddef>
#include <cstdint>

#include "npu/gemv.hpp"

// Qwen3.5-4B's decode GEMV (src/models/qwen35_4b/kernels): its own NPU
// program, built with the 4B's geometry. These constants must match
// kernels/variants.py.
namespace gufo_npu::qwen35_4b::npu {

// The kernel subdirectory (under the kernel directory) holding its xclbin and
// instruction sequences.
inline constexpr char kKernelSubdir[] = "qwen35_4b";
inline constexpr char kDecodeName[] = "qwen35_4b_decode";
// The prefill GEMM program (kernels/prefill_gemm.py).
inline constexpr char kPrefillName[] = "qwen35_4b_prefill";
// Residual width of the resnorm input stage.
inline constexpr std::size_t kHidden = 2560;
// The longest input, in values (FFN down).
inline constexpr std::size_t kMaxDim = 9216;
inline constexpr std::size_t kTokens = gufo_npu::npu::kGemvTokens;
// The input region: kTokens slots of kSlot fp32, then a kHidden bf16 norm
// weight at kNormOffset (same layout as the 2B's, wider).
inline constexpr std::size_t kSlot = 2 * kHidden;  // fp32
inline constexpr std::size_t kNormOffset = kTokens * kSlot * sizeof(float);
inline constexpr std::size_t kInputBytes =
    kNormOffset + kHidden * sizeof(std::uint16_t);
// The residual buffer: token t's kHidden fp32 at t * kSlot.
inline constexpr std::size_t kResidualBytes = (kSlot + kHidden) * sizeof(float);

inline gufo_npu::npu::GemvDesign DecodeDesign() {
  return {kDecodeName, kHidden, kMaxDim};
}

static_assert(kSlot == 5120 && kInputBytes == 46080,
              "the core program's token buffer (decode_gemv.py X_FLOATS)");

}  // namespace gufo_npu::qwen35_4b::npu
