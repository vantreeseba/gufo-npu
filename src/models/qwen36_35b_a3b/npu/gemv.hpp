#pragma once

#include <cstddef>

#include "npu/gemv.hpp"

// Qwen3.6-35B-A3B's decode GEMV and expert gathers
// (src/models/qwen36_35b_a3b/kernels): its own NPU program. These constants
// must match kernels/decode_gemv_variants.py.
namespace gufo_npu::qwen36_35b_a3b::npu {

// The decode GEMV program (kernels/decode_gemv.py): its xclbin and the
// prefix of its instruction sequences, in the kernel directory.
inline constexpr char kDecodeName[] = "qwen36_35b_a3b_decode";
// Residual width of the resnorm input stage.
inline constexpr std::size_t kHidden = gufo_npu::npu::kGemvHidden;
// The longest input, in values (a down gather over 17 experts).
inline constexpr std::size_t kMaxDim = gufo_npu::npu::kGemvMaxDim;

inline gufo_npu::npu::GemvDesign DecodeDesign() {
  return {kDecodeName, kHidden, kMaxDim};
}

static_assert(kHidden == 2048 && kMaxDim == 8704,
              "decode_gemv_variants.py HIDDEN and MAX_DIM");

}  // namespace gufo_npu::qwen36_35b_a3b::npu
