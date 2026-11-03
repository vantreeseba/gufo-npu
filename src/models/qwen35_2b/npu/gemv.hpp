#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "npu/gemv.hpp"

// Qwen3.5-2B's decode GEMV (src/models/qwen35_2b/kernels): its own NPU
// program. These constants must match kernels/decode_gemv_variants.py.
namespace gufo_npu::qwen35_2b::npu {

// The decode GEMV program (kernels/decode_gemv.py): its xclbin and the
// prefix of its instruction sequences, in the kernel directory.
inline constexpr char kDecodeName[] = "qwen35_2b_decode";
// The prefill GEMM program (kernels/prefill_gemm.py).
inline constexpr char kPrefillName[] = "qwen35_2b_prefill";
// Residual width of the resnorm input stage.
inline constexpr std::size_t kHidden = 2048;
// The token buffer's per-token activation width, in values (MAX_DIM: the
// FFN down input is 6144; the program keeps 8704).
inline constexpr std::size_t kMaxDim = 8704;
inline constexpr std::size_t kTokens = gufo_npu::npu::kGemvTokens;
// The input region: kTokens slots of kSlot fp32, then a kHidden bf16 norm
// weight at kNormOffset.
inline constexpr std::size_t kSlot = 2 * kHidden;  // fp32
inline constexpr std::size_t kNormOffset = kTokens * kSlot * sizeof(float);
inline constexpr std::size_t kInputBytes =
    kNormOffset + kHidden * sizeof(std::uint16_t);
// The residual buffer: token t's kHidden fp32 at t * kSlot.
inline constexpr std::size_t kResidualBytes = (kSlot + kHidden) * sizeof(float);

inline gufo_npu::npu::GemvDesign DecodeDesign() {
  return {kDecodeName, kHidden, kMaxDim};
}

static_assert(kSlot == 4096 && kInputBytes == 36864,
              "the core program's token buffer (decode_gemv.py X_FLOATS)");

// Chains (decode_gemv_variants.py CHAIN_NEXT): one instruction sequence runs
// a layer's out, gate + up and down and the call after them, on regions of
// one activation buffer laid out in blocks, bytes throughout:
//   block: in_x | ffn_x | in_y | out_x | ffn_y
//   head:  x | r | y      (an LM head's input, residual and logits)
// A chain's window starts at its block's ffn_x and runs into the next block
// or head. Arguments: the window, the layer's out, gate + up and down
// matrices back to back (in that order), the next call's matrix, scratch.
inline constexpr std::size_t kInYBytes = kTokens * 8192 * sizeof(float);
inline constexpr std::size_t kFfnYBytes = 8 * (kMaxDim / 2) * sizeof(float);
inline constexpr std::size_t kBlockInX = 0;
inline constexpr std::size_t kBlockFfnX = kInputBytes;
inline constexpr std::size_t kBlockInY = 2 * kInputBytes;
inline constexpr std::size_t kBlockOutX = 2 * kInputBytes + kInYBytes;
inline constexpr std::size_t kBlockFfnY = 3 * kInputBytes + kInYBytes;
inline constexpr std::size_t kBlockBytes =
    3 * kInputBytes + kInYBytes + kFfnYBytes;
// A head's regions, at a block's in_x, ffn_x and in_y.
inline constexpr std::size_t kHeadX = kBlockInX;
inline constexpr std::size_t kHeadR = kBlockFfnX;
inline constexpr std::size_t kHeadY = kBlockInY;

// The call after a chain's FFN.
enum class ChainNext { kDeltaNet, kAttention, kLmHead, kDraftHead, kNone };

inline const char* ChainNextName(ChainNext next) {
  switch (next) {
    case ChainNext::kDeltaNet:
      return "deltanet";
    case ChainNext::kAttention:
      return "attention";
    case ChainNext::kLmHead:
      return "lm";
    case ChainNext::kDraftHead:
      return "draft";
    case ChainNext::kNone:
      return "bare";
  }
  return "";
}

// Rows of the next call (a resnorm/store GEMV on the hidden vector), 0 for
// none.
inline std::size_t ChainNextRows(ChainNext next) {
  switch (next) {
    case ChainNext::kDeltaNet:
      return 8192;
    case ChainNext::kAttention:
      return 5120;
    case ChainNext::kLmHead:
      return 248832;
    case ChainNext::kDraftHead:
      return 32768;
    case ChainNext::kNone:
      return 0;
  }
  return 0;
}

// The chain's sequence name (decode_gemv_variants.py chain_name).
inline std::string ChainSequence(ChainNext next, std::size_t tokens) {
  return std::string(kDecodeName) + "_chain_" + ChainNextName(next) + "_t" +
         std::to_string(tokens);
}

// The window a chain reads and writes, from its block's ffn_x
// (decode_gemv_variants.py chain_window_bytes).
inline constexpr std::size_t kWindowNext = kBlockBytes - kBlockFfnX;
inline std::size_t ChainWindowBytes(ChainNext next) {
  const std::size_t rows = ChainNextRows(next);
  if (rows == 0) {
    return kWindowNext + kInputBytes;
  }
  return kWindowNext + kHeadY + kTokens * rows * sizeof(float);
}

static_assert(kBlockBytes == 315392 && kFfnYBytes == 139264,
              "decode_gemv_variants.py BLOCK_BYTES and FFN_Y_BYTES");

}  // namespace gufo_npu::qwen35_2b::npu
