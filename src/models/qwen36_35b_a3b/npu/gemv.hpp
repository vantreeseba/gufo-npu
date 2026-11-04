#pragma once

#include <cstddef>
#include <string>

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

// Chains (decode_gemv_variants.py CHAIN_NEXT): one instruction sequence runs
// a MoE layer's gate + up gather, down gather and the call after them, on
// regions of one activation buffer laid out in blocks, bytes throughout:
//   block: in_x | ffn_x | in_y | out_x | ffn_y | scratch
//   head:  x | r | y      (an LM head's input, residual and logits)
// A chain's window starts at its block's ffn_x and runs into the next block
// or head. Arguments: the window, the layer's expert arena twice (the gate +
// up gather's weight patches name the second argument, the down gather's the
// third), the next call's matrix. Instruction-buffer flow only.
inline constexpr std::size_t kTokens = gufo_npu::npu::kGemvTokens;
inline constexpr std::size_t kInputBytes = gufo_npu::npu::kGemvInputBytes;
inline constexpr std::size_t kInYBytes = kTokens * 12288 * sizeof(float);
inline constexpr std::size_t kFfnYBytes = 8 * (kMaxDim / 2) * sizeof(float);
// A residual argument nobody reads (calls that write none).
inline constexpr std::size_t kScratchBytes =
    gufo_npu::npu::kGemvResidualBytes;
inline constexpr std::size_t kBlockInX = 0;
inline constexpr std::size_t kBlockFfnX = kInputBytes;
inline constexpr std::size_t kBlockInY = 2 * kInputBytes;
inline constexpr std::size_t kBlockOutX = 2 * kInputBytes + kInYBytes;
inline constexpr std::size_t kBlockFfnY = 3 * kInputBytes + kInYBytes;
inline constexpr std::size_t kBlockScratch =
    3 * kInputBytes + kInYBytes + kFfnYBytes;
inline constexpr std::size_t kBlockBytes = kBlockScratch + kScratchBytes;
// A head's regions, at a block's in_x, ffn_x and in_y.
inline constexpr std::size_t kHeadX = kBlockInX;
inline constexpr std::size_t kHeadR = kBlockFfnX;
inline constexpr std::size_t kHeadY = kBlockInY;

// The call after a chain's down gather.
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
      return 12288;
    case ChainNext::kAttention:
      return 9216;
    case ChainNext::kLmHead:
      return 248832;
    case ChainNext::kDraftHead:
      return 32768;
    case ChainNext::kNone:
      return 0;
  }
  return 0;
}

// The chain's sequence name (decode_gemv_variants.py chain_name): one token
// over 9 experts, two over 9 to 17.
inline std::string ChainSequence(ChainNext next, std::size_t experts,
                                 std::size_t tokens) {
  return std::string(kDecodeName) + "_chain_" + ChainNextName(next) + "_e" +
         std::to_string(experts) + "_t" + std::to_string(tokens);
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

static_assert(kInputBytes == 36864 && kFfnYBytes == 139264 &&
                  kBlockScratch == 348160 && kBlockBytes == 372736,
              "decode_gemv_variants.py INPUT_BYTES, FFN_Y_BYTES, "
              "BLOCK_SCRATCH and BLOCK_BYTES");

}  // namespace gufo_npu::qwen36_35b_a3b::npu
