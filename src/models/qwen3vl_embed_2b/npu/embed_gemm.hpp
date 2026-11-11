#pragma once

#include <cstddef>

// Qwen3-VL-Embedding-2B's GEMM program (src/models/qwen3vl_embed_2b/kernels):
// IRON's bfp16 GEMM built with this model's shapes into its own xclbin. These
// constants must match kernels/embed_gemm_variants.py.
namespace gufo_npu::qwen3vl_embed_2b::npu {

// The xclbin and the prefix of its instruction sequences.
inline constexpr char kGemmName[] = "qwen3vl_embed_2b_gemm";

inline constexpr std::size_t kHidden = 2048;
inline constexpr std::size_t kFfn = 6144;
// q (16 heads) | k (8) | v (8), 128 values per head.
inline constexpr std::size_t kInRows = 4096;
inline constexpr std::size_t kQDim = 2048;

// The vision tower (mmproj): width, MLP, a merged 2x2 patch block, and the
// patch embedding's K (3 x 16 x 16 values) padded to a multiple of 512.
inline constexpr std::size_t kVitHidden = 1024;
inline constexpr std::size_t kVitFfn = 4096;
inline constexpr std::size_t kVitMerged = 4 * kVitHidden;
inline constexpr std::size_t kPatchDim = 1024;

}  // namespace gufo_npu::qwen3vl_embed_2b::npu
