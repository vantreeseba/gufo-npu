#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "models/qwen3vl_embed_2b/ops.hpp"

// Qwen3-VL-Embedding-2B's ViT attention on the NPU (src/models/
// qwen3vl_embed_2b/kernels/vit_attention.py): softmax(q k^T) v for the 16
// heads of one image, every patch seeing every other, on the whole array in
// the program's own hardware context. The constants must match
// kernels/vit_attention_variants.py and the core program
// (kernels/aie/vit_attention.cc).
//
// The host splits q, k and v into bfp16ebs8 hi and lo parts in the order
// the cores read them; the cores keep the softmax state and the output in
// fp32 and return bf16 (docs/models/qwen3-vl-embedding-2b/EXPERIMENTS.md).
namespace gufo_npu::qwen3vl_embed_2b::npu {

// The xclbin and the prefix of its instruction sequences.
inline constexpr char kVitAttnName[] = "qwen3vl_embed_2b_vit_attn";
inline constexpr int kVitAttnHeads = 16;
inline constexpr int kVitAttnHeadDim = 64;
// Patches are padded to a multiple of kVitAttnBucket (4 query blocks of 64,
// one per core of a column), up to the embedder's image cap.
inline constexpr int kVitAttnBucket = 256;
inline constexpr int kVitAttnMaxPatches = 1800 * 4;
inline constexpr int kVitAttnQueryBlock = 64;
inline constexpr int kVitAttnKeyBlock = 32;
// An 8x8 tile as bfp16ebs8 (8 blocks of an exponent byte and 8 mantissas).
inline constexpr std::size_t kVitAttnTileBytes = 72;
// A query block: split tiles [query tile][dim tile] (hi, then lo).
inline constexpr std::size_t kVitAttnQBytes = 8 * 8 * 2 * kVitAttnTileBytes;
// A key block: split k tiles [dim tile][key tile], kVitAttnKeyBlock fp32
// biases, then split v^T tiles [dim tile / 4][key tile][dim tile % 4].
inline constexpr std::size_t kVitAttnKSplitBytes =
    8 * 4 * 2 * kVitAttnTileBytes;
inline constexpr std::size_t kVitAttnKBytes =
    kVitAttnKSplitBytes + kVitAttnKeyBlock * 4;
inline constexpr std::size_t kVitAttnKVBytes =
    kVitAttnKBytes + 8 * 4 * 2 * kVitAttnTileBytes;
// A quarter of a query block's output: 2 query tiles x 8 dim tiles, bf16.
inline constexpr std::size_t kVitAttnOBytes = 2 * 8 * 64 * 2;

// `patches` rounded up to the program's lengths; throws std::invalid_argument
// outside [1, kVitAttnMaxPatches].
int VitAttnLength(int patches);
// The instruction sequence for a padded length.
std::string VitAttnSequence(int length);

namespace detail {
// Whether the CPU runs the AVX-512 tile split.
bool HasAvx512();
// An 8x8 tile (row-major fp32) as bfp16ebs8 hi then lo, 2 x 72 bytes, one
// block per tile row: hi = bfp16(x), lo = bfp16(x - hi), each rounded as
// bfp16::EncodeBlock does (ties to even, saturating). With avx512 (throws
// std::invalid_argument without HasAvx512()) or the scalar encoder. For
// tests.
void SplitTile(const float* tile, std::uint8_t* out, bool avx512);
}  // namespace detail

class VitAttention {
public:
  // Loads kVitAttnName from `kernel_dir` into its own hardware context and
  // allocates the operands for the longest image (~70 MB). Throws when the
  // program is missing or the NPU cannot be opened.
  explicit VitAttention(const std::filesystem::path& kernel_dir);
  ~VitAttention();
  VitAttention(const VitAttention&) = delete;
  VitAttention& operator=(const VitAttention&) = delete;

  // Sets up for an image of `patches` patches: its padded length and the
  // key masks. Throws std::invalid_argument outside [1,
  // kVitAttnMaxPatches].
  void Reset(int patches);
  // Attention over the image's patches. `qkv` holds a row per patch of 3 *
  // heads * head_dim values, as FullAttention takes them (ops.hpp): q | k |
  // v, heads contiguous, q and k rotated and q scaled by 1 / sqrt(head_dim).
  void Run(const float* qkv, const ParallelRun& run);
  // Patch `row`'s heads' outputs (heads * head_dim values, bf16) of the last
  // Run.
  void Output(int row, std::uint16_t* out) const;

  // Where the time went since construction.
  struct Profile {
    long calls = 0;
    double pack_seconds = 0;  // splitting q, k and v on the CPU
    double npu_seconds = 0;   // syncing the operands and the NPU call
  };
  const Profile& profile() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::qwen3vl_embed_2b::npu
