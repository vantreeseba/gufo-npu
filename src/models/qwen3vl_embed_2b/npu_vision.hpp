#pragma once

#include <memory>

#include "core/gguf.hpp"
#include "models/qwen3vl_embed_2b/npu_embedder.hpp"
#include "models/qwen3vl_embed_2b/vision.hpp"

// Qwen3-VL-Embedding-2B's vision tower on the NPU (docs/models/
// qwen3-vl-embedding-2b/VISION.md, stage 2). Every linear (the patch
// embedding, each block's qkv, attention output, fc1 and fc2, and the four
// mergers' fc1 and fc2) is a bfp16 GEMM on the model's GEMM program, in the
// embedder's hardware context. The CPU runs the rest: the bias adds,
// LayerNorms, GELUs, 2D RoPE, the full attention (FullAttention, ops.hpp)
// and the position-embedding resample.
//
// An image goes through the tower a block at a time, each projection over
// its patches in GEMM chunks of up to 1,024 rows, two in flight (as the
// embedder's text batches).
//
// The weights come from the mmproj GGUF (F16, BF16 or F32) and are rounded
// to bfp16 once, at load: about 0.5 GB of device memory.
namespace gufo_npu::qwen3vl_embed_2b {

class NpuVision final : public VisionEncoder {
public:
  // Keeps pointers into `mmproj`, which must outlive it, and runs on
  // `embedder`'s GEMM program, so `embedder` must outlive it too. Throws
  // when the mmproj is not a Qwen3-VL one or its shapes are not the
  // program's.
  NpuVision(const gguf::File& mmproj, NpuEmbedder& embedder);
  ~NpuVision() override;
  NpuVision(const NpuVision&) = delete;
  NpuVision& operator=(const NpuVision&) = delete;

  const VisionConfig& config() const override;
  // As VisionReference::Encode.
  ImageFeatures Encode(const Patches& patches) override;

  // Where the time went since construction.
  struct Profile {
    long images = 0;
    long patches = 0;
    long gemms = 0;                // NPU GEMM calls
    double gemm_seconds = 0;       // waiting for them
    double sync_seconds = 0;       // syncing their operand buffers
    double attention_seconds = 0;  // attention on the CPU
    double other_seconds = 0;  // the rest (norms, GELU, biases, conversions)
    double total_seconds = 0;
  };
  const Profile& profile() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::qwen3vl_embed_2b
