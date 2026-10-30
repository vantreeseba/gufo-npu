#pragma once

#include <filesystem>
#include <memory>

#include "core/gguf.hpp"
#include "models/qwen3vl_embed_2b/embedder.hpp"

// Qwen3-VL-Embedding-2B on the NPU. Every matrix product is a bfp16 GEMM on
// the model's own GEMM program (npu/embed_gemm.hpp); the CPU runs the norms,
// residual adds, SwiGLU, attention and pooling (ops.hpp).
//
// Inputs are packed back to back into one batch of rows (up to
// kMaxBatchRows, or one longer input alone), and the batch goes through the
// model a layer at a time. Each projection runs over the batch in GEMM
// chunks of up to 1,024 rows, two in flight: while the NPU multiplies one
// chunk, the CPU writes the next chunk's input and reads the previous
// one's output. Attention runs between the projections, over each input's
// own rows.
//
// The weights come from the GGUF (BF16, F16, F32 or a quantized type) and
// are rounded to bfp16 once, at load: about 1.6 GB of device memory for the
// 2B's 28 layers.
namespace gufo_npu::qwen3vl_embed_2b {

// Rows (tokens) per batch. Activations take ~45 KB per row on the host.
inline constexpr int kMaxBatchRows = 8192;

class NpuEmbedder final : public Embedder {
public:
  // Keeps pointers into `file`, which must outlive it. Loads the GEMM
  // program (npu::kGemmName) from kernel_dir; throws when it is missing,
  // the NPU cannot be opened or the GGUF's shapes are not the program's.
  NpuEmbedder(const gguf::File& file, const std::filesystem::path& kernel_dir);
  ~NpuEmbedder() override;
  NpuEmbedder(const NpuEmbedder&) = delete;
  NpuEmbedder& operator=(const NpuEmbedder&) = delete;

  const Config& config() const override;
  std::vector<std::vector<float>> Embed(
      const std::vector<std::vector<int>>& inputs) override;

  // Where the time went since construction.
  struct Profile {
    long tokens = 0;
    long gemms = 0;                // NPU GEMM calls
    double gemm_seconds = 0;       // waiting for them
    double sync_seconds = 0;       // syncing their operand buffers
    double attention_seconds = 0;  // attention on the CPU
    double other_seconds = 0;      // the rest (norms, conversions, adds)
    double total_seconds = 0;
  };
  const Profile& profile() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::qwen3vl_embed_2b
