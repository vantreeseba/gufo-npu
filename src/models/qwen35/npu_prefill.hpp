#pragma once

#include <filesystem>
#include <memory>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/npu_decoder.hpp"

// Qwen3.5 prompt processing (prefill) on the NPU. A chunk of up to
// npu::kGemmChunk prompt tokens goes through each layer at once: its matrix
// products run as bfp16 GEMMs on the NPU (kernels/designs/bfp16_gemm.py; the
// weights are read once per chunk), and the CPU runs the norms, residual adds
// and SwiGLU, the DeltaNet conv and recurrence over the chunk (value heads in
// parallel) and causal attention over the cache plus the chunk (heads and
// query blocks in parallel). The weights are the decoder's q4nx weights,
// rounded to bfp16.
//
// It fills the same DecodeState NpuDecoder decodes from (conv windows,
// DeltaNet states, KV caches, position), and the MTP layer's cache.
namespace gufo_npu::qwen35 {

class NpuPrefill {
public:
  // Keeps pointers into `file`, which must outlive it.
  NpuPrefill(const gguf::File& file, const std::filesystem::path& kernel_dir);
  ~NpuPrefill();
  NpuPrefill(const NpuPrefill&) = delete;
  NpuPrefill& operator=(const NpuPrefill&) = delete;

  const Config& config() const;

  // Runs `count` tokens from state.position and advances the state. With
  // `hidden`, writes each token's final residual (before output_norm), count
  // rows of hidden values: the MTP head's input.
  void Run(const int* tokens, int count, DecodeState& state, float* hidden);

  // The MTP layer's cache: for i < count, the pair (hidden row i, tokens[i])
  // at position + i, where hidden row i is the main model's final residual
  // at position + i - 1. Appends the MTP attention's keys and values to
  // `mtp` (an NpuDecoder MTP state); computes no logits.
  void FillMtp(const float* hidden, const int* tokens, int count, int position,
               DecodeState& mtp);

  // Where the time went since construction.
  struct Profile {
    long gemms = 0;                // NPU GEMM calls
    double gemm_seconds = 0;       // in them (start to wait)
    double sync_seconds = 0;       // syncing their operand buffers
    double deltanet_seconds = 0;   // DeltaNet mixers on the CPU
    double attention_seconds = 0;  // attention mixers on the CPU
    double other_seconds = 0;      // the rest (norms, conversions, adds)
  };
  const Profile& profile() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The prompt hook for SpeculativeGenerate: every prompt token but the last
// through `prefill`, and the MTP cache over them; the decoder then runs the
// last token for its logits.
PromptPrefill PrefillPrompt(NpuPrefill& prefill);

}  // namespace gufo_npu::qwen35
