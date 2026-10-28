#pragma once

#include <filesystem>
#include <memory>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/generate.hpp"
#include "models/qwen35/prompt_cache.hpp"
#include "models/qwen35/state_capture.hpp"
#include "models/qwen35_4b/npu_decoder.hpp"

// Qwen3.5-4B prompt processing (prefill) on the NPU, on the 4B's own GEMM
// program (src/models/qwen35_4b/kernels/prefill_gemm.py). Copied from
// models/qwen35/npu_prefill (Qwen3.5-2B). A chunk of up to
// gufo_npu::npu::kGemmChunk prompt tokens goes through each layer at once: its
// matrix products run as bfp16 GEMMs on the NPU (the weights are read once per
// chunk), and the CPU runs the norms, residual adds
// and SwiGLU, the DeltaNet conv and recurrence over the chunk (value heads in
// parallel) and causal attention over the cache plus the chunk (heads and
// query blocks in parallel). The weights are the decoder's q4nx weights,
// rounded to bfp16.
//
// It fills the same DecodeState NpuDecoder decodes from (conv windows,
// DeltaNet states, KV caches, position), and the MTP layer's cache.
namespace gufo_npu::qwen35_4b {

using qwen35::Config;
using qwen35::DecodeState;
using qwen35::PromptCacheModel;
using qwen35::PromptPrefill;
using qwen35::StateCapture;

class NpuPrefill {
public:
  // Keeps pointers into `file`, which must outlive it. Loads the 4B's
  // prefill program (npu::kPrefillName) from kernel_dir.
  NpuPrefill(const gguf::File& file, const std::filesystem::path& kernel_dir);
  ~NpuPrefill();
  NpuPrefill(const NpuPrefill&) = delete;
  NpuPrefill& operator=(const NpuPrefill&) = delete;

  const Config& config() const;

  // Runs `count` tokens from state.position and advances the state. With
  // `hidden`, writes each token's final residual (before output_norm), count
  // rows of hidden values: the MTP head's input. With `capture`, copies the
  // DeltaNet layers' state at capture->position on the way.
  void Run(const int* tokens, int count, DecodeState& state, float* hidden,
           const StateCapture* capture = nullptr);

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

// Prompts shorter than this go through the decoder: a prefill pass costs
// ~0.07 s however short the prompt, the decode path ~28 ms per two tokens
// (crossover measured in BENCHMARKS.md).
inline constexpr int kMinPrefillPrompt = 7;

// The prompt hook for SpeculativeGenerate: the whole prompt through
// `prefill`, and the MTP cache over it; the decoder then runs only the LM
// head on the last token's final residual for its logits. Prompts too short
// to win from a prefill pass are left to the decoder.
PromptPrefill PrefillPrompt(NpuPrefill& prefill);

// How this model runs prompt tokens for a PromptCache (prompt_cache.hpp):
// `prefill` from kMinPrefillPrompt tokens, `decoder` below that.
PromptCacheModel MakePromptCacheModel(NpuDecoder& decoder, NpuPrefill& prefill);

}  // namespace gufo_npu::qwen35_4b
