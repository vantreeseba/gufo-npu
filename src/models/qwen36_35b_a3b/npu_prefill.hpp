#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/npu_decoder.hpp"
#include "models/qwen35/prompt_cache.hpp"
#include "models/qwen35/state_capture.hpp"

// Qwen3.6-35B-A3B prompt processing (prefill) on the NPU and the CPU
// (docs/research/NPU-MOE-PREFILL.md, docs/research/TTFT.md). The prompt goes
// through the model layer by layer, all of it (up to kMaxPromptChunk tokens
// per pass) through each layer before the next, so each expert's weights are
// read once per pass. The NPU converts the decoder's device weights (q4nx)
// to bfp16 (qwen36_35b_a3b_dequant), and bfp16 GEMMs (qwen36_35b_a3b_prefill)
// multiply them. Per layer:
//   1. GEMM program: the input projection, the mixer on the CPU (the
//      family's PrefillMixer: chunked DeltaNet or causal attention), the
//      output projection, in blocks of up to 1,024 tokens.
//   2. CPU: post-attention norm, router logits in fp32, top 8 (moe.hpp, as
//      decode routes), and the tokens grouped by expert.
//   3. The expert step (ExpertBackend below): the routed experts and the
//      shared one (one more expert over every token, weighed by its sigmoid
//      gate), each expert's SwiGLU times its router weight, summed into the
//      residual.
// It fills the same DecodeState NpuDecoder decodes from, and the MTP layer's
// cache.
namespace gufo_npu::qwen36_35b_a3b {

// Tokens per pass: the prompt is processed in passes of at most this many.
inline constexpr int kMaxPromptChunk = 16384;

// Where the routed and shared experts run (expert_step.hpp):
//   kCpu  cpu_experts.hpp: the decoder's q4nx arenas read in place on the
//         CPU. The NPU then only runs the dense matrices, so their bfp16
//         copies are converted once, at construction, and kept (one visit
//         to the dequant program in all).
//   kNpu  npu_experts.hpp: each expert with tokens converted per pass on
//         the NPU and run as GEMMs, with the next layer's dense matrices
//         converted beside it (two program switches per layer).
enum class ExpertBackend { kCpu, kNpu };

class NpuPrefill {
public:
  // Reads `decoder`'s device weights in place; `file` and `decoder` must
  // outlive it. Throws for models other than Qwen3.6-35B-A3B's shapes.
  NpuPrefill(const gguf::File& file, qwen35::NpuDecoder& decoder,
             const std::filesystem::path& kernel_dir,
             ExpertBackend experts = ExpertBackend::kCpu);
  ~NpuPrefill();
  NpuPrefill(const NpuPrefill&) = delete;
  NpuPrefill& operator=(const NpuPrefill&) = delete;

  const qwen35::Config& config() const;
  ExpertBackend experts() const;

  // Runs `count` tokens from state.position and advances the state. With
  // `hidden`, writes each token's final residual (before output_norm), count
  // rows of hidden values: the MTP head's input. With `capture`, copies the
  // DeltaNet layers' state at capture->position on the way.
  void Run(const int* tokens, int count, qwen35::DecodeState& state,
           float* hidden, const qwen35::StateCapture* capture = nullptr);

  // The MTP layer's cache: for i < count, the pair (hidden row i, tokens[i])
  // at position + i, where hidden row i is the main model's final residual
  // at position + i - 1. Appends the MTP attention's keys and values to
  // `mtp` (an NpuDecoder MTP state); computes no logits.
  void FillMtp(const float* hidden, const int* tokens, int count, int position,
               qwen35::DecodeState& mtp);

  // A test hook. With `experts` ([token][layer][experts_used] for the next
  // Run's tokens, e.g. NpuDecoder::experts() recorded token by token), each
  // token goes to those experts instead of its own top k, weighed by its own
  // router probabilities renormalized over them; nullptr routes by itself.
  // Wherever they differ from its own choice, routing_differences() counts
  // it and routing_gap() keeps the largest relative probability gap between
  // an expert only `experts` chose and one only the prefill chose (as
  // tests/models/qwen36_35b_a3b/npu_decode_test.cpp measures near ties).
  void FollowRouting(const std::vector<int>* experts);
  int routing_differences() const;
  double routing_gap() const;

  // Where the time went since construction.
  struct Profile {
    long passes = 0;             // layer-by-layer passes over the prompt
    long dequants = 0;           // NPU dequant calls
    long gemms = 0;              // NPU GEMM calls
    long switches = 0;           // moves between the two NPU programs
    double dequant_seconds = 0;  // in dequant batches (and their switches)
    double gemm_seconds = 0;     // in GEMM batches (and their switches)
    double sync_seconds = 0;     // syncing operand buffers
    double setup_seconds = 0;    // building the calls (runs, windows)
    double deltanet_seconds = 0;
    double attention_seconds = 0;
    double router_seconds = 0;  // router logits, top k, grouping
    // The experts' CPU work: gathering rows, SwiGLU and the scatter-add
    // (kNpu), or all of it (kCpu).
    double expert_seconds = 0;
    double other_seconds = 0;  // the rest (embedding, norms, conversions)
  };
  const Profile& profile() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The prompt hook for SpeculativeGenerate: every prompt token but the last
// through `prefill`, and the MTP cache over them; the decoder then runs the
// last token for its logits. Prompts shorter than MinPrefillPrompt(experts),
// which the decoder's two-token passes run faster, are left to the decoder
// (measured, BENCHMARKS.md; the decode path costs ~30 ms per token). With
// kCpu a prefill pass costs ~0.15 s at 4 tokens plus ~2 ms per token (time
// to first token 0.134 s by the decode path against 0.157 s at 4 tokens,
// 0.240 s against 0.175 s at 8); with kNpu ~2.78 s at 64 tokens plus ~9 ms
// per token, even with the decode path at ~108.
inline constexpr int kMinPrefillPromptNpu = 108;
inline constexpr int kMinPrefillPromptCpu = 6;
constexpr int MinPrefillPrompt(ExpertBackend experts) {
  return experts == ExpertBackend::kCpu ? kMinPrefillPromptCpu
                                        : kMinPrefillPromptNpu;
}
qwen35::PromptPrefill PrefillPrompt(NpuPrefill& prefill);

// How this model runs prompt tokens for a PromptCache (prompt_cache.hpp):
// `prefill` from MinPrefillPrompt tokens, `decoder` (the one `prefill`
// reads) below that; the prompt's last token stays with the decoder, as in
// PrefillPrompt.
qwen35::PromptCacheModel MakePromptCacheModel(qwen35::NpuDecoder& decoder,
                                              NpuPrefill& prefill);

}  // namespace gufo_npu::qwen36_35b_a3b
