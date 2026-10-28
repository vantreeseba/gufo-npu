#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "core/q4nx.hpp"
#include "core/sampling.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/generate.hpp"

// Qwen3.5-4B decode on the NPU, on the 4B's own decode GEMV program
// (src/models/qwen35_4b/kernels; hidden 2560). Copied from
// models/qwen35/npu_decoder (Qwen3.5-2B) with the 4B's geometry; the CPU
// side (config, Mixer, MTP draft vocabulary) is the family's shared code.
// Every matrix product, every RMS norm before one,
// the residual adds and SwiGLU run in the q4nx GEMV (its resnorm input stage
// and SwiGLU output stage); the residual stream stays on the device. Each
// GEMV writes into the input region of the next, so the calls from one
// layer's mixer output to the next layer's input projection (out, gate + up,
// down, next in) run as one chain. The CPU runs the token mixers (Gated
// DeltaNet, attention) and the DeltaNet alpha/beta rows: 25 NPU submissions
// per step.
//
// A step takes one or two tokens; the weights are read once for both. The
// second token of a pair can be a speculative draft: Resolve() keeps it or
// rolls the state back to just after the first. The MTP head drafts the
// token after next on the NPU as well (two submissions per draft); its LM
// head scores only the draft vocabulary, and the main model still checks
// each draft over every token.
namespace gufo_npu::qwen35_4b {

// The q4nx fit for this model's requantized weights (decoder, MTP block,
// draft head and prefill): the search fit lowers KL against the GGUF by a
// fifth and raises top-1 on held-out text (qwen3.5-4b/EXPERIMENTS.md).
inline constexpr q4nx::Fit kQ4nxFit = q4nx::Fit::kSearch;

using qwen35::Config;
using qwen35::DecodeState;
using qwen35::PromptPrefill;
using qwen35::SpeculativeStats;

class NpuDecoder {
public:
  // Keeps pointers into `file`, which must outlive it. Loads the 4B's
  // decode program (npu::kDecodeName) from kernel_dir.
  NpuDecoder(const gguf::File& file, const std::filesystem::path& kernel_dir);
  ~NpuDecoder();
  NpuDecoder(const NpuDecoder&) = delete;
  NpuDecoder& operator=(const NpuDecoder&) = delete;

  const Config& config() const;
  DecodeState NewState() const;

  // Runs `token` at state.position, advances the state and returns the
  // logits (vocab values, valid until the next call).
  const std::vector<float>& Step(int token, DecodeState& state);

  // Runs `count` (1 or 2) tokens from state.position in one pass. Returns the
  // logits of token t at t * vocab. With count 2 the second token stays
  // speculative until Resolve().
  const float* Step(const int* tokens, int count, DecodeState& state);
  // After a 2-token Step: keep the second token, or roll the state back to
  // just after the first.
  void Resolve(DecodeState& state, bool keep_second);

  // The final residual (before output_norm) of token t of the last Step:
  // the MTP head's input.
  const float* hidden(int t = 0) const;

  // The LM head alone (output_norm, then the logits) on one final residual
  // `hidden` (config().hidden values), e.g. a prompt's last row from a
  // prefill; one NPU submission. Returns vocab logits, valid until the next
  // call; hidden(0) becomes `hidden`.
  const float* Logits(const float* hidden);

  // MTP draft: for each of `count` (1 or 2) positions p = position + i, the
  // pair (hidden[i], tokens[i]), where hidden[i] is the main model's final
  // residual at p - 1 and tokens[i] the token at p. Appends to the MTP
  // layer's cache in `state` (an MtpState) and returns the draft for the
  // position after the last: the MTP head's argmax over the draft
  // vocabulary (DraftVocab()).
  int Draft(const float* const* hidden, const int* tokens, int count,
            int position, DecodeState& state);
  // Draft() without the draft: only appends the pairs to the MTP layer's
  // cache (for prompt positions).
  void FillMtp(const float* const* hidden, const int* tokens, int count,
               int position, DecodeState& state);
  DecodeState NewMtpState() const;

  // Wall time inside NPU submissions since construction, and their count;
  // CPU time in the mixers and draft input (including buffer syncs).
  double npu_seconds() const;
  long submissions() const;
  double cpu_seconds() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Decoding with MTP speculation: each step verifies one MTP draft in a
// 2-token pass. The draft is a point proposal (the MTP head's argmax), so the
// step draws the first token from the main model's distribution and keeps
// the draft when they agree: it is accepted with probability p(draft), and a
// miss is distributed as p without the draft, which is speculative sampling's
// rule. The output therefore follows the main model's distribution exactly;
// greedy `sampling` (the default) gives its greedy output. The prompt runs
// through `prefill` when given, the rest of it two tokens per pass. With
// `draft` false, every step decodes one token (no MTP). `on_token` sees each
// generated id (including a stop id) as it is produced; returning false ends
// the generation after that id.
std::vector<int> SpeculativeGenerate(
    NpuDecoder& model, const std::vector<int>& prompt, int max_tokens,
    const std::vector<int>& stop, bool draft,
    const std::function<bool(int)>& on_token, SpeculativeStats* stats = nullptr,
    const PromptPrefill& prefill = {}, const SamplingParams& sampling = {});

}  // namespace gufo_npu::qwen35_4b
