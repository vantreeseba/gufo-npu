#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"

// Qwen3.5 / Qwen3.6 decode on the NPU. Every matrix product, every RMS norm
// before one, the residual adds and SwiGLU run in the q4nx GEMV (its resnorm
// input stage and SwiGLU output stage); the residual stream stays on the
// device. Each GEMV writes into the input region of the next, so the calls
// from one layer's mixer output to the next layer's input projection (out,
// gate + up, down, next in) run as one chain. The CPU runs the token mixers
// (Gated DeltaNet, attention) and the DeltaNet alpha/beta rows: 25 NPU
// submissions per step for Qwen3.5-2B.
//
// MoE models (qwen35moe, Qwen3.6-35B-A3B) keep each layer's experts and its
// shared expert in one device arena. After the mixer's output projection
// (its own submission) the CPU routes in fp32 (moe.hpp) and points two
// gathers at the chosen experts (npu::Q4nxGather): gate + up with SwiGLU
// scaled by the router weights, then one down over all of them, chained
// with the next layer's input projection. Two submissions per layer. A
// two-token step reads the union of both tokens' experts once; each token
// weighs the experts it did not choose 0.
//
// A step takes one or two tokens; the weights are read once for both. The
// second token of a pair can be a speculative draft: Resolve() keeps it or
// rolls the state back to just after the first. The MTP head drafts the
// token after next on the NPU as well (two submissions per draft, four for
// an MoE MTP block); its LM head scores only the draft vocabulary, and the
// main model still checks each draft over every token.
namespace gufo_npu::qwen35 {

class NpuDecoder {
public:
  // Keeps pointers into `file`, which must outlive it.
  NpuDecoder(const gguf::File& file, const std::filesystem::path& kernel_dir);
  ~NpuDecoder();
  NpuDecoder(const NpuDecoder&) = delete;
  NpuDecoder& operator=(const NpuDecoder&) = delete;

  const Config& config() const;
  DecodeState NewState() const;

  // Whether the model has an MTP head, so Draft() works.
  bool can_draft() const;

  // Runs `token` at state.position, advances the state and returns the
  // logits (vocab values, valid until the next call).
  const std::vector<float>& Step(int token, DecodeState& state);

  // Runs `count` (1 or 2) tokens from state.position in one pass. Returns the
  // logits of token t at t * vocab. With count 2 the second token stays
  // speculative until Resolve(). Without `logits` (prompt passes whose
  // logits nobody reads) the LM head is skipped and it returns nullptr;
  // hidden() is set either way.
  const float* Step(const int* tokens, int count, DecodeState& state,
                    bool logits = true);
  // After a 2-token Step: keep the second token, or roll the state back to
  // just after the first.
  void Resolve(DecodeState& state, bool keep_second);

  // The final residual (before output_norm) of token t of the last Step:
  // the MTP head's input.
  const float* hidden(int t = 0) const;

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

  // MoE models: the experts layer `layer` chose for the last Step's first
  // token, config().experts_used of them by descending weight.
  const int* experts(int layer) const;

  // Wall time inside NPU submissions since construction, and their count;
  // CPU time in the mixers and draft input (including buffer syncs), and in
  // MoE routing (including patching the gathers).
  double npu_seconds() const;
  long submissions() const;
  double cpu_seconds() const;
  double router_seconds() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct SpeculativeStats {
  int prompt_tokens = 0;
  int generated_tokens = 0;
  int steps = 0;     // verify passes
  int accepted = 0;  // drafts kept
  double prefill_seconds = 0;
  double decode_seconds = 0;
};

// Prompt processing ahead of the decoder (NpuPrefill): runs prompt tokens
// [0, k) into a fresh `state` and returns k (< prompt.size()), writing each
// token's final residual to rows [0, k) of `hidden`; with `mtp`, also fills
// the MTP cache at positions [1, k] (pairs (hidden row p - 1, prompt[p])).
using PromptPrefill =
    std::function<int(const std::vector<int>& prompt, DecodeState& state,
                      DecodeState* mtp, float* hidden)>;

// Greedy decoding with MTP speculation: each step verifies one MTP draft in a
// 2-token pass and keeps it when the main model agrees, so the output is the
// main model's greedy output. The prompt runs through `prefill` when given,
// the rest of it two tokens per pass. With `draft` false, every step decodes
// one token (no MTP); `draft` needs can_draft(). `on_token` sees each
// generated id (including a stop id) as it is produced.
std::vector<int> SpeculativeGenerate(NpuDecoder& model,
                                     const std::vector<int>& prompt,
                                     int max_tokens,
                                     const std::vector<int>& stop, bool draft,
                                     const std::function<void(int)>& on_token,
                                     SpeculativeStats* stats = nullptr,
                                     const PromptPrefill& prefill = {});

}  // namespace gufo_npu::qwen35
