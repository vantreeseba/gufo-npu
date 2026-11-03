#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "core/sampling.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/generate.hpp"

// Qwen3.5-2B decode on the NPU, on the 2B's own decode GEMV program
// (src/models/qwen35_2b/kernels). Copied from the decoder it shared with the
// 35B (now models/qwen36_35b_a3b/npu_decoder) without the MoE layers; the CPU
// side (config, Mixer, MTP draft vocabulary) is the family's shared code. Every
// matrix product, every RMS norm before one, the residual adds and SwiGLU run
// in the q4nx GEMV (its resnorm input stage and SwiGLU output stage); the
// residual stream stays on the device. Each GEMV writes into the input region
// of the next, so the calls from one layer's mixer output to the next layer's
// input projection (out, gate + up, down, next in) run as one chain sequence:
// one call, on regions of one activation buffer (npu/gemv.hpp). The CPU
// runs the token mixers (Gated DeltaNet, attention) and the DeltaNet alpha/beta
// rows: 25 NPU submissions per step.
//
// A step takes one or two tokens; the weights are read once for both. The
// second token of a pair can be a speculative draft: Resolve() keeps it or
// rolls the state back to just after the first. The MTP head drafts the
// token after next on the NPU as well (two submissions per draft); its LM
// head scores only the draft vocabulary, and the main model still checks
// each draft over every token.
namespace gufo_npu::qwen35_2b {

using qwen35::Config;
using qwen35::DecodeState;
using qwen35::PromptPrefill;
using qwen35::SpeculativeStats;

class NpuDecoder {
public:
  // Keeps pointers into `file`, which must outlive it. Loads the 2B's
  // decode program (npu::kDecodeName) from kernel_dir.
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

  // States for a generation from the decoder's pool (StatePool): equal to
  // NewState()'s, keeping a released state's buffers. The MTP state is made
  // the same way. Release hands one back when the generation is over.
  DecodeState AcquireState();
  void ReleaseState(DecodeState&& state);
  // A generation's per-prompt-token final residuals, `floats` values,
  // zeroed; reused (grow-only) across generations, valid until the next
  // call.
  float* PromptHidden(std::size_t floats);

  // Wall time inside NPU submissions since construction, and their count;
  // CPU time in the mixers and draft input (including buffer syncs).
  double npu_seconds() const;
  long submissions() const;
  double cpu_seconds() const;

  // Where decode time goes, since construction: per kind of NPU submission
  // its count and wall time (submit to completion), the host's buffer syncs
  // (inside the mixers and around the submissions), and the wall time inside
  // Step().
  struct Timeline {
    struct Submissions {
      long count = 0;
      double seconds = 0;
    };
    Submissions first;   // layer 0's input projection
    Submissions layers;  // a layer's FFN into the next projection or LM head
    Submissions head;    // the LM head alone (Logits)
    Submissions mtp;     // the MTP head's
    double sync_seconds = 0;
    long steps = 0;
    double step_seconds = 0;
  };
  const Timeline& timeline() const;

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
// `draft` false, every step decodes one token (no MTP); `draft` needs
// can_draft(). `on_token` sees each generated id (including a stop id) as it is
// produced; returning false ends the generation after that id.
std::vector<int> SpeculativeGenerate(
    NpuDecoder& model, const std::vector<int>& prompt, int max_tokens,
    const std::vector<int>& stop, bool draft,
    const std::function<bool(int)>& on_token, SpeculativeStats* stats = nullptr,
    const PromptPrefill& prefill = {}, const SamplingParams& sampling = {});

}  // namespace gufo_npu::qwen35_2b
