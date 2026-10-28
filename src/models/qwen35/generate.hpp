#pragma once

#include <algorithm>
#include <functional>
#include <vector>

#include "models/qwen35/decoder.hpp"
#include "models/qwen35/reference.hpp"

namespace gufo_npu::qwen35 {

struct GenerateStats {
  int prompt_tokens = 0;
  int generated_tokens = 0;
  double prefill_seconds = 0;
  double decode_seconds = 0;
};

// Runs `ids` after everything earlier and returns the last token's logits
// (vocab values, valid until the next call).
using ForwardFn = std::function<const float*(const std::vector<int>& ids)>;

// Greedy decoding: runs the prompt, then appends the argmax token until a
// stop id or `max_tokens`. `on_token` sees each generated id (including the
// stop id) as it is produced. Returns the generated ids.
std::vector<int> GreedyGenerate(const ForwardFn& forward, int vocab,
                                const std::vector<int>& prompt, int max_tokens,
                                const std::vector<int>& stop,
                                const std::function<void(int)>& on_token,
                                GenerateStats* stats = nullptr);

// The same over the CPU reference; `projections` is passed to every Forward.
std::vector<int> GreedyGenerate(const Reference& model,
                                const std::vector<int>& prompt, int max_tokens,
                                const std::vector<int>& stop,
                                const std::function<void(int)>& on_token,
                                GenerateStats* stats = nullptr,
                                const Projections* projections = nullptr);

// Speculative decoding on a model's NPU decoder (each model's
// SpeculativeGenerate): what one generation took.
struct SpeculativeStats {
  int prompt_tokens = 0;
  int generated_tokens = 0;
  int steps = 0;     // verify passes
  int accepted = 0;  // drafts kept
  double prefill_seconds = 0;
  double decode_seconds = 0;
};

// Prompt processing ahead of the decoder (NpuPrefill): runs prompt tokens
// [0, k) into a fresh `state` and returns k (<= prompt.size()), writing each
// token's final residual to rows [0, k) of `hidden`; with `mtp`, also fills
// the MTP cache at positions 1 .. min(k, prompt.size() - 1) (pairs (hidden
// row p - 1, prompt[p])). When k is the whole prompt, the decoder computes
// only the LM head on hidden row k - 1 for the first token; otherwise it
// runs the rest of the prompt.
using PromptPrefill =
    std::function<int(const std::vector<int>& prompt, DecodeState& state,
                      DecodeState* mtp, float* hidden)>;

}  // namespace gufo_npu::qwen35
