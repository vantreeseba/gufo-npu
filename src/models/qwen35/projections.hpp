#pragma once

#include <cstddef>
#include <stdexcept>

#include "models/qwen35/config.hpp"

// The matrix products of a Qwen3.5 forward pass, grouped by shared input so a
// backend can fuse them. The reference computes everything else (norms,
// convolution, DeltaNet, attention, activations, expert routing) and calls a
// Projections for these. Each output row is one token, laid out as listed.
namespace gufo_npu::qwen35 {

enum class Stage {
  // DeltaNet input: [qkv (conv channels) | z (value dim) | alpha | beta
  // (value heads each)].
  kLinearIn,
  // Full-attention input: [q and gate (2 * heads * head_dim) | k | v
  // (kv_heads * head_dim each)].
  kAttentionIn,
  // ssm_out or attn_output: hidden.
  kMixerOut,
  // [gate | up] (Config::FfnWidth() each): the dense FFN, or the MoE shared
  // expert.
  kFfnIn,
  // ffn_down or ffn_down_shexp: hidden.
  kFfnDown,
  // The LM head (token_embd when tied, else output): vocab.
  kLmHead,
  // MoE router: [expert logits (experts) | shared-expert gate logit (1)].
  kRouter,
  // A routed expert's [gate | up] (expert_ffn each); ApplyExpert only.
  kExpertIn,
  // A routed expert's down projection: hidden; ApplyExpert only.
  kExpertDown,
  // MTP layer input, nextn.eh_proj: [embedding | hidden] (2 * hidden) to
  // hidden. Its layer is the MTP layer (Config::layers).
  kMtpIn,
};

// Input and output widths of `stage` at `layer`.
std::size_t StageInputs(const Config& c, Stage stage, int layer);
std::size_t StageOutputs(const Config& c, Stage stage, int layer);

class Projections {
public:
  virtual ~Projections() = default;
  // y[t] = W x[t] for `tokens` rows of StageInputs values; writes `tokens`
  // rows of StageOutputs values.
  virtual void Apply(Stage stage, int layer, const float* x, int tokens,
                     float* y) const = 0;
  // The same for routed expert `expert` of an MoE layer (kExpertIn or
  // kExpertDown).
  virtual void ApplyExpert(Stage /*stage*/, int /*layer*/, int /*expert*/,
                           const float* /*x*/, int /*tokens*/,
                           float* /*y*/) const {
    throw std::logic_error("qwen35: these projections have no MoE experts");
  }
};

}  // namespace gufo_npu::qwen35
