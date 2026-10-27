#pragma once

#include <cstddef>

#include "models/qwen35/config.hpp"

// The matrix products of a Qwen3.5 forward pass, grouped by shared input so a
// backend can fuse them. The reference computes everything else (norms,
// convolution, DeltaNet, attention, activations) and calls a Projections for
// these. Each output row is one token, laid out as listed.
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
  // [gate (ffn) | up (ffn)].
  kFfnIn,
  // ffn_down: hidden.
  kFfnDown,
  // Tied embedding as the output projection: vocab.
  kLmHead,
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
};

}  // namespace gufo_npu::qwen35
