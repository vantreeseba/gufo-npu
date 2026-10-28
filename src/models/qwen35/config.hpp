#pragma once

#include <cstdint>
#include <string>

#include "core/gguf.hpp"
#include "core/q4nx.hpp"

// Shape of a Qwen3.5 / Qwen3.6 text model, read from GGUF arch `qwen35`
// (dense) or `qwen35moe`.
namespace gufo_npu::qwen35 {

struct Config {
  std::string architecture;
  int layers = 0;      // decoder layers, excluding MTP layers
  int mtp_layers = 0;  // nextn layers stored after the decoder layers
  int hidden = 0;
  int ffn = 0;  // dense FFN width (0 for MoE)
  int vocab = 0;
  float rms_eps = 0;
  int full_attention_interval = 0;

  // Gated full attention.
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int rope_dim = 0;  // leading dims of each head that are rotated
  float rope_theta = 0;

  // Gated DeltaNet (linear attention).
  int conv_kernel = 0;
  int linear_key_heads = 0;
  int linear_value_heads = 0;
  int linear_head_dim = 0;  // key and value head dim

  // MoE (qwen35moe only): `experts_used` of `experts` routed SwiGLU experts
  // (softmax over all router logits, top-k, weights renormalized to sum 1)
  // plus one shared expert scaled by sigmoid(ffn_gate_inp_shexp . x).
  int experts = 0;
  int experts_used = 0;
  int expert_ffn = 0;
  int shared_expert_ffn = 0;

  // Without output.weight the LM head is token_embd.
  bool tied_embeddings = false;

  bool IsMoe() const { return experts > 0; }
  // Width of the dense FFN, or of the MoE shared expert.
  int FfnWidth() const { return IsMoe() ? shared_expert_ffn : ffn; }

  // MTP layers (index >= layers) are full attention.
  bool IsFullAttention(int layer) const {
    return layer >= layers || (layer + 1) % full_attention_interval == 0;
  }
  int LinearKeyDim() const { return linear_key_heads * linear_head_dim; }
  int LinearValueDim() const { return linear_value_heads * linear_head_dim; }
  // Channels of the causal conv: q, k and v of the linear-attention input.
  int ConvChannels() const { return 2 * LinearKeyDim() + LinearValueDim(); }
};

// Throws std::runtime_error when required keys are missing or inconsistent.
Config ReadConfig(const gguf::File& file);

// The q4nx fit this package packs a model's requantized weights with: the
// search fit for Qwen3.6-35B-A3B (the MoE), min/max for Qwen3.5-2B (the dense
// model), where the search fit lowers KL but not top-1 on the matrices the NPU
// runs (qwen3.5-2b/EXPERIMENTS.md, qwen3.6-35b-a3b/EXPERIMENTS.md).
inline q4nx::Fit Q4nxFit(const Config& c) {
  return c.IsMoe() ? q4nx::Fit::kSearch : q4nx::Fit::kMinMax;
}

// True when the GGUF declares MTP layers and holds the first one's nextn
// tensors (only that one is used). Without them the main model still loads.
bool HasMtpLayer(const gguf::File& file, const Config& c);

// Returns `c` if it is a dense model with tied embeddings (token_embd is the
// LM head; ffn_gate/up/down per layer), else throws std::runtime_error naming
// `who`. For the paths that only support Qwen3.5-2B-like models.
Config RequireDenseTied(Config c, const char* who);

}  // namespace gufo_npu::qwen35
