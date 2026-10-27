#pragma once

#include <cstdint>
#include <string>

#include "core/gguf.hpp"

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

  // MoE (qwen35moe only).
  int experts = 0;
  int experts_used = 0;
  int expert_ffn = 0;
  int shared_expert_ffn = 0;

  bool tied_embeddings = false;

  bool IsFullAttention(int layer) const {
    return (layer + 1) % full_attention_interval == 0;
  }
  int LinearKeyDim() const { return linear_key_heads * linear_head_dim; }
  int LinearValueDim() const { return linear_value_heads * linear_head_dim; }
  // Channels of the causal conv: q, k and v of the linear-attention input.
  int ConvChannels() const { return 2 * LinearKeyDim() + LinearValueDim(); }
};

// Throws std::runtime_error when required keys are missing or inconsistent.
Config ReadConfig(const gguf::File& file);

}  // namespace gufo_npu::qwen35
