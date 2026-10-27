#pragma once

#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/projections.hpp"

// Fast single-token decode for dense Qwen3.5, in fp32: the reference's math,
// one token per step. The reference (reference.hpp) stays the oracle; these
// are tested against it.
namespace gufo_npu::qwen35 {

// Per-sequence state of a decode.
struct DecodeState {
  int position = 0;
  struct Layer {
    std::vector<float> conv;    // [conv_kernel - 1][channels], oldest first
    std::vector<float> ssm;     // [value heads][key dim i][value dim j]
    std::vector<float> keys;    // [position][kv_heads * head_dim]
    std::vector<float> values;  // [position][kv_heads * head_dim]
  };
  std::vector<Layer> layers;
};

// The token mixers between a layer's input and output projections: Gated
// DeltaNet (conv, delta rule, gated norm) and gated full attention (q/k norm,
// RoPE, softmax over the cache), with their small weights.
class Mixer {
public:
  explicit Mixer(const gguf::File& file);
  ~Mixer();
  Mixer(const Mixer&) = delete;
  Mixer& operator=(const Mixer&) = delete;

  const Config& config() const { return config_; }
  DecodeState NewState() const;

  // `in` is the layer's kLinearIn output [qkv | z | alpha | beta]; writes
  // the value_dim inputs of ssm_out to `y`.
  void Linear(int layer, const float* in, DecodeState::Layer& state, float* y);
  // `in` is the layer's kAttentionIn output [q and gate | k | v] (k is
  // normalized and rotated in place); writes the heads * head_dim inputs of
  // attn_output to `ctx`.
  void Attention(int layer, float* in, int position, DecodeState::Layer& state,
                 float* ctx);

private:
  struct Weights;
  struct Scratch;
  Config config_;
  std::unique_ptr<Weights> weights_;
  std::unique_ptr<Scratch> scratch_;
};

// The whole decode step on the CPU, with the matrix products delegated to a
// Projections.
class Decoder {
public:
  using State = DecodeState;

  // Keeps pointers to `file` and `projections`, which must outlive it.
  Decoder(const gguf::File& file, const Projections& projections);
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  const Config& config() const { return mixer_.config(); }
  State NewState() const { return mixer_.NewState(); }

  // Runs `token` at state.position, advances the state and returns the
  // logits (vocab values, valid until the next Step).
  const std::vector<float>& Step(int token, State& state);

private:
  struct Weights;
  struct Scratch;
  Mixer mixer_;
  const Projections& projections_;
  std::unique_ptr<Weights> weights_;
  std::unique_ptr<Scratch> scratch_;
};

}  // namespace gufo_npu::qwen35
