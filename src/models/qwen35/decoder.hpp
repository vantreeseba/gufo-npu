#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/projections.hpp"

// Fast single-token decode for Qwen3.5 / Qwen3.6, dense or MoE, in fp32: the
// reference's math, one token per step. The reference (reference.hpp) stays
// the oracle; these are tested against it.
namespace gufo_npu::qwen35 {

// Per-sequence state of a decode (decoder layers, then MTP layers).
struct DecodeState {
  int position = 0;
  struct Layer {
    std::vector<float> conv;  // [conv_kernel - 1][channels], oldest first
    std::vector<float> ssm;   // [value heads][key dim i][value dim j]
    // The attention cache in fp16 bits (vec::ToHalf), [position][kv_heads *
    // head_dim]; keys after q/k norm and RoPE.
    std::vector<std::uint16_t> keys;
    std::vector<std::uint16_t> values;
    // A speculative token's DeltaNet state and the conv window before it,
    // kept until the token is accepted or rolled back (NpuDecoder).
    std::vector<float> ssm_spec;
    std::vector<float> conv_saved;
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
  // the value_dim inputs of ssm_out to `y`. The DeltaNet state is updated in
  // place, or, with `ssm_out`, written there and left unchanged in `state`
  // (a speculative token that may be rolled back). The conv window always
  // advances in place.
  void Linear(int layer, const float* in, DecodeState::Layer& state, float* y,
              float* ssm_out = nullptr);
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
  // Adds the routed experts of MoE layer `il` to the shared expert's output
  // in the scratch, scaled by its gate (input: the normed hidden state).
  void MoeFfn(int il);
  Mixer mixer_;
  const Projections& projections_;
  std::unique_ptr<Weights> weights_;
  std::unique_ptr<Scratch> scratch_;
};

}  // namespace gufo_npu::qwen35
