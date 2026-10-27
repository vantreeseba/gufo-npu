#pragma once

#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/projections.hpp"

// Fast single-token decode for dense Qwen3.5: the reference's math in fp32,
// one token per step, with the matrix products delegated to a Projections
// (normally the NPU). The reference (reference.hpp) stays the oracle; this is
// tested against it with the same projections.
namespace gufo_npu::qwen35 {

class Decoder {
public:
  struct State {
    int position = 0;
    struct Layer {
      std::vector<float> conv;    // [conv_kernel - 1][channels], oldest first
      std::vector<float> ssm;     // [value heads][key dim i][value dim j]
      std::vector<float> keys;    // [position][kv_heads * head_dim]
      std::vector<float> values;  // [position][kv_heads * head_dim]
    };
    std::vector<Layer> layers;
  };

  // Keeps pointers to `file` and `projections`, which must outlive it.
  Decoder(const gguf::File& file, const Projections& projections);
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  const Config& config() const { return config_; }
  State NewState() const;

  // Runs `token` at state.position, advances the state and returns the
  // logits (vocab values, valid until the next Step).
  const std::vector<float>& Step(int token, State& state);

private:
  struct Weights;
  struct Scratch;
  void LinearLayer(int il, State::Layer& st);
  void AttentionLayer(int il, int position, State::Layer& st);

  Config config_;
  const Projections& projections_;
  std::unique_ptr<Weights> weights_;
  std::unique_ptr<Scratch> scratch_;
};

}  // namespace gufo_npu::qwen35
