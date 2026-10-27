#pragma once

#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/projections.hpp"

// fp32 CPU reference forward pass for dense Qwen3.5 (GGUF arch `qwen35`).
// Slow by design: weights are dequantized row by row and dot products
// accumulate in double. It is the numerical oracle for the NPU path.
// Math and tensor layout: docs/models/qwen3.5-2b/REFERENCE.md.
namespace gufo_npu::qwen35 {

// Per-sequence state: conv windows, DeltaNet states and the KV cache.
struct State {
  int position = 0;
  struct Linear {
    std::vector<float> conv;  // [conv_kernel - 1][conv channels], oldest first
    std::vector<float> ssm;   // [value heads][key dim i][value dim j]
  };
  struct Attention {
    std::vector<float> keys;    // [position][kv_heads * head_dim], post-RoPE
    std::vector<float> values;  // [position][kv_heads * head_dim]
  };
  std::vector<Linear> linear;        // indexed by layer (unused for attention)
  std::vector<Attention> attention;  // indexed by layer (unused for linear)
};

struct ForwardOutput {
  // [tokens][vocab] when requested.
  std::vector<float> logits;
  // [layers + 1][tokens][hidden]: embeddings, then each layer's output.
  std::vector<float> hidden;
};

class Reference {
public:
  // Keeps a pointer to `file`, which must outlive the Reference.
  explicit Reference(const gguf::File& file);
  ~Reference();
  Reference(const Reference&) = delete;
  Reference& operator=(const Reference&) = delete;

  const Config& config() const { return config_; }
  State NewState() const;

  // Runs `tokens` after whatever `state` already holds and advances it.
  // `projections` replaces the reference's own GGUF matrix products (for
  // example with the NPU); the rest of the pass stays on the CPU.
  ForwardOutput Forward(const std::vector<int>& tokens, State& state,
                        bool want_logits = true, bool want_hidden = false,
                        const Projections* projections = nullptr) const;

private:
  struct Weights;
  class GgufProjections;
  void LoadWeights(const gguf::File& file);
  Config config_;
  std::unique_ptr<Weights> weights_;
  std::unique_ptr<GgufProjections> gguf_projections_;
};

}  // namespace gufo_npu::qwen35
