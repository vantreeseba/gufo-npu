#pragma once

#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/gguf_projections.hpp"
#include "models/qwen35/projections.hpp"

// fp32 CPU reference forward pass for Qwen3.5 / Qwen3.6, dense (GGUF arch
// `qwen35`) or MoE (`qwen35moe`). Slow by design: weights stay in the
// memory-mapped GGUF and are dequantized row by row on use, and dot products
// accumulate in double. It is the numerical oracle for the NPU path.
// Math and tensor layout: docs/models/qwen3.5-2b/REFERENCE.md and, for the
// MoE FFN, docs/models/qwen3.6-35b-a3b/README.md.
// The const methods may run on several threads with separate States; the
// GGUF matrix products then take turns on one thread pool, and a
// caller-supplied Projections must be safe to share.
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
  // The reference's own matrix products over the GGUF weights.
  const Projections& projections() const;

  // Runs `tokens` after whatever `state` already holds and advances it.
  // `projections` replaces the reference's own GGUF matrix products (for
  // example with the NPU); the rest of the pass stays on the CPU.
  ForwardOutput Forward(const std::vector<int>& tokens, State& state,
                        bool want_logits = true, bool want_hidden = false,
                        const Projections* projections = nullptr) const;

  // The two pieces of Forward, for tests that feed another implementation's
  // activations into one layer at a time. ForwardLayer runs decoder layer
  // `layer` on rows of its residual input `h` (tokens x hidden, in place) at
  // positions state.position onwards; it updates that layer's state but not
  // state.position. Logits applies the final norm and the LM head to rows of
  // the last layer's output.
  void ForwardLayer(int layer, std::vector<float>& h, State& state,
                    const Projections* projections = nullptr) const;
  std::vector<float> Logits(const std::vector<float>& h,
                            const Projections* projections = nullptr) const;

  // The MTP (nextn) head, one draft per row: `h` holds rows of the main
  // model's last-layer output (before output_norm) at positions p - 1, and
  // `tokens` the tokens at positions p; returns logits for positions p + 1:
  //   x = eh_proj [RMS(embed(token)) * enorm ; RMS(RMS(h) * output_norm) *
  //   hnorm], the MTP block, then the LM head on RMS(x) * shared_head_norm.
  // The block runs at RoPE positions state.position onwards with its own
  // cache in `state` (keep one State for the MTP head, with position set to
  // the first token's p), and advances state.position.
  std::vector<float> Mtp(const std::vector<float>& h,
                         const std::vector<int>& tokens, State& state,
                         const Projections* projections = nullptr) const;

private:
  struct Weights;
  void LoadWeights(const gguf::File& file);
  // Runs layer `il` on `n` rows of the residual stream `h` (in place), at
  // positions state.position onwards.
  void RunLayer(int il, std::vector<float>& h, int n, State& state,
                const Projections& proj) const;
  // The layer's FFN of `n` rows of normed input: dense SwiGLU, or routed
  // experts plus the gated shared expert. Writes `n` rows of hidden.
  void Ffn(int il, const float* x, int n, const Projections& proj,
           float* out) const;
  // LM head logits of rows of `h` after RMS(h) * `norm`.
  std::vector<float> Head(const std::vector<float>& h,
                          const std::vector<float>& norm,
                          const Projections& proj) const;
  Config config_;
  std::unique_ptr<Weights> weights_;
  std::unique_ptr<GgufProjections> gguf_projections_;
};

}  // namespace gufo_npu::qwen35
