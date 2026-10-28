#pragma once

#include <cstddef>
#include <new>
#include <vector>

#include "core/burst_pool.hpp"
#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/state_capture.hpp"

// The token mixers of a prompt chunk on the CPU, for the NPU prefills of the
// Qwen3.5 family (src/models/qwen35/npu_prefill.cpp, the 2B's; the 35B's in
// src/models/qwen36_35b_a3b): Gated DeltaNet (conv, delta rule, gated norm)
// over the chunk with the value heads in parallel, and gated attention,
// causal over the cache plus the chunk, with heads and query blocks in
// parallel. They read the chunk's input projection (fp32 rows, as the
// decoder lays it out) and fill the same DecodeState the decoders decode
// from.
namespace gufo_npu::qwen35 {

// Scratch rows start on a cache line, so parallel jobs that write different
// 128-byte column blocks of a row never share a line.
template<typename T>
struct CacheAligned {
  using value_type = T;
  static constexpr std::align_val_t kAlign{64};
  CacheAligned() = default;
  template<typename U>
  explicit CacheAligned(const CacheAligned<U>&) {}
  T* allocate(std::size_t n) {
    return static_cast<T*>(::operator new(n * sizeof(T), kAlign));
  }
  void deallocate(T* p, std::size_t) { ::operator delete(p, kAlign); }
  friend bool operator==(const CacheAligned&, const CacheAligned&) {
    return true;
  }
};
using Floats = std::vector<float, CacheAligned<float>>;

// The small weights one layer's mixer reads on the CPU.
struct PrefillMixerWeights {
  bool attention = false;
  // Gated DeltaNet.
  std::vector<float> alpha, beta;  // [value heads][hidden]
  std::vector<float> conv;         // [kernel][channels] (transposed)
  std::vector<float> ssm_a, dt_bias, ssm_norm;
  // Gated attention.
  std::vector<float> q_norm, k_norm;
};

// Layer `il`'s mixer weights (tensors blk.<il>.*).
PrefillMixerWeights LoadPrefillMixer(const gguf::File& file,
                                     const Config& config, int il);

class PrefillMixer {
public:
  // Scratch for chunks of up to `max_tokens`; jobs run on `pool`, which must
  // outlive it. Throws for head shapes the kernels do not handle (a multiple
  // of 4 query heads per kv head, attention and DeltaNet head sizes that are
  // multiples of 32, DeltaNet heads of at most 128).
  PrefillMixer(const Config& config, BurstPool& pool, int max_tokens);

  // The chunk the next calls work on: `count` tokens from `position` (their
  // RoPE angles, as the decoder computes them).
  void SetChunk(int count, int position);
  int count() const { return n_; }

  // `in`: the chunk's input projection rows (fp32, `in_stride` apart):
  // [q and gate | k | v] for attention, [qkv | z] for DeltaNet.

  // Normalizes and rotates the chunk's keys and appends them and the
  // values to `state`'s cache, as fp16 (the decoder's cache format).
  void AppendKv(const float* in, std::size_t in_stride,
                const PrefillMixerWeights& w, DecodeState::Layer& state);
  // Gated attention of the chunk into `mixed` (heads * head_dim per token):
  // appends its keys and values, then attends causally over the cache.
  void Attention(const float* in, std::size_t in_stride,
                 const PrefillMixerWeights& w, DecodeState::Layer& state,
                 float* mixed);
  // Gated DeltaNet of the chunk into `mixed` (value_dim per token); `normed`
  // holds the layer's normalized input (hidden per token), for the gates.
  // With `capture`, also copies the state after its first offset tokens.
  void DeltaNet(const float* in, std::size_t in_stride, const float* normed,
                const PrefillMixerWeights& w, DecodeState::Layer& state,
                float* mixed, const DeltaCapture& capture = {});

private:
  void Rope(float* x, int t) const;

  Config config_;
  BurstPool& pool_;
  int max_tokens_ = 0;
  int n_ = 0;
  int position_ = 0;
  std::vector<float> rope_inv_freq_;
  Floats gates_;                // decay | beta per token and value head
  Floats u_;                    // conv output: q | k | v per token
  Floats queries_;              // normed, rotated and scaled
  Floats rope_cos_, rope_sin_;  // per token
};

}  // namespace gufo_npu::qwen35
