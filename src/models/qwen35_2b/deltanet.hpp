#pragma once

#include <cstddef>
#include <memory>

#include "core/gguf.hpp"
#include "core/thread_pool.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"

// Qwen3.5-2B's Gated DeltaNet decode step on the CPU, fused: one pool
// dispatch per layer for one or two tokens. Each value head's job computes
// its alpha/beta rows, the causal conv and SiLU of its q, k and v channels,
// the q/k L2 norms, the delta rule and the gated RMS norm; the delta rule
// runs over column blocks of the head's state that stay in L1 while both
// tokens of a pair are applied. Bit for bit the arithmetic of the family's
// qwen35::Mixer::Linear with the decoder's alpha/beta rows (each value is
// computed by the same expression in the same order; only the scheduling
// differs), which Reference() runs for the tests.
namespace gufo_npu::qwen35_2b {

using qwen35::Config;
using qwen35::DecodeState;

class DeltaNetStep {
public:
  // Loads every DeltaNet layer's small weights from `file`. Runs its jobs on
  // `pool` (the Mixer's, so the decode loop keeps one set of workers).
  DeltaNetStep(const gguf::File& file, const Config& config, SpinPool& pool);
  ~DeltaNetStep();
  DeltaNetStep(const DeltaNetStep&) = delete;
  DeltaNetStep& operator=(const DeltaNetStep&) = delete;

  // Layer `il` for `count` (1 or 2) tokens. Token t's input projection
  // output [qkv | z] is at in + t * in_stride, its residual (the layer's
  // input before attn_norm, for alpha/beta) at h + t * h_stride; its
  // value_dim outputs (the input of ssm_out) go to y + t * y_stride. One
  // token updates the state in place. Two tokens are a pair whose second
  // may be rolled back: the state after the first stays in state.ssm and
  // the conv window after it in state.conv_saved, the second's go to
  // state.ssm_spec and state.conv (NpuDecoder::Resolve picks).
  void Run(int il, int count, const float* in, std::size_t in_stride,
           const float* h, std::size_t h_stride, DecodeState::Layer& state,
           float* y, std::size_t y_stride);

  // The same through qwen35::Mixer::Linear, token by token, with the
  // alpha/beta rows computed as the decoder did before the fused step.
  void Reference(qwen35::Mixer& mixer, int il, int count, const float* in,
                 std::size_t in_stride, const float* h, std::size_t h_stride,
                 DecodeState::Layer& state, float* y, std::size_t y_stride);

  // Whether Run uses the AVX-512 delta rule (else the AVX2 build's loops;
  // the results are the same). For tests.
  bool wide() const;
  void set_wide(bool wide);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::qwen35_2b
