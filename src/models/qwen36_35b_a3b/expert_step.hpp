#pragma once

#include <cstdint>
#include <vector>

#include "npu/program.hpp"

// The expert step of Qwen3.6-35B-A3B's prefill: what runs between a layer's
// routing (done by the prefill core, as decode routes) and the next layer.
// The core owns the layer-by-layer flow, the mixers, routing and grouping,
// the dense matrices and the MTP cache; an ExpertStep computes the layer's
// routed experts and its shared expert over the pass's tokens and adds them
// into the residual. The NPU one (npu_experts.hpp) converts each expert with
// tokens from the decoder's arena and runs GEMMs per expert; a CPU one could
// plug in the same way (docs/research/TTFT.md weighs that design).
namespace gufo_npu::qwen36_35b_a3b {

// A layer's routing over a pass's tokens, per arena expert (the routed ones,
// then the shared one, which takes every token): the tokens it takes, in
// ascending order, and its weight for each (the router's renormalized
// probability; the shared expert's sigmoid gate).
struct ExpertRouting {
  const std::vector<std::vector<int>>& tokens;
  const std::vector<std::vector<float>>& weights;
};

class ExpertStep {
public:
  virtual ~ExpertStep() = default;

  // Before Run: queues on `dequant` whatever conversions layer `il`'s experts
  // need on the NPU. The core runs that batch next, together with its own
  // (the next layer's dense matrices), in one visit to the dequant program.
  virtual void Prepare(int il, const ExpertRouting& routing,
                       npu::Batch& dequant) = 0;

  // h[t] += the sum over the experts e that take token t of weight(e, t) *
  // expert_e(x[t]), for x the post-attention norm of the pass's tokens
  // (hidden bf16 values per token) and h their residual (hidden fp32 values
  // per token).
  virtual void Run(int il, const ExpertRouting& routing, const std::uint16_t* x,
                   float* h) = 0;
};

}  // namespace gufo_npu::qwen36_35b_a3b
