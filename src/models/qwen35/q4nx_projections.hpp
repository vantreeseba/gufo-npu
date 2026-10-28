#pragma once

#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "core/q4nx.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/projections.hpp"

// Qwen3.5 / Qwen3.6 projections on the CPU over q4nx weights: the same packed
// weights the NPU decoder uses (NpuDecoder), so tests can separate the NPU's
// error from requantization. Each stage's matrices are stacked (qkv + z,
// q + k + v, gate + up); the DeltaNet alpha and beta rows and the MoE router
// (F32 in the GGUF) stay in fp32, as on the NPU path. MoE routed experts that
// are Q4_K are unpacked per call by q4nx's rule (their codes carry over), so
// only the experts that requantize are held packed.
namespace gufo_npu::qwen35 {

class Q4nxProjections : public Projections {
public:
  // Keeps pointers into `file`, which must outlive this. `fit` must be
  // the one the model's NPU path packs with (qwen35::Q4nxFit, or the
  // model package's own), so both run the same weights.
  Q4nxProjections(const gguf::File& file, q4nx::Fit fit);
  ~Q4nxProjections() override;

  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override;
  void ApplyExpert(Stage stage, int layer, int expert, const float* x,
                   int tokens, float* y) const override;

private:
  struct Layer;
  Config config_;
  std::vector<std::unique_ptr<Layer>> layers_;
  q4nx::Matrix lm_head_;
};

}  // namespace gufo_npu::qwen35
