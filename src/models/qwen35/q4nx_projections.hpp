#pragma once

#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "core/q4nx.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/projections.hpp"

// Qwen3.5 projections on the CPU over q4nx weights: the same packed weights
// the NPU decoder uses (NpuDecoder), so tests can separate the NPU's error
// from requantization. Each stage's matrices are stacked (qkv + z, q + k + v,
// gate + up); the DeltaNet alpha and beta rows (16 each, F32) stay in fp32.
namespace gufo_npu::qwen35 {

class Q4nxProjections : public Projections {
public:
  // Keeps pointers into `file`, which must outlive this.
  explicit Q4nxProjections(const gguf::File& file);
  ~Q4nxProjections() override;

  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override;

private:
  struct Layer;
  Config config_;
  std::vector<std::unique_ptr<Layer>> layers_;
  q4nx::Matrix lm_head_;
};

}  // namespace gufo_npu::qwen35
