#pragma once

#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/projections.hpp"

// A Qwen3.5 / Qwen3.6 model's matrix products on the CPU, straight from the
// memory-mapped GGUF: each weight row is dequantized when it is used and
// nothing is unpacked ahead, so a 35B MoE costs no memory beyond the file's
// page cache. Rows are spread over a persistent thread pool. Checks every
// projection tensor's shape when constructed.
namespace gufo_npu::qwen35 {

class GgufProjections : public Projections {
public:
  enum class Precision {
    // Each dot product accumulated in double: the reference's oracle math.
    kDouble,
    // fp32 (AVX2) dot products: the CPU decoder's speed.
    kFloat,
  };

  // Keeps pointers into `file`, which must outlive this.
  GgufProjections(const gguf::File& file, Precision precision);
  ~GgufProjections() override;
  GgufProjections(const GgufProjections&) = delete;
  GgufProjections& operator=(const GgufProjections&) = delete;

  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override;
  void ApplyExpert(Stage stage, int layer, int expert, const float* x,
                   int tokens, float* y) const override;

  // Rows [first, first + rows) of a weight matrix.
  struct Rows {
    const gguf::TensorInfo* w = nullptr;
    std::size_t first = 0;
    std::size_t rows = 0;
  };

private:
  struct Layer;
  struct Pool;
  // y[t] = [W_0 x[t] | W_1 x[t] | ...] for the row ranges in `parts`.
  void MatMul(const std::vector<Rows>& parts, const float* x, int tokens,
              float* y) const;

  Config config_;
  Precision precision_;
  std::vector<Layer> layers_;
  const gguf::TensorInfo* lm_head_ = nullptr;
  std::unique_ptr<Pool> pool_;
};

}  // namespace gufo_npu::qwen35
