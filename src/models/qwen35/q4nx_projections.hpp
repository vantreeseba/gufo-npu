#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "core/gguf.hpp"
#include "core/q4nx.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/projections.hpp"

namespace gufo_npu::npu {
class GemvRuntime;
class Q4nxGemv;
}  // namespace gufo_npu::npu

// Qwen3.5 projections on q4nx weights: each stage's matrices are stacked into
// one q4nx matrix (qkv + z, q + k + v, gate + up) so a stage is one GEMV per
// token. The DeltaNet alpha and beta rows (16 each, F32) stay on the CPU.
//
// kNpu runs the GEMVs on the NPU; kCpu runs the same packed weights on the
// CPU, which separates kernel error from requantization error in tests.
namespace gufo_npu::qwen35 {

class Q4nxProjections : public Projections {
public:
  enum class Backend { kNpu, kCpu };

  // Keeps pointers into `file`, which must outlive this. `kernel_dir` is
  // used by kNpu.
  Q4nxProjections(const gguf::File& file, Backend backend,
                  const std::filesystem::path& kernel_dir = {});
  ~Q4nxProjections() override;

  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override;

  // Wall time spent in GEMVs since construction, and how many ran.
  double gemv_seconds() const { return gemv_seconds_; }
  long gemv_calls() const { return gemv_calls_; }

private:
  struct Gemv;
  struct Layer;
  void Run(Gemv& gemv, const float* x, float* y) const;

  Config config_;
  Backend backend_;
  std::unique_ptr<npu::GemvRuntime> runtime_;
  std::vector<std::unique_ptr<Layer>> layers_;
  std::unique_ptr<Gemv> lm_head_;
  mutable double gemv_seconds_ = 0;
  mutable long gemv_calls_ = 0;
};

}  // namespace gufo_npu::qwen35
