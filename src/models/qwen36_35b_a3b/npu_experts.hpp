#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "core/burst_pool.hpp"
#include "models/qwen36_35b_a3b/expert_step.hpp"
#include "models/qwen36_35b_a3b/prefill_device.hpp"
#include "npu/gemv.hpp"

// The prefill's experts on the NPU (v1, docs/research/NPU-MOE-PREFILL.md):
// Prepare converts each expert with tokens (q4nx, read in place from the
// decoder's arena of its layer) to bfp16 in its slot of a scratch buffer.
// Run groups the tokens by expert into GEMM rows (each expert's padded to
// 256, 512 or 1,024 per call), and per group of about kGroupRows rows runs
// every expert's gate + up GEMM, SwiGLU times the weights on the CPU, every
// down GEMM, and the sum into the residual on the CPU.
namespace gufo_npu::qwen36_35b_a3b {

// `arenas`: per layer, the decoder's expert arena (npu::ExpertArena of
// `arena_experts` experts, the shared one last). `max_tokens`: the most
// tokens a pass has. The device, the pool and the arenas must outlive it.
std::unique_ptr<ExpertStep> MakeNpuExperts(PrefillDevice& device,
                                           BurstPool& pool,
                                           std::vector<npu::Buffer> arenas,
                                           std::size_t arena_experts,
                                           std::size_t max_tokens);

}  // namespace gufo_npu::qwen36_35b_a3b
