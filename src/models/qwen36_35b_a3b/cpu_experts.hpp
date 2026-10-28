#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/burst_pool.hpp"
#include "models/qwen36_35b_a3b/expert_step.hpp"

// The prefill's experts on the CPU (docs/research/TTFT.md, item 5): the
// decoder's q4nx expert arenas are read in place from host memory, one
// 32-row tile at a time unpacked to bf16 (each weight rounded once), and
// multiplied with bf16 activations into fp32 sums (AVX-512 BF16 dot
// products when the CPU has them, a portable loop otherwise). Per layer:
//   1. gate + up, per (expert, 32-row tile, run of its tokens): the tile's
//      16 gate and 16 up rows (q4nx::SwiGluOrder) give 16 SwiGLU values,
//      times the token's weight, rounded to bf16;
//   2. down, per 32 output columns: every expert in arena order over its
//      tokens, added into the residual, so each column of the residual has
//      one writer and the same order on every run.
// It converts nothing on the NPU (Prepare adds no calls).
namespace gufo_npu::qwen36_35b_a3b {

// `arenas`: per layer, the host address of the decoder's expert arena
// (npu::ExpertArena of `arena_experts` experts, the shared one last), which
// must outlive it. `max_tokens`: the most tokens a pass has. `portable`
// forces the portable path (tests); otherwise AVX-512 BF16 is used when the
// CPU has it.
std::unique_ptr<ExpertStep> MakeCpuExperts(
    BurstPool& pool, std::vector<const std::uint8_t*> arenas,
    std::size_t arena_experts, std::size_t max_tokens, bool portable = false);

// Whether the fast path (AVX-512 BF16 and VBMI) runs on this CPU.
bool CpuExpertsFast();

}  // namespace gufo_npu::qwen36_35b_a3b
