#include "models/qwen36_35b_a3b/npu_experts.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/quant.hpp"
#include "core/simd.hpp"

namespace gufo_npu::qwen36_35b_a3b {
namespace {

using npu::Batch;
using npu::Buffer;
using npu::ExpertArena;

// Expert GEMM rows run between two CPU steps (a group), and the group
// buffers' capacity: the target plus the largest expert, the shared one
// (every token of a pass), padded.
constexpr std::size_t kGroupRows = 16384;

constexpr std::size_t kUp = 2 * kExpertFfn;  // gate + up rows
constexpr std::size_t kUpBfp = BfpBytes(kUp, kHidden);
constexpr std::size_t kDownBfp = BfpBytes(kHidden, kExpertFfn);
constexpr std::size_t kExpertBfp = kUpBfp + kDownBfp;
static_assert(kUpBfp % 64 == 0 && kDownBfp % 64 == 0,
              "buffer windows start on 64 bytes");

class NpuExperts : public ExpertStep {
public:
  NpuExperts(PrefillDevice& device, BurstPool& pool, std::vector<Buffer> arenas,
             std::size_t arena_experts, std::size_t max_tokens)
      : device_(device),
        pool_(pool),
        arenas_(std::move(arenas)),
        arena_experts_(arena_experts),
        capacity_(kGroupRows + max_tokens + kMaxRows) {
    expert_b_.emplace(device_.gemm, arena_experts_ * kExpertBfp);
    a_up_.emplace(device_.gemm, capacity_ * kHidden * sizeof(std::uint16_t));
    c_up_.emplace(device_.gemm, capacity_ * kUp * sizeof(std::uint16_t));
    a_dn_.emplace(device_.gemm, capacity_ * kExpertFfn * sizeof(std::uint16_t));
  }

  // Each expert with tokens, from layer il's arena into its slot of
  // expert_b.
  void Prepare(int il, const ExpertRouting& routing, Batch& dequant) override {
    const auto setup = Clock::now();
    const ExpertArena arena{arena_experts_};
    const Buffer& source = arenas_.at(static_cast<std::size_t>(il));
    for (std::size_t e = 0; e < arena_experts_; ++e) {
      if (routing.tokens[e].empty()) {
        continue;
      }
      const Buffer& up =
          device_.Window(source, arena.UpOffset(e), ExpertArena::kUpBytes);
      const Buffer& down =
          device_.Window(source, arena.DownOffset(e), ExpertArena::kDownBytes);
      const Buffer& up_b = device_.Window(*expert_b_, e * kExpertBfp, kUpBfp);
      const Buffer& down_b =
          device_.Window(*expert_b_, e * kExpertBfp + kUpBfp, kDownBfp);
      dequant.Add(DequantSequence(kHidden, kUp), {&up, &up_b});
      dequant.Add(DequantSequence(kExpertFfn, kHidden), {&down, &down_b});
    }
    device_.profile().setup_seconds += SecondsSince(setup);
  }

  // Every expert with tokens, in groups of about kGroupRows GEMM rows.
  void Run(int, const ExpertRouting& routing, const std::uint16_t* x,
           float* h) override {
    std::vector<Call> group;
    std::size_t rows = 0;
    for (std::size_t e = 0; e < arena_experts_; ++e) {
      const std::size_t tokens = routing.tokens[e].size();
      for (std::size_t first = 0; first < tokens; first += kMaxRows) {
        Call call;
        call.expert = e;
        call.first = first;
        call.rows = std::min(kMaxRows, tokens - first);
        call.m = PaddedRows(call.rows);
        call.row0 = rows;
        rows += call.m;
        group.push_back(call);
      }
      if (rows >= kGroupRows) {
        RunGroup(group, rows, routing, x, h);
        group.clear();
        rows = 0;
      }
    }
    RunGroup(group, rows, routing, x, h);
  }

private:
  // One expert GEMM call: `rows` of expert e's tokens (from index `first` of
  // its list), at row `row0` of the group buffers, `m` rows.
  struct Call {
    std::size_t expert = 0;
    std::size_t first = 0;
    std::size_t rows = 0;
    std::size_t m = 0;
    std::size_t row0 = 0;
  };

  void RunGroup(const std::vector<Call>& group, std::size_t rows,
                const ExpertRouting& routing, const std::uint16_t* x,
                float* h) {
    if (group.empty()) {
      return;
    }
    if (rows > capacity_) {
      throw std::logic_error("qwen36_35b_a3b prefill: expert group overflow");
    }
    auto& profile = device_.profile();
    // Each call's token rows of the normalized input.
    auto start = Clock::now();
    auto* a = a_up_->as<std::uint16_t>();
    pool_.Run(group.size(), [&](std::size_t i) {
      const Call& call = group[i];
      const auto& tokens = routing.tokens[call.expert];
      for (std::size_t r = 0; r < call.rows; ++r) {
        std::memcpy(
            a + (call.row0 + r) * kHidden,
            x + static_cast<std::size_t>(tokens[call.first + r]) * kHidden,
            kHidden * sizeof(std::uint16_t));
      }
    });
    profile.expert_seconds += SecondsSince(start);
    device_.Sync(*a_up_, rows * kHidden * sizeof(std::uint16_t), true);
    auto setup = Clock::now();
    Batch ups(device_.gemm);
    for (const Call& call : group) {
      const std::size_t e = call.expert;
      const Buffer& aw =
          device_.Window(*a_up_, call.row0 * kHidden * 2, call.m * kHidden * 2);
      const Buffer& bw = device_.Window(*expert_b_, e * kExpertBfp, kUpBfp);
      const Buffer& cw =
          device_.Window(*c_up_, call.row0 * kUp * 2, call.m * kUp * 2);
      ups.Add(GemmSequence(call.m, kHidden, kUp), {&aw, &bw, &cw});
    }
    profile.setup_seconds += SecondsSince(setup);
    device_.Run(ups, device_.gemm);
    device_.Sync(*c_up_, rows * kUp * sizeof(std::uint16_t), false);

    // SwiGLU times the expert's weight for the token. The gate + up rows
    // are in q4nx::SwiGluOrder: per 32, 16 gate values then their 16 up.
    start = Clock::now();
    const auto* cu = c_up_->as<std::uint16_t>();
    auto* ad = a_dn_->as<std::uint16_t>();
    pool_.Run(group.size(), [&](std::size_t i) {
      const Call& call = group[i];
      const auto& w = routing.weights[call.expert];
      for (std::size_t r = 0; r < call.rows; ++r) {
        const float scale = w[call.first + r];
        const std::uint16_t* row = cu + (call.row0 + r) * kUp;
        std::uint16_t* dst = ad + (call.row0 + r) * kExpertFfn;
        for (std::size_t j = 0; j < kExpertFfn; ++j) {
          const std::size_t tile = j / 16;
          const std::size_t lane = j % 16;
          const float g = quant::Bf16ToFloat(row[tile * 32 + lane]);
          const float u = quant::Bf16ToFloat(row[tile * 32 + 16 + lane]);
          dst[j] = quant::FloatToBf16(simd::Silu(g) * u * scale);
        }
      }
    });
    profile.expert_seconds += SecondsSince(start);
    device_.Sync(*a_dn_, rows * kExpertFfn * sizeof(std::uint16_t), true);
    setup = Clock::now();
    Batch downs(device_.gemm);
    for (const Call& call : group) {
      const std::size_t e = call.expert;
      const Buffer& aw = device_.Window(*a_dn_, call.row0 * kExpertFfn * 2,
                                        call.m * kExpertFfn * 2);
      const Buffer& bw =
          device_.Window(*expert_b_, e * kExpertBfp + kUpBfp, kDownBfp);
      const Buffer& cw =
          device_.Window(*a_up_, call.row0 * kHidden * 2, call.m * kHidden * 2);
      downs.Add(GemmSequence(call.m, kExpertFfn, kHidden), {&aw, &bw, &cw});
    }
    profile.setup_seconds += SecondsSince(setup);
    device_.Run(downs, device_.gemm);
    device_.Sync(*a_up_, rows * kHidden * sizeof(std::uint16_t), false);

    // The sum into the residual, in expert order (as decode sums them), by
    // blocks of columns so no two jobs write the same values.
    start = Clock::now();
    const auto* cd = a_up_->as<std::uint16_t>();
    constexpr std::size_t kColumns = 128;
    pool_.Run(kHidden / kColumns, [&](std::size_t block) {
      const std::size_t c0 = block * kColumns;
      for (const Call& call : group) {
        const auto& tokens = routing.tokens[call.expert];
        for (std::size_t r = 0; r < call.rows; ++r) {
          float* o =
              h + static_cast<std::size_t>(tokens[call.first + r]) * kHidden +
              c0;
          simd::FromBf16Row(cd + (call.row0 + r) * kHidden + c0, o,
                            static_cast<int>(kColumns), true);
        }
      }
    });
    profile.expert_seconds += SecondsSince(start);
  }

  PrefillDevice& device_;
  BurstPool& pool_;
  std::vector<Buffer> arenas_;
  std::size_t arena_experts_;
  std::size_t capacity_;  // group rows the buffers hold
  // Per arena expert: gate + up, then down, as bfp16 (the GEMMs' B).
  std::optional<Buffer> expert_b_;
  // Group operands: gathered input rows (and the downs' output), gate + up
  // output, SwiGLU output.
  std::optional<Buffer> a_up_, c_up_, a_dn_;
};

}  // namespace

std::unique_ptr<ExpertStep> MakeNpuExperts(PrefillDevice& device,
                                           BurstPool& pool,
                                           std::vector<npu::Buffer> arenas,
                                           std::size_t arena_experts,
                                           std::size_t max_tokens) {
  return std::make_unique<NpuExperts>(device, pool, std::move(arenas),
                                      arena_experts, max_tokens);
}

}  // namespace gufo_npu::qwen36_35b_a3b
