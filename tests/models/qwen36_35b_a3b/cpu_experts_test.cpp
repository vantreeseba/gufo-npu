#include "models/qwen36_35b_a3b/cpu_experts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "core/burst_pool.hpp"
#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/gemv.hpp"

// The CPU expert step of Qwen3.6-35B-A3B's prefill
// (src/models/qwen36_35b_a3b/cpu_experts.hpp) on random q4nx experts in the
// decoder's arena layout (8 routed experts and a shared one), random routing
// and bf16 activations, against a double-precision computation over the same
// q4nx weights (SwiGLU unrounded): the experts' contribution to the residual
// within 1e-2 relative error per token, for the portable path and, when the
// CPU has it, the AVX-512 BF16 one. Repeated runs are bit-identical. Needs no
// NPU.
namespace {

namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
namespace moe = gufo_npu::qwen36_35b_a3b;
using gufo_npu::BurstPool;
using gufo_npu::gguf::GgmlType;
using gufo_npu::npu::ExpertArena;

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kFfn = 512;
constexpr std::size_t kRouted = 8;
constexpr std::size_t kExperts = kRouted + 1;  // the shared one last
constexpr std::size_t kTokens = 37;
constexpr double kMaxError = 1e-2;

struct Expert {
  q4nx::Matrix up, down;
  std::vector<float> up_w, down_w;  // the q4nx weights, row-major
};

}  // namespace

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> weight(0.0f, 0.03f);
  std::normal_distribution<float> act(0.0f, 1.0f);
  std::uniform_real_distribution<float> unit(0.05f, 1.0f);

  const ExpertArena layout{kExperts};
  std::vector<std::uint8_t> arena(layout.bytes());
  std::vector<Expert> experts(kExperts);
  const auto order = q4nx::SwiGluOrder(kFfn);
  for (std::size_t e = 0; e < kExperts; ++e) {
    std::vector<float> gate(kFfn * kHidden), up(kFfn * kHidden),
        down(kHidden * kFfn);
    for (auto* v : {&gate, &up, &down}) {
      for (float& w : *v) {
        w = weight(rng);
      }
    }
    Expert& x = experts[e];
    x.up = q4nx::PackRows(
        {{GgmlType::kF32, reinterpret_cast<const std::uint8_t*>(gate.data()),
          kFfn},
         {GgmlType::kF32, reinterpret_cast<const std::uint8_t*>(up.data()),
          kFfn}},
        kHidden, order);
    x.down = q4nx::PackFloat(down.data(), kHidden, kFfn);
    if (x.up.bytes.size() != ExpertArena::kUpBytes ||
        x.down.bytes.size() != ExpertArena::kDownBytes) {
      std::fprintf(stderr, "FAIL: experts pack to the wrong size\n");
      return EXIT_FAILURE;
    }
    std::memcpy(arena.data() + layout.UpOffset(e), x.up.bytes.data(),
                x.up.bytes.size());
    std::memcpy(arena.data() + layout.DownOffset(e), x.down.bytes.data(),
                x.down.bytes.size());
    x.up_w = q4nx::Unpack(x.up);
    x.down_w = q4nx::Unpack(x.down);
  }

  // Routing: each token takes 2 to 4 of the routed experts and the shared
  // one; expert 5 takes none.
  std::vector<std::vector<int>> tokens(kExperts);
  std::vector<std::vector<float>> weights(kExperts);
  for (std::size_t t = 0; t < kTokens; ++t) {
    for (std::size_t e = 0; e < kRouted; ++e) {
      if (e != 5 && (t * 7 + e * 3) % 5 < 2) {
        tokens[e].push_back(static_cast<int>(t));
        weights[e].push_back(unit(rng));
      }
    }
    tokens[kRouted].push_back(static_cast<int>(t));
    weights[kRouted].push_back(unit(rng));
  }
  const moe::ExpertRouting routing{tokens, weights};

  std::vector<std::uint16_t> x(kTokens * kHidden);
  std::vector<float> h0(kTokens * kHidden);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = quant::FloatToBf16(act(rng));
    h0[i] = act(rng);
  }

  // Reference: each expert's down(silu(gate) * up * weight) in double.
  std::vector<double> want(kTokens * kHidden, 0.0);
  for (std::size_t e = 0; e < kExperts; ++e) {
    const Expert& ex = experts[e];
    for (std::size_t i = 0; i < tokens[e].size(); ++i) {
      const std::size_t t = static_cast<std::size_t>(tokens[e][i]);
      std::vector<double> a(kFfn);
      for (std::size_t f = 0; f < kFfn; ++f) {
        const std::size_t g_row = (f / 16) * 32 + f % 16;
        const std::size_t u_row = g_row + 16;
        double g = 0;
        double u = 0;
        for (std::size_t k = 0; k < kHidden; ++k) {
          const double xv = quant::Bf16ToFloat(x[t * kHidden + k]);
          g += ex.up_w[g_row * kHidden + k] * xv;
          u += ex.up_w[u_row * kHidden + k] * xv;
        }
        a[f] = g / (1.0 + std::exp(-g)) * u * weights[e][i];
      }
      for (std::size_t r = 0; r < kHidden; ++r) {
        double y = 0;
        for (std::size_t f = 0; f < kFfn; ++f) {
          y += ex.down_w[r * kFfn + f] * a[f];
        }
        want[t * kHidden + r] += y;
      }
    }
  }

  BurstPool pool(3);
  const BurstPool::Burst burst(pool);
  bool ok = true;
  auto check = [&](bool portable, const char* name) {
    auto step =
        moe::MakeCpuExperts(pool, {arena.data()}, kExperts, kTokens, portable);
    std::vector<float> h = h0;
    step->Run(0, routing, x.data(), h.data());
    double worst = 0;
    for (std::size_t t = 0; t < kTokens; ++t) {
      double err = 0;
      double ref = 0;
      for (std::size_t r = 0; r < kHidden; ++r) {
        const std::size_t i = t * kHidden + r;
        const double got = static_cast<double>(h[i]) - h0[i];
        err += (got - want[i]) * (got - want[i]);
        ref += want[i] * want[i];
      }
      worst = std::max(worst, std::sqrt(err / ref));
    }
    std::vector<float> again = h0;
    step->Run(0, routing, x.data(), again.data());
    const bool same =
        std::memcmp(h.data(), again.data(), h.size() * sizeof(float)) == 0;
    const bool pass = worst < kMaxError && same;
    std::printf("%s %-9s worst token rel. err %.2e (limit %.0e), repeat %s\n",
                pass ? "ok  " : "FAIL", name, worst, kMaxError,
                same ? "bit-identical" : "DIFFERS");
    ok = ok && pass;
  };
  check(true, "portable");
  if (moe::CpuExpertsFast()) {
    check(false, "avx512");
  } else {
    std::puts("note: no AVX-512 BF16 here; the fast path is not tested");
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
