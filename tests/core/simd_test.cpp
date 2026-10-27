#include "core/simd.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "core/quant.hpp"

// The prefill's row math (built with AVX2 + FMA, as the prefill is) against
// scalar references: exp and SiLU within a few float ulps, bf16 conversions
// bit for bit with quant::FloatToBf16, and the tails past the last 8 lanes.
namespace {

namespace simd = gufo_npu::simd;
namespace quant = gufo_npu::quant;

int Fail(const char* what, double got, double want) {
  std::fprintf(stderr, "FAIL: %s: %.9g, want %.9g\n", what, got, want);
  return 1;
}

bool Close(double got, double want, double rel) {
  return std::fabs(got - want) <= rel * std::fabs(want) + 1e-30;
}

int CheckExp() {
  std::vector<float> x;
  for (float v = -87.0f; v <= 88.0f; v += 0.01173f) {
    x.push_back(v);
  }
  std::vector<float> out(x.size());
  const float total = simd::ExpShiftSum(x.data(), 10.0f, out.data(),
                                        static_cast<int>(x.size()));
  double want_total = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double want = std::exp(static_cast<double>(x[i]) - 10.0);
    want_total += want;
    if (!Close(out[i], want, 1e-6)) {
      return Fail("exp", out[i], want);
    }
  }
  if (!Close(total, want_total, 1e-5)) {
    return Fail("exp sum", total, want_total);
  }
  return 0;
}

int CheckRows(std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 4.0f);
  constexpr int kN = 203;  // 25 vectors and a tail of 3
  std::vector<float> x(kN), y(kN), z(kN), out(kN), w(kN);
  std::vector<std::uint16_t> g(kN), u(kN), bits(kN);
  for (int i = 0; i < kN; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
    z[i] = dist(rng);
    w[i] = 1.0f + 0.1f * dist(rng);
    g[i] = quant::FloatToBf16(dist(rng));
    u[i] = quant::FloatToBf16(dist(rng));
  }
  simd::SiluRow(x.data(), out.data(), kN);
  for (int i = 0; i < kN; ++i) {
    const double want = x[i] / (1.0 + std::exp(-static_cast<double>(x[i])));
    if (!Close(out[i], want, 1e-6)) {
      return Fail("silu", out[i], want);
    }
  }
  std::vector<float> yy = y;
  simd::MulSiluRow(yy.data(), z.data(), kN);
  for (int i = 0; i < kN; ++i) {
    const double want =
        y[i] * (z[i] / (1.0 + std::exp(-static_cast<double>(z[i]))));
    if (!Close(yy[i], want, 1e-6)) {
      return Fail("y * silu(z)", yy[i], want);
    }
  }
  // bf16 rounding is exact: the same bits as the scalar converter.
  simd::ToBf16Row(x.data(), bits.data(), kN);
  for (int i = 0; i < kN; ++i) {
    if (bits[i] != quant::FloatToBf16(x[i])) {
      return Fail("bf16", bits[i], quant::FloatToBf16(x[i]));
    }
  }
  simd::ScaleToBf16Row(x.data(), 0.37f, w.data(), bits.data(), kN);
  for (int i = 0; i < kN; ++i) {
    if (bits[i] != quant::FloatToBf16(x[i] * 0.37f * w[i])) {
      return Fail("scaled bf16", bits[i],
                  quant::FloatToBf16(x[i] * 0.37f * w[i]));
    }
  }
  std::vector<float> acc = x;
  simd::FromBf16Row(g.data(), acc.data(), kN, true);
  for (int i = 0; i < kN; ++i) {
    if (acc[i] != x[i] + quant::Bf16ToFloat(g[i])) {
      return Fail("bf16 add", acc[i], x[i] + quant::Bf16ToFloat(g[i]));
    }
  }
  // SwiGLU rounds once to bf16: within half a bf16 step of the exact value
  // (plus exp's ulps).
  simd::SwiGluBf16Row(g.data(), u.data(), bits.data(), kN);
  for (int i = 0; i < kN; ++i) {
    const double gv = quant::Bf16ToFloat(g[i]);
    const double want = gv / (1.0 + std::exp(-gv)) * quant::Bf16ToFloat(u[i]);
    if (!Close(quant::Bf16ToFloat(bits[i]), want, 1.0 / 256)) {
      return Fail("swiglu", quant::Bf16ToFloat(bits[i]), want);
    }
  }
  return 0;
}

}  // namespace

int main() {
  std::mt19937 rng(9);
  const int failures = CheckExp() + CheckRows(rng);
  if (failures == 0) {
    std::puts("simd_test: ok");
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
