#include "core/vec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "core/quant.hpp"

// The decode loop's fp32 and fp16 helpers against plain scalar math: fp16
// rounds to nearest even and agrees with quant::HalfToFloat, and the vector
// paths (F16C/AVX2 when the build has them) match a double-precision sum.
namespace {

namespace vec = gufo_npu::vec;

int Fail(const char* what) {
  std::fprintf(stderr, "FAIL: %s\n", what);
  return 1;
}

double RelErr(double got, double want) {
  return std::fabs(got - want) / std::max(std::fabs(want), 1e-30);
}

}  // namespace

int main() {
  // Exact values and ties: 1 + 2^-11 is halfway between 1 and the next
  // half (1 + 2^-10) and rounds to even (1); 1 + 3 * 2^-11 rounds up.
  struct Case {
    float in;
    std::uint16_t bits;
  };
  const Case cases[] = {
      {0.0f, 0x0000},
      {1.0f, 0x3C00},
      {-2.0f, 0xC000},
      {65504.0f, 0x7BFF},
      {1.0f + 0x1p-11f, 0x3C00},
      {1.0f + 3 * 0x1p-11f, 0x3C02},
      {0x1p-24f, 0x0001},
      {1e6f, 0x7C00},
  };
  for (const auto& c : cases) {
    if (vec::ToHalf(c.in) != c.bits) {
      std::fprintf(stderr, "ToHalf(%g) = 0x%04X, want 0x%04X\n", c.in,
                   vec::ToHalf(c.in), c.bits);
      return Fail("fp16 rounding");
    }
  }
  for (std::uint32_t h = 0; h < 0x7C00; ++h) {  // every finite non-negative
    const auto bits = static_cast<std::uint16_t>(h);
    if (vec::FromHalf(bits) != gufo_npu::quant::HalfToFloat(bits) ||
        vec::ToHalf(vec::FromHalf(bits)) != bits) {
      return Fail("fp16 round trip");
    }
  }

  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 3.0f);
  for (const int n : {1, 7, 8, 15, 16, 17, 256, 2048, 2051}) {
    std::vector<float> a(n);
    std::vector<float> b(n);
    for (int i = 0; i < n; ++i) {
      a[i] = normal(rng);
      b[i] = normal(rng);
    }
    std::vector<std::uint16_t> bh(n);
    vec::ToHalf(b.data(), bh.data(), n);
    double dot = 0;
    double dot_half = 0;
    double norm = 0;
    for (int i = 0; i < n; ++i) {
      if (bh[i] != vec::ToHalf(b[i])) {
        return Fail("vector ToHalf differs from scalar");
      }
      dot += static_cast<double>(a[i]) * b[i];
      dot_half += static_cast<double>(a[i]) * vec::FromHalf(bh[i]);
      norm += std::fabs(static_cast<double>(a[i]) * b[i]);
    }
    // Float sums carry ~1e-7 relative to the sum of |terms|.
    if (std::fabs(vec::Dot(a.data(), b.data(), n) - dot) > 1e-5 * norm) {
      return Fail("Dot");
    }
    if (std::fabs(vec::DotHalf(a.data(), bh.data(), n) - dot_half) >
        1e-5 * norm) {
      return Fail("DotHalf");
    }
    std::vector<float> y(a);
    vec::AddScaledHalf(0.5f, bh.data(), y.data(), n);
    for (int i = 0; i < n; ++i) {
      const double want = a[i] + 0.5 * vec::FromHalf(bh[i]);
      if (RelErr(y[i], want) > 1e-6 && std::fabs(y[i] - want) > 1e-6) {
        return Fail("AddScaledHalf");
      }
    }
  }
  std::puts(
      "ok   vec: fp16 rounding and round trip, Dot, DotHalf, "
      "AddScaledHalf");
  return EXIT_SUCCESS;
}
