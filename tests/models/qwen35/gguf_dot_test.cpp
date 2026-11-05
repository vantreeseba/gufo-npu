#include "models/qwen35/gguf_dot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "core/gguf.hpp"
#include "core/quant.hpp"

// GgufProjections' fused row dots (gguf_dot.hpp) against quant::Dequantize
// and a double-precision sum, on random rows: every code, scale and
// sub-block position must land on its own weight. Built with the decoders'
// flags (AVX2 path) and with the default flags (portable path).
namespace {

namespace dot = gufo_npu::qwen35::gguf_dot;
using gufo_npu::gguf::GgmlType;
namespace quant = gufo_npu::quant;

// A finite, normal fp16 of either sign, magnitude 2^-10 .. 2^-1.
std::uint16_t RandomHalf(std::mt19937& rng) {
  const auto sign = static_cast<std::uint16_t>((rng() & 1u) << 15);
  const auto exponent = static_cast<std::uint16_t>((5 + rng() % 10) << 10);
  const auto mantissa = static_cast<std::uint16_t>(rng() & 0x3FFu);
  return static_cast<std::uint16_t>(sign | exponent | mantissa);
}

// Byte offsets of a block's fp16 scales: Q8_0's d first, Q4_K's and
// Q5_K's d and dmin first, Q6_K's d last.
std::vector<std::size_t> HalfOffsets(GgmlType type) {
  switch (type) {
    case GgmlType::kQ8_0:
      return {0};
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
      return {0, 2};
    case GgmlType::kQ6_K:
      return {208};
    default:
      return {};
  }
}

// A random row of `n` values of `type`: random codes, scales and mins, each
// block's fp16 scales normal halves; F32 rows standard normal.
std::vector<std::uint8_t> RandomRow(GgmlType type, std::size_t n,
                                    std::mt19937& rng) {
  std::vector<std::uint8_t> row(quant::RowBytes(type, n));
  if (type == GgmlType::kF32) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
      const float v = normal(rng);
      std::memcpy(&row[4 * i], &v, sizeof(v));
    }
    return row;
  }
  for (auto& byte : row) {
    byte = static_cast<std::uint8_t>(rng());
  }
  const std::size_t block_bytes = quant::BlockBytes(type);
  for (std::size_t b = 0; b < row.size(); b += block_bytes) {
    for (const std::size_t at : HalfOffsets(type)) {
      const std::uint16_t half = RandomHalf(rng);
      std::memcpy(&row[b + at], &half, sizeof(half));
    }
  }
  return row;
}

}  // namespace

int main() {
  struct Case {
    const char* name;
    GgmlType type;
    float (*fused)(const std::uint8_t*, const float*, std::size_t);
  };
  const Case cases[] = {
      {"Q8_0", GgmlType::kQ8_0, &dot::DotQ8_0},
      {"Q4_K", GgmlType::kQ4_K, &dot::DotQ4_K},
      {"Q5_K", GgmlType::kQ5_K, &dot::DotQ5_K},
      {"Q6_K", GgmlType::kQ6_K, &dot::DotQ6_K},
      {"F32", GgmlType::kF32, &dot::DotF32},
  };
  std::mt19937 rng(20261105);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  int failures = 0;
  for (const auto& c : cases) {
    // One super-block, an expert down projection's 512 and the hidden 2048.
    for (const std::size_t n :
         {std::size_t{256}, std::size_t{512}, std::size_t{2048}}) {
      double worst = 0;
      for (int trial = 0; trial < 200; ++trial) {
        const auto row = RandomRow(c.type, n, rng);
        std::vector<float> x(n);
        for (auto& v : x) {
          v = normal(rng);
        }
        std::vector<float> w(n);
        quant::Dequantize(c.type, row.data(), w.data(), n);
        double want = 0;
        double magnitude = 0;  // sum of |w x|: the scale of fp32 rounding
        for (std::size_t i = 0; i < n; ++i) {
          want += static_cast<double>(w[i]) * x[i];
          magnitude += std::fabs(static_cast<double>(w[i]) * x[i]);
        }
        const double got = c.fused(row.data(), x.data(), n);
        worst = std::max(worst, std::fabs(got - want) / magnitude);
      }
      // fp32 sums of up to 2048 products round to ~1e-6 of the magnitude;
      // a misplaced code or scale is off by a sizeable fraction of it.
      const bool ok = worst < 2e-5;
      std::printf("%s n=%zu: worst error %.2e of sum |w x| %s\n", c.name, n,
                  worst, ok ? "ok" : "FAIL");
      failures += ok ? 0 : 1;
    }
  }
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d fused dot cases\n", failures);
    return 1;
  }
  std::puts("PASS");
  return 0;
}
