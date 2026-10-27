#include "core/q4nx.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "core/quant.hpp"

// q4nx packing: Q4_K keeps its codes (only scales and minima round to bf16),
// other weights requantize within a 4-bit step, and padding rows are zero.
namespace {

using gufo_npu::gguf::GgmlType;
namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;

int Fail(const char* what) {
  std::fprintf(stderr, "FAIL: %s\n", what);
  return 1;
}

// Random but well-formed Q4_K rows: small positive d and dmin, random 6-bit
// scales/minima and codes.
std::vector<std::uint8_t> RandomQ4K(std::size_t rows, std::size_t dim,
                                    std::mt19937& rng) {
  std::vector<std::uint8_t> data(rows * dim / 256 * 144);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> half(0x1C00, 0x2800);  // ~0.004 .. 0.03
  for (std::size_t b = 0; b < data.size() / 144; ++b) {
    std::uint8_t* blk = data.data() + b * 144;
    const auto d = static_cast<std::uint16_t>(half(rng));
    const auto dmin = static_cast<std::uint16_t>(half(rng));
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &dmin, 2);
    for (std::size_t i = 4; i < 144; ++i) {
      blk[i] = static_cast<std::uint8_t>(byte(rng));
    }
  }
  return data;
}

int CheckQ4KExact() {
  std::mt19937 rng(7);
  constexpr std::size_t kRows = 40;  // not a multiple of the 1024-row round
  constexpr std::size_t kDim = 512;
  const auto data = RandomQ4K(kRows, kDim, rng);
  const auto packed = q4nx::Pack(GgmlType::kQ4_K, data.data(), kRows, kDim);
  if (!packed.exact || packed.padded_rows != q4nx::kOutPerRound) {
    return Fail("Q4_K pack: exact flag or padding");
  }
  if (packed.bytes.size() != packed.padded_rows * kDim * 5 / 8) {
    return Fail("Q4_K pack: byte count");
  }
  const auto w = q4nx::Unpack(packed);
  std::vector<float> ref(kRows * kDim);
  quant::Dequantize(GgmlType::kQ4_K, data.data(), ref.data(), ref.size());
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t k = 0; k < kDim; ++k) {
      // Codes match; scale = d*sc and min = -dmin*m each move by at most half
      // a bf16 ulp (2^-8 relative: bf16 keeps 8 significant bits), so the
      // weight moves by at most 2^-8 * (15 * scale + |min|).
      const std::uint8_t* blk = data.data() + (r * kDim + k) / 256 * 144;
      std::uint16_t d_bits;
      std::uint16_t dmin_bits;
      std::memcpy(&d_bits, blk, 2);
      std::memcpy(&dmin_bits, blk + 2, 2);
      std::uint8_t sc;
      std::uint8_t mn;
      quant::ScaleMinK4(k % 256 / 32, blk + 4, &sc, &mn);
      const float scale = quant::HalfToFloat(d_bits) * sc;
      const float min = quant::HalfToFloat(dmin_bits) * mn;
      const float bound = std::ldexp(15.0f * scale + min, -8) + 1e-7f;
      const float got = w[r * kDim + k];
      const float want = ref[r * kDim + k];
      if (std::fabs(got - want) > bound) {
        std::fprintf(stderr, "FAIL: Q4_K [%zu,%zu]: %g != %g\n", r, k, got,
                     want);
        return 1;
      }
    }
  }
  for (std::size_t i = kRows * kDim; i < w.size(); ++i) {
    if (w[i] != 0.0f) {
      return Fail("Q4_K pack: padding row not zero");
    }
  }
  std::puts("ok: Q4_K packs with its codes");
  return 0;
}

int CheckRequant() {
  std::mt19937 rng(11);
  std::normal_distribution<float> normal(0.0f, 0.02f);
  constexpr std::size_t kRows = 1024 + 96;
  constexpr std::size_t kDim = 768;
  std::vector<float> w(kRows * kDim);
  for (auto& v : w) {
    v = normal(rng);
  }
  const auto packed = q4nx::PackFloat(w.data(), kRows, kDim);
  if (packed.exact || packed.padded_rows != 2 * q4nx::kOutPerRound) {
    return Fail("requant: exact flag or padding");
  }
  const auto u = q4nx::Unpack(packed);
  double err2 = 0;
  double ref2 = 0;
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t g = 0; g < kDim / q4nx::kGroup; ++g) {
      const float* src = w.data() + r * kDim + g * q4nx::kGroup;
      float lo = src[0];
      float hi = src[0];
      for (std::size_t l = 0; l < q4nx::kGroup; ++l) {
        lo = std::fmin(lo, src[l]);
        hi = std::fmax(hi, src[l]);
      }
      // Within half a step, plus the bf16 rounding of min and scale.
      const float step = (hi - lo) / 15.0f;
      const float bound = 0.5f * step * 1.05f + std::ldexp(std::fabs(lo), -7);
      for (std::size_t l = 0; l < q4nx::kGroup; ++l) {
        const std::size_t i = r * kDim + g * q4nx::kGroup + l;
        const double e = u[i] - w[i];
        if (std::fabs(e) > bound) {
          std::fprintf(stderr, "FAIL: requant [%zu]: %g vs %g (bound %g)\n", i,
                       u[i], w[i], bound);
          return 1;
        }
        err2 += e * e;
        ref2 += static_cast<double>(w[i]) * w[i];
      }
    }
  }
  std::printf("ok: requant within half a step (rel rms %.4f)\n",
              std::sqrt(err2 / ref2));
  return 0;
}

int CheckReference() {
  std::mt19937 rng(3);
  constexpr std::size_t kRows = 64;
  constexpr std::size_t kDim = 256;
  const auto data = RandomQ4K(kRows, kDim, rng);
  const auto packed = q4nx::Pack(GgmlType::kQ4_K, data.data(), kRows, kDim);
  const auto u = q4nx::Unpack(packed);
  std::vector<float> x(kDim);
  for (std::size_t k = 0; k < kDim; ++k) {
    x[k] = static_cast<float>(k % 7) - 3.0f;
  }
  const auto y = q4nx::Reference(u, packed, x.data());
  for (std::size_t r = 0; r < kRows; ++r) {
    double acc = 0;
    for (std::size_t k = 0; k < kDim; ++k) {
      acc += static_cast<double>(u[r * kDim + k]) * x[k];
    }
    if (y[r] != acc) {
      return Fail("reference GEMV");
    }
  }
  std::puts("ok: reference GEMV");
  return 0;
}

int CheckBf16() {
  struct Case {
    float value;
    std::uint16_t bits;
  };
  const Case cases[] = {{1.0f, 0x3F80},
                        {-2.0f, 0xC000},
                        {1.00390625f, 0x3F80},  // tie, rounds to even
                        {1.01171875f, 0x3F82},  // tie, rounds to even
                        {1.0078125f, 0x3F81}};
  for (const auto& c : cases) {
    if (quant::FloatToBf16(c.value) != c.bits) {
      std::fprintf(stderr, "FAIL: bf16 of %g: 0x%04x\n", c.value,
                   quant::FloatToBf16(c.value));
      return 1;
    }
  }
  std::puts("ok: bf16 rounding");
  return 0;
}

}  // namespace

int main() {
  int failures = CheckBf16();
  failures += CheckQ4KExact();
  failures += CheckRequant();
  failures += CheckReference();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
