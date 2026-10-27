#include "core/q4nx.hpp"

#include <algorithm>
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
  if (packed.bytes.size() != packed.padded_rows / q4nx::kMTile *
                                 (kDim / q4nx::kKTile) * q4nx::kBlockBytes) {
    return Fail("Q4_K pack: byte count");
  }
  const auto w = q4nx::Unpack(packed);
  std::vector<float> ref(kRows * kDim);
  quant::Dequantize(GgmlType::kQ4_K, data.data(), ref.data(), ref.size());
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t k = 0; k < kDim; ++k) {
      // Codes, sub-scales and sub-minima match; d and dmin each move by at
      // most half a bf16 ulp (2^-8 relative: bf16 keeps 8 significant bits),
      // so the weight moves by at most 2^-8 * (15 * scale + |min|).
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
  constexpr std::size_t kGroups = q4nx::kKTile / q4nx::kGroup;
  constexpr float kUp = 1.0f + 1.0f / 128;  // bf16 rounded up: <= 2^-7 above
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t kb = 0; kb < kDim / q4nx::kKTile; ++kb) {
      const float* src = w.data() + r * kDim + kb * q4nx::kKTile;
      float lo[kGroups];
      float hi[kGroups];
      float max_abs_min = 0;
      for (std::size_t g = 0; g < kGroups; ++g) {
        lo[g] = *std::min_element(src + g * q4nx::kGroup,
                                  src + (g + 1) * q4nx::kGroup);
        hi[g] = *std::max_element(src + g * q4nx::kGroup,
                                  src + (g + 1) * q4nx::kGroup);
        max_abs_min = std::fmax(max_abs_min, std::fabs(lo[g]));
      }
      // A min/max fit would put every weight within half of its group's step,
      // (hi - lo + dmin) / 15 rounded up by at most d. The packer trades the
      // extremes for a lower error overall (it may clip a weight at either
      // end), but no weight may be off by more than that whole step.
      const float dmin = max_abs_min / 127.0f * kUp;
      float max_need = 0;
      for (std::size_t g = 0; g < kGroups; ++g) {
        max_need = std::fmax(max_need, (hi[g] - lo[g] + dmin) / 15.0f);
      }
      const float d = max_need / 255.0f * kUp;
      for (std::size_t g = 0; g < kGroups; ++g) {
        const float bound =
            ((hi[g] - lo[g] + dmin) / 15.0f + d) * 1.0001f + 1e-9f;
        for (std::size_t l = 0; l < q4nx::kGroup; ++l) {
          const std::size_t i =
              r * kDim + kb * q4nx::kKTile + g * q4nx::kGroup + l;
          const double e = u[i] - w[i];
          if (std::fabs(e) > bound) {
            std::fprintf(stderr, "FAIL: requant [%zu]: %g vs %g (bound %g)\n",
                         i, u[i], w[i], bound);
            return 1;
          }
          err2 += e * e;
          ref2 += static_cast<double>(w[i]) * w[i];
        }
      }
    }
  }
  // The min/max fit gave 0.0787 on these weights.
  constexpr double kMaxRelRms = 0.074;
  const double rel_rms = std::sqrt(err2 / ref2);
  if (rel_rms > kMaxRelRms) {
    std::fprintf(stderr, "FAIL: requant rel rms %.4f > %.4f\n", rel_rms,
                 kMaxRelRms);
    return 1;
  }
  std::printf("ok: requant within a step, rel rms %.4f (min/max fit 0.0787)\n",
              rel_rms);
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
  std::vector<float> fast(kRows);
  q4nx::Gemv(packed, x.data(), fast.data());
  for (std::size_t r = 0; r < kRows; ++r) {
    if (std::fabs(fast[r] - y[r]) > 1e-4 * (1 + std::fabs(y[r]))) {
      return Fail("Gemv differs from the reference");
    }
  }
  std::puts("ok: reference GEMV and Gemv");
  return 0;
}

int CheckPackRows() {
  std::mt19937 rng(5);
  constexpr std::size_t kDim = 256;
  const auto a = RandomQ4K(40, kDim, rng);
  std::vector<float> fb(24 * kDim);
  std::normal_distribution<float> normal(0.0f, 0.05f);
  for (auto& v : fb) {
    v = normal(rng);
  }
  std::vector<std::uint16_t> b(fb.size());
  for (std::size_t i = 0; i < fb.size(); ++i) {
    b[i] = quant::FloatToBf16(fb[i]);
  }
  const auto* b_bytes = reinterpret_cast<const std::uint8_t*>(b.data());
  const auto fused = q4nx::PackRows(
      {{GgmlType::kQ4_K, a.data(), 40}, {GgmlType::kBF16, b_bytes, 24}}, kDim);
  const auto alone_a = q4nx::Pack(GgmlType::kQ4_K, a.data(), 40, kDim);
  const auto alone_b = q4nx::Pack(GgmlType::kBF16, b_bytes, 24, kDim);
  if (fused.rows != 64 || fused.exact) {
    return Fail("PackRows: rows or exact flag");
  }
  const auto uf = q4nx::Unpack(fused);
  const auto ua = q4nx::Unpack(alone_a);
  const auto ub = q4nx::Unpack(alone_b);
  if (!std::equal(ua.begin(), ua.begin() + 40 * kDim, uf.begin()) ||
      !std::equal(ub.begin(), ub.begin() + 24 * kDim, uf.begin() + 40 * kDim)) {
    return Fail("PackRows: fused rows differ from packing each alone");
  }
  std::puts("ok: PackRows stacks segments");

  // SwiGLU order: packed tile t = gate rows 16t.. then up rows 16t...
  const auto order = q4nx::SwiGluOrder(64);
  if (order.size() != 128 || order[0] != 0 || order[15] != 15 ||
      order[16] != 64 || order[31] != 79 || order[32] != 16 ||
      order[48] != 80) {
    return Fail("SwiGluOrder");
  }
  const auto ordered = q4nx::PackRows(
      {{GgmlType::kQ4_K, a.data(), 40}, {GgmlType::kBF16, b_bytes, 24}}, kDim,
      [] {
        std::vector<std::size_t> o(64);
        for (std::size_t i = 0; i < 64; ++i) {
          o[i] = 63 - i;
        }
        return o;
      }());
  const auto uo = q4nx::Unpack(ordered);
  for (std::size_t i = 0; i < 64; ++i) {
    if (!std::equal(uo.begin() + i * kDim, uo.begin() + (i + 1) * kDim,
                    uf.begin() + (63 - i) * kDim)) {
      return Fail("PackRows: order not applied");
    }
  }
  std::puts("ok: PackRows row order and SwiGluOrder");
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
  failures += CheckPackRows();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
