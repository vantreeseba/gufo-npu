#include "core/bfp16.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// bfp16ebs8 encoding (round to nearest even, as a core in conv_even mode and
// IRON's weight packer do) and the prefill GEMM's B layout (IRON's pack_b
// order, rebuilt here directly from its transpose).
namespace {

namespace bfp16 = gufo_npu::bfp16;

int Fail(const char* what) {
  std::fprintf(stderr, "FAIL: %s\n", what);
  return 1;
}

// The block's largest value sets the exponent (127 -> 2^6, one LSB = 1), so
// the others round to integers: ties go to even, as IRON measured on the
// device (106.5 -> 106, 94.5 -> 94, 14.9375 -> 15).
int CheckRounding() {
  const float in[8] = {127.0f, 106.5f, 94.5f, 14.9375f,
                       -3.5f,  -0.5f,  1.5f,  -127.6f};
  const float want[8] = {127, 106, 94, 15, -4, 0, 2, -128};
  std::uint8_t block[bfp16::kBlockBytes];
  bfp16::EncodeBlock(in, block);
  if (block[0] != 127 + 6) {
    return Fail("rounding: shared exponent");
  }
  float out[8];
  bfp16::DecodeBlock(block, out);
  for (int i = 0; i < 8; ++i) {
    if (out[i] != want[i]) {
      std::fprintf(stderr, "FAIL: rounding [%d]: %g -> %g, want %g\n", i, in[i],
                   out[i], want[i]);
      return 1;
    }
  }
  // +127.6 would round to 128, which the mantissa cannot hold: saturate.
  const float big[8] = {127.6f, 1, 1, 1, 1, 1, 1, 1};
  bfp16::EncodeBlock(big, block);
  bfp16::DecodeBlock(block, out);
  if (out[0] != 127.0f) {
    return Fail("rounding: +128 saturates to 127");
  }
  return 0;
}

// Random blocks: every value within half an LSB of the block's grid.
int CheckRoundTrip() {
  std::mt19937 rng(3);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::uniform_int_distribution<int> exp_shift(-20, 20);
  for (int b = 0; b < 20000; ++b) {
    float in[8];
    const float scale = std::ldexp(1.0f, exp_shift(rng));
    for (float& v : in) {
      v = dist(rng) * scale;
    }
    std::uint8_t block[bfp16::kBlockBytes];
    bfp16::EncodeBlock(in, block);
    float out[8];
    bfp16::DecodeBlock(block, out);
    const double lsb = std::ldexp(1.0, block[0] - 133);
    for (int i = 0; i < 8; ++i) {
      // Saturation at +127 can cost up to a whole LSB.
      if (std::fabs(static_cast<double>(out[i]) - in[i]) > lsb) {
        std::fprintf(stderr, "FAIL: round trip %g -> %g (lsb %g)\n", in[i],
                     out[i], lsb);
        return 1;
      }
    }
  }
  return 0;
}

// pack_b's reshape (kb, ks, i, s, cb, tb, t) -> transpose (cb, kb, ks, tb,
// i, t, s), emitted block by block.
std::vector<std::uint8_t> ReferencePack(const std::vector<float>& w,
                                        std::size_t rows, std::size_t dim) {
  std::vector<std::uint8_t> out;
  const std::size_t col_a = bfp16::kCtK / 8;
  for (std::size_t cb = 0; cb < rows / bfp16::kNTile; ++cb) {
    for (std::size_t kb = 0; kb < dim / bfp16::kKTile; ++kb) {
      for (std::size_t ks = 0; ks < bfp16::kKTile / bfp16::kCtK; ++ks) {
        for (std::size_t tb = 0; tb < bfp16::kNTile / 8; ++tb) {
          for (std::size_t i = 0; i < col_a; ++i) {
            for (std::size_t t = 0; t < 8; ++t) {
              float values[8];
              for (std::size_t s = 0; s < 8; ++s) {
                const std::size_t k =
                    kb * bfp16::kKTile + ks * bfp16::kCtK + i * 8 + s;
                const std::size_t n = cb * bfp16::kNTile + tb * 8 + t;
                values[s] = w[n * dim + k];  // B[k][n] = W[n][k]
              }
              std::uint8_t block[bfp16::kBlockBytes];
              bfp16::EncodeBlock(values, block);
              out.insert(out.end(), block, block + bfp16::kBlockBytes);
            }
          }
        }
      }
    }
  }
  return out;
}

int CheckLayout() {
  constexpr std::size_t kRows = 192;
  constexpr std::size_t kDim = 1024;
  std::mt19937 rng(5);
  std::normal_distribution<float> dist(0.0f, 0.05f);
  std::vector<float> w(kRows * kDim);
  for (auto& v : w) {
    v = dist(rng);
  }
  const auto packed = bfp16::PackGemmB(w.data(), kRows, kDim);
  if (packed.size() != bfp16::PackedBytes(kRows, kDim) ||
      packed != ReferencePack(w, kRows, kDim)) {
    return Fail("layout: PackGemmB differs from pack_b's order");
  }
  const auto back = bfp16::UnpackGemmB(packed, kRows, kDim);
  double diff = 0;
  double norm = 0;
  for (std::size_t i = 0; i < w.size(); ++i) {
    diff += (back[i] - w[i]) * static_cast<double>(back[i] - w[i]);
    norm += w[i] * static_cast<double>(w[i]);
  }
  // 7 magnitude bits per shared exponent: a few 1e-3 on Gaussian values.
  if (std::sqrt(diff / norm) > 1e-2) {
    return Fail("layout: unpacked weights too far from the input");
  }
  return 0;
}

}  // namespace

int main() {
  int failures = CheckRounding() + CheckRoundTrip() + CheckLayout();
  if (failures == 0) {
    std::puts("bfp16_test: ok");
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
