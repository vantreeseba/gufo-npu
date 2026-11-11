// The ViT attention program's host side without an NPU (npu/
// vit_attention.hpp): its lengths, and the bfp16 hi/lo split of a tile.
// hi + lo must give the tile back to within 2^-14 of each row's largest
// value, and the AVX-512 split must give the scalar encoder's bytes
// (bfp16::EncodeBlock), on random tiles, rows that round up to the
// saturating mantissa, and zeros. Hosted CI runs it; the AVX-512 half runs
// where the CPU has AVX-512.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/bfp16.hpp"
#include "models/qwen3vl_embed_2b/npu/vit_attention.hpp"

namespace {

namespace npu = gufo_npu::qwen3vl_embed_2b::npu;
namespace bfp16 = gufo_npu::bfp16;

int failures = 0;

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

// The tile hi + lo encode.
std::vector<float> Decode(const std::uint8_t* split) {
  std::vector<float> out(64);
  for (int r = 0; r < 8; ++r) {
    float hi[8], lo[8];
    bfp16::DecodeBlock(split + r * 9, hi);
    bfp16::DecodeBlock(split + npu::kVitAttnTileBytes + r * 9, lo);
    for (int c = 0; c < 8; ++c) {
      out[r * 8 + c] = hi[c] + lo[c];
    }
  }
  return out;
}

void CheckTile(const float* tile, const char* what) {
  std::uint8_t scalar[2 * npu::kVitAttnTileBytes];
  npu::detail::SplitTile(tile, scalar, false);
  const auto back = Decode(scalar);
  for (int r = 0; r < 8; ++r) {
    float peak = 0;
    for (int c = 0; c < 8; ++c) {
      peak = std::max(peak, std::fabs(tile[r * 8 + c]));
    }
    for (int c = 0; c < 8; ++c) {
      if (std::fabs(back[r * 8 + c] - tile[r * 8 + c]) >
          std::ldexp(peak, -14)) {
        std::fprintf(stderr, "FAIL: %s: row %d lane %d: %g back as %g\n", what,
                     r, c, tile[r * 8 + c], back[r * 8 + c]);
        ++failures;
        return;
      }
    }
  }
  if (npu::detail::HasAvx512()) {
    std::uint8_t avx[2 * npu::kVitAttnTileBytes];
    npu::detail::SplitTile(tile, avx, true);
    if (std::memcmp(avx, scalar, sizeof(avx)) != 0) {
      std::fprintf(stderr,
                   "FAIL: %s: the AVX-512 split differs from the scalar one\n",
                   what);
      ++failures;
    }
  }
}

}  // namespace

int main() {
  Expect(npu::VitAttnLength(1) == 256, "1 patch pads to 256");
  Expect(npu::VitAttnLength(256) == 256, "256 patches stay 256");
  Expect(npu::VitAttnLength(257) == 512, "257 patches pad to 512");
  Expect(npu::VitAttnLength(npu::kVitAttnMaxPatches) == 7424,
         "the cap pads to 7424");
  for (const int bad : {0, npu::kVitAttnMaxPatches + 1}) {
    bool threw = false;
    try {
      npu::VitAttnLength(bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Expect(threw, "a length outside the program's range throws");
  }
  Expect(npu::VitAttnSequence(1280) == "qwen3vl_embed_2b_vit_attn_n1280",
         "sequence names");

  std::mt19937 rng(5);
  float tile[64];
  for (int t = 0; t < 2000; ++t) {
    // Rows at different scales, some with one large value.
    std::normal_distribution<float> g(0.0f, std::ldexp(1.0f, t % 40 - 20));
    for (float& x : tile) {
      x = g(rng);
    }
    if (t % 3 == 0) {
      tile[t % 64] *= 300.0f;
    }
    CheckTile(tile, "random tile");
  }
  // Values that round to mantissa 128 saturate to 127; an all-zero tile.
  for (int i = 0; i < 64; ++i) {
    tile[i] = i % 8 == 0 ? 1.999f : (i % 2 ? -1.0f : 1.9961f);
  }
  CheckTile(tile, "saturating rows");
  std::fill(std::begin(tile), std::end(tile), 0.0f);
  CheckTile(tile, "zeros");
  if (!npu::detail::HasAvx512()) {
    std::puts("note: no AVX-512, the scalar split only");
  }
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS: ViT attention lengths and tile splits");
  return EXIT_SUCCESS;
}
