#include "models/qwen36_35b_a3b/cpu_gemv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "core/q4nx.hpp"
#include "core/simd.hpp"
#include "models/qwen36_35b_a3b/q4nx_tile.hpp"

#if defined(__x86_64__)
#include <immintrin.h>
#define GUFO_CPU_GEMV_AVX512 1
#endif

namespace gufo_npu::qwen36_35b_a3b {
namespace {

using q4nx_tile::Bf16At;
using q4nx_tile::BlockAt;
using q4nx_tile::kCodes;
using q4nx_tile::kD;
using q4nx_tile::kDmin;
using q4nx_tile::kGroups;
using q4nx_tile::kHalf;
using q4nx_tile::kKTile;
using q4nx_tile::kMn;
using q4nx_tile::kSc;
using q4nx_tile::kTile;

constexpr std::size_t kGroup = q4nx::kGroup;
constexpr int kTokens = CpuGemv::kMaxTokens;

// The inputs of one call: per token its bf16 x and the fp32 sums of each
// 32 of them.
struct Inputs {
  const std::uint16_t* x[kTokens];
  const float* sums[kTokens];
  int tokens;
};

// Code of row m (of the tile), group g, input l (of the group) in `block`.
int CodeAt(const std::uint8_t* block, std::size_t m, std::size_t g,
           std::size_t l) {
  const std::size_t i = (((m / 16) * kGroups + g) * 32 + l) * 16 + m % 16;
  const std::uint8_t byte = block[kCodes + i / 2];
  return (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
}

// Row tile `tile`'s kTile outputs per token into out[t * kTile + m], in the
// fast path's order: per block and half, per group the code dot products
// over input pairs (odd input first, as VDPBF16PS adds them), then the
// sub-scale and sub-minimum, then per block d and dmin.
void TilePortable(const std::uint8_t* matrix, std::size_t k_blocks,
                  std::size_t tile, const Inputs& in, float* out) {
  float row[kTokens][kTile] = {};
  for (std::size_t kb = 0; kb < k_blocks; ++kb) {
    const std::uint8_t* block = BlockAt(matrix, k_blocks, tile, kb);
    for (std::size_t m = 0; m < kTile; ++m) {
      const float d = Bf16At(block + kD + m * 2);
      const float dmin = Bf16At(block + kDmin + m * 2);
      for (int t = 0; t < in.tokens; ++t) {
        const std::uint16_t* x = in.x[t] + kb * kKTile;
        float scaled = 0.0f;
        float minned = 0.0f;
        for (std::size_t g = 0; g < kGroups; ++g) {
          float dot = 0.0f;
          for (std::size_t l = 0; l < kGroup; l += 2) {
            const float x0 = simd::FromBf16(x[g * kGroup + l]);
            const float x1 = simd::FromBf16(x[g * kGroup + l + 1]);
            // Each product is exact in fp32 (a 4-bit code times a bf16).
            dot += static_cast<float>(CodeAt(block, m, g, l + 1)) * x1;
            dot += static_cast<float>(CodeAt(block, m, g, l)) * x0;
          }
          const float sc = static_cast<float>(block[kSc + g * kTile + m]);
          const float mn = static_cast<float>(
              static_cast<std::int8_t>(block[kMn + g * kTile + m]));
          scaled = std::fma(sc, dot, scaled);
          minned = std::fma(mn, in.sums[t][kb * kGroups + g], minned);
        }
        row[t][m] = std::fma(d, scaled, row[t][m]);
        row[t][m] = std::fma(dmin, minned, row[t][m]);
      }
    }
  }
  for (int t = 0; t < in.tokens; ++t) {
    std::copy_n(row[t], kTile, out + static_cast<std::size_t>(t) * kTile);
  }
}

#ifdef GUFO_CPU_GEMV_AVX512
#define GUFO_FAST_ISA "avx512f,avx512bw,avx512vl,avx512vbmi,avx512bf16,fma"
#define GUFO_FAST __attribute__((target(GUFO_FAST_ISA)))

// bf16 bits of 0..15 (and 16..31, unused), for VPERMW.
struct CodeTable {
  alignas(64) std::uint16_t bf16[32];
  CodeTable() {
    for (int i = 0; i < 32; ++i) {
      bf16[i] = simd::ToBf16(static_cast<float>(i));
    }
  }
};

GUFO_FAST inline __m512 Bf16x16(const std::uint8_t* p) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))),
      16));
}

// One pair of inputs for a half's 16 rows: `codes` 32 bytes (row r's codes
// of the pair's two inputs at 2r, 2r + 1) to bf16, then into dot[t].
template<int T>
GUFO_FAST inline void PairDot(__m256i codes, __m512i table, const Inputs& in,
                              std::size_t pair, __m512* dot) {
  const __m512bh w =
      (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepu8_epi16(codes), table);
  for (int t = 0; t < T; ++t) {
    std::int32_t two;
    std::memcpy(&two, in.x[t] + 2 * pair, sizeof(two));
    dot[t] = _mm512_dpbf16_ps(dot[t], w, (__m512bh)_mm512_set1_epi32(two));
  }
}

template<int T>
GUFO_FAST void TileFastT(const std::uint8_t* matrix, std::size_t k_blocks,
                         std::size_t tile, const Inputs& in, float* out) {
  static const q4nx_tile::Perms perms;
  static const CodeTable codes_table;
  const __m512i perm_low = _mm512_load_si512(perms.low);
  const __m512i perm_high = _mm512_load_si512(perms.high);
  const __m512i table = _mm512_load_si512(codes_table.bf16);
  const __m512i nibble = _mm512_set1_epi8(0x0F);
  __m512 row[T][2];
  for (int t = 0; t < T; ++t) {
    row[t][0] = _mm512_setzero_ps();
    row[t][1] = _mm512_setzero_ps();
  }
  for (std::size_t kb = 0; kb < k_blocks; ++kb) {
    const std::uint8_t* block = BlockAt(matrix, k_blocks, tile, kb);
    for (std::size_t h = 0; h < 2; ++h) {
      __m512 scaled[T];
      __m512 minned[T];
      for (int t = 0; t < T; ++t) {
        scaled[t] = _mm512_setzero_ps();
        minned[t] = _mm512_setzero_ps();
      }
      for (std::size_t g = 0; g < kGroups; ++g) {
        __m512 dot[T];
        for (int t = 0; t < T; ++t) {
          dot[t] = _mm512_setzero_ps();
        }
        const std::uint8_t* codes = block + kCodes + (h * kGroups + g) * 32 * 8;
        // Pair index of the group's first input pair in x.
        const std::size_t pair0 = (kb * kKTile + g * kGroup) / 2;
        for (std::size_t run = 0; run < 4; ++run) {
          const __m512i raw = _mm512_loadu_si512(codes + run * 64);
          const __m512i lo = _mm512_and_si512(raw, nibble);
          const __m512i hi =
              _mm512_and_si512(_mm512_srli_epi16(raw, 4), nibble);
          const __m512i p01 = _mm512_permutex2var_epi8(lo, perm_low, hi);
          const __m512i p23 = _mm512_permutex2var_epi8(lo, perm_high, hi);
          const std::size_t pair = pair0 + run * 4;
          PairDot<T>(_mm512_extracti64x4_epi64(p01, 0), table, in, pair, dot);
          PairDot<T>(_mm512_extracti64x4_epi64(p01, 1), table, in, pair + 1,
                     dot);
          PairDot<T>(_mm512_extracti64x4_epi64(p23, 0), table, in, pair + 2,
                     dot);
          PairDot<T>(_mm512_extracti64x4_epi64(p23, 1), table, in, pair + 3,
                     dot);
        }
        const __m512 sc = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                block + kSc + g * kTile + h * kHalf))));
        const __m512 mn = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                block + kMn + g * kTile + h * kHalf))));
        for (int t = 0; t < T; ++t) {
          scaled[t] = _mm512_fmadd_ps(sc, dot[t], scaled[t]);
          minned[t] = _mm512_fmadd_ps(
              mn, _mm512_set1_ps(in.sums[t][kb * kGroups + g]), minned[t]);
        }
      }
      const __m512 d = Bf16x16(block + kD + h * kHalf * 2);
      const __m512 dmin = Bf16x16(block + kDmin + h * kHalf * 2);
      for (int t = 0; t < T; ++t) {
        row[t][h] = _mm512_fmadd_ps(d, scaled[t], row[t][h]);
        row[t][h] = _mm512_fmadd_ps(dmin, minned[t], row[t][h]);
      }
    }
  }
  for (int t = 0; t < T; ++t) {
    _mm512_storeu_ps(out + t * kTile, row[t][0]);
    _mm512_storeu_ps(out + t * kTile + kHalf, row[t][1]);
  }
}

GUFO_FAST void TileFast(const std::uint8_t* matrix, std::size_t k_blocks,
                        std::size_t tile, const Inputs& in, float* out) {
  if (in.tokens == 1) {
    TileFastT<1>(matrix, k_blocks, tile, in, out);
  } else {
    TileFastT<2>(matrix, k_blocks, tile, in, out);
  }
}
#undef GUFO_FAST
#undef GUFO_FAST_ISA
#endif

}  // namespace

CpuGemv::CpuGemv(SpinPool& pool, const std::uint8_t* q4nx, std::size_t rows,
                 std::size_t dim, bool portable)
    : pool_(pool),
      matrix_(q4nx),
      rows_(rows),
      dim_(dim),
      k_blocks_(dim / kKTile),
      tiles_((rows + kTile - 1) / kTile),
      fast_(!portable && q4nx_tile::HasFast()),
      xb_(kTokens * dim),
      sums_(kTokens * (dim / kGroup)) {
  if (dim == 0 || dim % kKTile != 0) {
    throw std::invalid_argument("cpu gemv: dim must be a multiple of 256");
  }
#ifndef GUFO_CPU_GEMV_AVX512
  fast_ = false;
#endif
}

void CpuGemv::Run(const float* x, std::size_t x_stride, float* y,
                  std::size_t y_stride, int tokens) {
  if (tokens < 1 || tokens > kTokens) {
    throw std::invalid_argument("cpu gemv: 1 or 2 tokens");
  }
  Inputs in{};
  in.tokens = tokens;
  const std::size_t groups = dim_ / kGroup;
  for (int t = 0; t < tokens; ++t) {
    const float* xt = x + static_cast<std::size_t>(t) * x_stride;
    std::uint16_t* xb = xb_.data() + static_cast<std::size_t>(t) * dim_;
    float* sums = sums_.data() + static_cast<std::size_t>(t) * groups;
    for (std::size_t g = 0; g < groups; ++g) {
      float sum = 0.0f;
      for (std::size_t l = 0; l < kGroup; ++l) {
        const std::size_t k = g * kGroup + l;
        xb[k] = simd::ToBf16(xt[k]);
        sum += simd::FromBf16(xb[k]);
      }
      sums[g] = sum;
    }
    in.x[t] = xb;
    in.sums[t] = sums;
  }
  pool_.Run(tiles_, [&](std::size_t tile) {
    float out[kTokens * kTile];
#ifdef GUFO_CPU_GEMV_AVX512
    if (fast_) {
      TileFast(matrix_, k_blocks_, tile, in, out);
    } else {
      TilePortable(matrix_, k_blocks_, tile, in, out);
    }
#else
    TilePortable(matrix_, k_blocks_, tile, in, out);
#endif
    const std::size_t first = tile * kTile;
    const std::size_t n = std::min(kTile, rows_ - first);
    for (int t = 0; t < tokens; ++t) {
      std::copy_n(out + static_cast<std::size_t>(t) * kTile, n,
                  y + static_cast<std::size_t>(t) * y_stride + first);
    }
  });
}

}  // namespace gufo_npu::qwen36_35b_a3b
