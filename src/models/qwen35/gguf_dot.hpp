#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#endif

#include "core/quant.hpp"
#include "core/vec.hpp"

// One GGUF weight row dotted with fp32 x without dequantizing the row first:
// GgufProjections' fp32 path for the types its models store. `n` is a multiple
// of the type's block. The functions are static (a copy per translation unit),
// as in core/vec.hpp, because the body depends on the unit's target flags:
// the decoders build with AVX2 + FMA + F16C.
namespace gufo_npu::qwen35::gguf_dot {

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
static inline float HorizontalSum(__m256 v) {
  const __m128 sum4 =
      _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  const __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
  return _mm_cvtss_f32(_mm_add_ss(sum2, _mm_movehdup_ps(sum2)));
}
#endif

// Q8_0: blocks of an fp16 scale and 32 int8.
static inline float DotQ8_0(const std::uint8_t* row, const float* x,
                            std::size_t n) {
  constexpr std::size_t kBlock = 32;
  constexpr std::size_t kBlockBytes = 34;
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
  __m256 acc = _mm256_setzero_ps();
  for (std::size_t b = 0; b < n / kBlock; ++b, row += kBlockBytes) {
    std::uint16_t bits;
    std::memcpy(&bits, row, sizeof(bits));
    const __m256 d = _mm256_set1_ps(_cvtsh_ss(bits));
    const auto* q = reinterpret_cast<const __m128i*>(row + 2);
    const float* xb = x + b * kBlock;
    __m256 block = _mm256_setzero_ps();
    for (int j = 0; j < 2; ++j) {
      const __m128i q16 = _mm_loadu_si128(q + j);
      const __m256 lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16));
      const __m256 hi =
          _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16, 8)));
      block = _mm256_fmadd_ps(lo, _mm256_loadu_ps(xb + 16 * j), block);
      block = _mm256_fmadd_ps(hi, _mm256_loadu_ps(xb + 16 * j + 8), block);
    }
    acc = _mm256_fmadd_ps(d, block, acc);
  }
  return HorizontalSum(acc);
#else
  float sum = 0;
  for (std::size_t b = 0; b < n / kBlock; ++b, row += kBlockBytes) {
    std::uint16_t bits;
    std::memcpy(&bits, row, sizeof(bits));
    const auto* q = reinterpret_cast<const std::int8_t*>(row + 2);
    float block = 0;
    for (std::size_t j = 0; j < kBlock; ++j) {
      block += static_cast<float>(q[j]) * x[b * kBlock + j];
    }
    sum += quant::HalfToFloat(bits) * block;
  }
  return sum;
#endif
}

// F32: the row as stored.
static inline float DotF32(const std::uint8_t* row, const float* x,
                           std::size_t n) {
  return vec::Dot(reinterpret_cast<const float*>(row), x, static_cast<int>(n));
}

// The weights' scale d * scale_j and offset dmin * min_j of a Q4_K/Q5_K
// super-block's 8 sub-blocks, from its 12 packed bytes of 6-bit scales and
// mins (quant::ScaleMinK4, all 8 at once).
static inline void ScalesMinsK4(const std::uint8_t* s, float d, float dmin,
                                float* scale, float* min) {
  for (int j = 0; j < 8; ++j) {
    const int sc =
        j < 4 ? s[j] & 63 : (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
    const int m = j < 4 ? s[j + 4] & 63 : (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    scale[j] = d * static_cast<float>(sc);
    min[j] = dmin * static_cast<float>(m);
  }
}

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
// acc + (q * scale - min) . x over one sub-block: 32 codes (bytes) and
// x[0..31].
static inline __m256 SubBlockK4(__m256i codes, __m256 scale, __m256 min,
                                const float* x, __m256 acc) {
  const __m128i lo = _mm256_castsi256_si128(codes);
  const __m128i hi = _mm256_extracti128_si256(codes, 1);
  const __m128i eighths[4] = {lo, _mm_srli_si128(lo, 8), hi,
                              _mm_srli_si128(hi, 8)};
  for (int i = 0; i < 4; ++i) {
    const __m256 q = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(eighths[i]));
    acc = _mm256_fmadd_ps(_mm256_fmsub_ps(q, scale, min),
                          _mm256_loadu_ps(x + 8 * i), acc);
  }
  return acc;
}
#endif

// Q4_K and, with kFifthBit, Q5_K: super-blocks of 256 values in 8
// sub-blocks of 32; weight = d * scale_j * q - dmin * min_j, as
// quant::Dequantize. Q4_K is d, dmin (fp16), 12 scale bytes and qs[128]
// (144 bytes); Q5_K puts qh[32] before qs (176 bytes). Byte l of qs's p-th
// 32 holds value l of sub-block 2p (low nibble) and of 2p + 1 (high
// nibble); bit j of qh[l] adds 16 to value l of sub-block j.
template<bool kFifthBit>
static inline float DotK4(const std::uint8_t* row, const float* x,
                          std::size_t n) {
  constexpr std::size_t kBlock = 256;
  constexpr std::size_t kBlockBytes = kFifthBit ? 176 : 144;
  float scale[8];
  float min[8];
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  const __m256i sixteen = _mm256_set1_epi8(0x10);
  __m256 acc = _mm256_setzero_ps();
  for (std::size_t b = 0; b < n / kBlock; ++b, row += kBlockBytes) {
    std::uint16_t d_bits;
    std::uint16_t dmin_bits;
    std::memcpy(&d_bits, row, sizeof(d_bits));
    std::memcpy(&dmin_bits, row + 2, sizeof(dmin_bits));
    ScalesMinsK4(row + 4, _cvtsh_ss(d_bits), _cvtsh_ss(dmin_bits), scale, min);
    const std::uint8_t* qs = row + 16 + (kFifthBit ? 32 : 0);
    __m256i qh = _mm256_setzero_si256();
    if constexpr (kFifthBit) {
      qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 16));
    }
    const float* xb = x + b * kBlock;
    for (int p = 0; p < 4; ++p, qs += 32) {
      const __m256i q =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
      __m256i codes[2] = {_mm256_and_si256(q, low4),
                          _mm256_and_si256(_mm256_srli_epi16(q, 4), low4)};
      for (int i = 0; i < 2; ++i) {
        const int j = 2 * p + i;
        if constexpr (kFifthBit) {
          const __m256i bit = _mm256_set1_epi8(static_cast<char>(1 << j));
          const __m256i set = _mm256_cmpeq_epi8(_mm256_and_si256(qh, bit), bit);
          codes[i] = _mm256_or_si256(codes[i], _mm256_and_si256(set, sixteen));
        }
        acc = SubBlockK4(codes[i], _mm256_set1_ps(scale[j]),
                         _mm256_set1_ps(min[j]), xb + 32 * j, acc);
      }
    }
  }
  return HorizontalSum(acc);
#else
  float sum = 0;
  for (std::size_t b = 0; b < n / kBlock; ++b, row += kBlockBytes) {
    std::uint16_t d_bits;
    std::uint16_t dmin_bits;
    std::memcpy(&d_bits, row, sizeof(d_bits));
    std::memcpy(&dmin_bits, row + 2, sizeof(dmin_bits));
    ScalesMinsK4(row + 4, quant::HalfToFloat(d_bits),
                 quant::HalfToFloat(dmin_bits), scale, min);
    const std::uint8_t* qh = row + 16;
    const std::uint8_t* qs = row + 16 + (kFifthBit ? 32 : 0);
    const float* xb = x + b * kBlock;
    float block = 0;
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t* bytes = qs + 32 * (j / 2);
      for (int l = 0; l < 32; ++l) {
        int v = (j % 2) == 0 ? bytes[l] & 0xF : bytes[l] >> 4;
        if (kFifthBit && ((qh[l] >> j) & 1) != 0) {
          v += 16;
        }
        block += (scale[j] * static_cast<float>(v) - min[j]) * xb[32 * j + l];
      }
    }
    sum += block;
  }
  return sum;
#endif
}

static inline float DotQ4_K(const std::uint8_t* row, const float* x,
                            std::size_t n) {
  return DotK4<false>(row, x, n);
}

static inline float DotQ5_K(const std::uint8_t* row, const float* x,
                            std::size_t n) {
  return DotK4<true>(row, x, n);
}

// Q6_K: super-blocks of 256 values, 210 bytes: ql[128] (low 4 bits),
// qh[64] (high 2 bits), 16 int8 scales (one per 16 values) and an fp16 d;
// weight = d * scale * (q - 32), as quant::Dequantize. Each half of 128
// values reads 64 bytes of ql and 32 of qh: values l, l + 32, l + 64 and
// l + 96 (l < 32) take ql[l] & 15, ql[l + 32] & 15, ql[l] >> 4 and
// ql[l + 32] >> 4, with bits 0-1, 2-3, 4-5 and 6-7 of qh[l] on top. Per 16
// values the codes times x sum in fp32, then times their scale; d
// multiplies each super-block's total.
static inline float DotQ6_K(const std::uint8_t* row, const float* x,
                            std::size_t n) {
  constexpr std::size_t kBlock = 256;
  constexpr std::size_t kBlockBytes = 210;
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  const __m256i high2 = _mm256_set1_epi8(0x30);
  const __m256i bias = _mm256_set1_epi8(32);
  __m256 acc = _mm256_setzero_ps();
  for (std::size_t b = 0; b < n / kBlock; ++b, row += kBlockBytes) {
    const std::uint8_t* ql = row;
    const std::uint8_t* qh = row + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(row + 192);
    std::uint16_t bits;
    std::memcpy(&bits, row + 208, sizeof(bits));
    const float* xb = x + b * kBlock;
    __m256 block = _mm256_setzero_ps();
    for (int half = 0; half < 2;
         ++half, ql += 64, qh += 32, sc += 8, xb += 128) {
      const __m256i l0 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
      const __m256i l1 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
      const __m256i h =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
      // Values 32 k .. 32 k + 31 of the half, as 6-bit codes. The 16-bit
      // shifts carry bits across bytes; the masks drop them.
      const __m256i codes[4] = {
          _mm256_or_si256(_mm256_and_si256(l0, low4),
                          _mm256_and_si256(_mm256_slli_epi16(h, 4), high2)),
          _mm256_or_si256(_mm256_and_si256(l1, low4),
                          _mm256_and_si256(_mm256_slli_epi16(h, 2), high2)),
          _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4),
                          _mm256_and_si256(h, high2)),
          _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4),
                          _mm256_and_si256(_mm256_srli_epi16(h, 2), high2)),
      };
      for (int k = 0; k < 4; ++k) {
        const __m256i q = _mm256_sub_epi8(codes[k], bias);
        const __m128i groups[2] = {_mm256_castsi256_si128(q),
                                   _mm256_extracti128_si256(q, 1)};
        for (int g = 0; g < 2; ++g) {
          const float* xg = xb + 32 * k + 16 * g;
          __m256 group =
              _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(groups[g])),
                            _mm256_loadu_ps(xg));
          group = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                      _mm_srli_si128(groups[g], 8))),
                                  _mm256_loadu_ps(xg + 8), group);
          block = _mm256_fmadd_ps(
              _mm256_set1_ps(static_cast<float>(sc[2 * k + g])), group, block);
        }
      }
    }
    acc = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(bits)), block, acc);
  }
  return HorizontalSum(acc);
#else
  float sum = 0;
  for (std::size_t b = 0; b < n / kBlock; ++b, row += kBlockBytes) {
    const std::uint8_t* ql = row;
    const std::uint8_t* qh = row + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(row + 192);
    std::uint16_t bits;
    std::memcpy(&bits, row + 208, sizeof(bits));
    const float* xb = x + b * kBlock;
    float block = 0;
    for (int half = 0; half < 2;
         ++half, ql += 64, qh += 32, sc += 8, xb += 128) {
      for (int g = 0; g < 8; ++g) {
        float group = 0;
        for (int i = 0; i < 16; ++i) {
          const int p = 16 * g + i;
          const int k = p / 32;
          const int l = p % 32;
          const int byte = (k & 1) != 0 ? ql[l + 32] : ql[l];
          const int low = k < 2 ? byte & 0xF : byte >> 4;
          const int high = (qh[l] >> (2 * k)) & 3;
          group += static_cast<float>((low | (high << 4)) - 32) * xb[p];
        }
        block += static_cast<float>(sc[g]) * group;
      }
    }
    sum += quant::HalfToFloat(bits) * block;
  }
  return sum;
#endif
}

}  // namespace gufo_npu::qwen35::gguf_dot
