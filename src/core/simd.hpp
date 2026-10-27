#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

// Row-wise fp32 math for the prefill's CPU side: exp, SiLU and bf16
// conversions over arrays, with AVX2 + FMA when the translation unit is built
// for it (static, like vec.hpp, so each unit gets its own target's body) and
// a scalar fallback otherwise.
namespace gufo_npu::simd {

// bf16 bits from fp32, rounding to nearest even (finite values).
static inline std::uint16_t ToBf16(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>(bits >> 16);
}

static inline float FromBf16(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

#if defined(__AVX2__) && defined(__FMA__)
// e^x per lane: Cephes' expf polynomial after x = n ln2 + r, |r| <= ln2 / 2
// (within 2 ulp of std::exp over the clamped range [-87.3, 88.3]).
static inline __m256 Exp8(__m256 x) {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)),
                    _mm256_set1_ps(88.3f));
  const __m256 n =
      _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  x = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
  x = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), x);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x),
                      _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
  const __m256i scale = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(scale));
}

// x / (1 + e^-x).
static inline __m256 Silu8(__m256 x) {
  const __m256 e = Exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
  return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

static inline __m256 LoadBf16x8(const std::uint16_t* p) {
  const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

static inline void StoreBf16x8(std::uint16_t* p, __m256 v) {
  __m256i bits = _mm256_castps_si256(v);
  const __m256i lsb =
      _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
  bits =
      _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb));
  bits = _mm256_srli_epi32(bits, 16);
  // packus interleaves 128-bit lanes: keep qwords 0 and 2.
  const __m256i packed =
      _mm256_permute4x64_epi64(_mm256_packus_epi32(bits, bits), 0x08);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p),
                   _mm256_castsi256_si128(packed));
}
#endif

static inline float Silu(float x) {
  return x / (1.0f + std::exp(-x));
}

// out[i] = e^(x[i] - shift); returns their sum (out may alias x).
static inline float ExpShiftSum(const float* x, float shift, float* out,
                                int n) {
  int i = 0;
  float total = 0;
#if defined(__AVX2__) && defined(__FMA__)
  __m256 acc = _mm256_setzero_ps();
  const __m256 s = _mm256_set1_ps(shift);
  for (; i + 8 <= n; i += 8) {
    const __m256 e = Exp8(_mm256_sub_ps(_mm256_loadu_ps(x + i), s));
    _mm256_storeu_ps(out + i, e);
    acc = _mm256_add_ps(acc, e);
  }
  float lanes[8];
  _mm256_storeu_ps(lanes, acc);
  for (float lane : lanes) {
    total += lane;
  }
#endif
  for (; i < n; ++i) {
    out[i] = std::exp(x[i] - shift);
    total += out[i];
  }
  return total;
}

// out[i] = silu(x[i]) (out may alias x).
static inline void SiluRow(const float* x, float* out, int n) {
  int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(out + i, Silu8(_mm256_loadu_ps(x + i)));
  }
#endif
  for (; i < n; ++i) {
    out[i] = Silu(x[i]);
  }
}

// y[i] *= silu(z[i]).
static inline void MulSiluRow(float* y, const float* z, int n) {
  int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i),
                                          Silu8(_mm256_loadu_ps(z + i))));
  }
#endif
  for (; i < n; ++i) {
    y[i] *= Silu(z[i]);
  }
}

// out = bf16(silu(gate) * up), gate and up bf16.
static inline void SwiGluBf16Row(const std::uint16_t* gate,
                                 const std::uint16_t* up, std::uint16_t* out,
                                 int n) {
  int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  for (; i + 8 <= n; i += 8) {
    StoreBf16x8(out + i,
                _mm256_mul_ps(Silu8(LoadBf16x8(gate + i)), LoadBf16x8(up + i)));
  }
#endif
  for (; i < n; ++i) {
    out[i] = ToBf16(Silu(FromBf16(gate[i])) * FromBf16(up[i]));
  }
}

// out = bf16(x * scale * w) (scale a scalar, w per value).
static inline void ScaleToBf16Row(const float* x, float scale, const float* w,
                                  std::uint16_t* out, int n) {
  int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  const __m256 s = _mm256_set1_ps(scale);
  for (; i + 8 <= n; i += 8) {
    StoreBf16x8(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), s),
                                       _mm256_loadu_ps(w + i)));
  }
#endif
  for (; i < n; ++i) {
    out[i] = ToBf16(x[i] * scale * w[i]);
  }
}

// out = bf16(x).
static inline void ToBf16Row(const float* x, std::uint16_t* out, int n) {
  int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  for (; i + 8 <= n; i += 8) {
    StoreBf16x8(out + i, _mm256_loadu_ps(x + i));
  }
#endif
  for (; i < n; ++i) {
    out[i] = ToBf16(x[i]);
  }
}

// out = x (bf16), or out += x with `add`.
static inline void FromBf16Row(const std::uint16_t* x, float* out, int n,
                               bool add) {
  int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  if (add) {
    for (; i + 8 <= n; i += 8) {
      _mm256_storeu_ps(
          out + i, _mm256_add_ps(_mm256_loadu_ps(out + i), LoadBf16x8(x + i)));
    }
  } else {
    for (; i + 8 <= n; i += 8) {
      _mm256_storeu_ps(out + i, LoadBf16x8(x + i));
    }
  }
#endif
  for (; i < n; ++i) {
    out[i] = add ? out[i] + FromBf16(x[i]) : FromBf16(x[i]);
  }
}

}  // namespace gufo_npu::simd
