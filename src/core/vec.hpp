#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__) || defined(__F16C__)
#include <immintrin.h>
#endif

// fp32 helpers for the decode loop's CPU math. They are static (a copy per
// translation unit) because the body depends on the unit's target flags: the
// decoders build with AVX2 + FMA.
namespace gufo_npu::vec {

// Dot product with independent partial sums, so it vectorizes (a single
// float sum may not be reordered).
static inline float Dot(const float* a, const float* b, int n) {
#if defined(__AVX2__) && defined(__FMA__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 =
        _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                           _mm256_loadu_ps(b + i + 8), acc1);
  }
  const __m256 acc = _mm256_add_ps(acc0, acc1);
  __m128 sum =
      _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float total = _mm_cvtss_f32(sum);
#else
  float lanes[8] = {};
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    for (int l = 0; l < 8; ++l) {
      lanes[l] += a[i + l] * b[i + l];
    }
  }
  float total = 0;
  for (float lane : lanes) {
    total += lane;
  }
#endif
  for (; i < n; ++i) {
    total += a[i] * b[i];
  }
  return total;
}

// out = RMS(x) * w over n values (out may alias x). The GGUF norm weights
// already hold Qwen's +1 (except ssm_norm, which has none to add).
static inline void RmsNorm(const float* x, const float* w, int n, float eps,
                           float* out) {
  const float inv =
      1.0f / std::sqrt(Dot(x, x, n) / static_cast<float>(n) + eps);
  for (int i = 0; i < n; ++i) {
    out[i] = x[i] * inv * w[i];
  }
}

// IEEE half precision (fp16) storage, e.g. for the attention K/V cache:
// half the bytes of fp32 and 3 more mantissa bits than bf16. F16C converts
// 8 values per instruction; without it, _Float16 does it one at a time.
static inline std::uint16_t ToHalf(float x) {
#if defined(__F16C__)
  return static_cast<std::uint16_t>(_cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT));
#else
  const auto h = static_cast<_Float16>(x);
  std::uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
#endif
}

static inline float FromHalf(std::uint16_t bits) {
#if defined(__F16C__)
  return _cvtsh_ss(bits);
#else
  _Float16 h;
  std::memcpy(&h, &bits, sizeof(h));
  return static_cast<float>(h);
#endif
}

// out[i] = fp16(in[i]), rounded to nearest even.
static inline void ToHalf(const float* in, std::uint16_t* out, int n) {
  int i = 0;
#if defined(__F16C__)
  for (; i + 8 <= n; i += 8) {
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(out + i),
        _mm256_cvtps_ph(_mm256_loadu_ps(in + i), _MM_FROUND_TO_NEAREST_INT));
  }
#endif
  for (; i < n; ++i) {
    out[i] = ToHalf(in[i]);
  }
}

// Dot product of fp32 a and fp16 b.
static inline float DotHalf(const float* a, const std::uint16_t* b, int n) {
  int i = 0;
  float total = 0;
#if defined(__F16C__) && defined(__FMA__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  for (; i + 16 <= n; i += 16) {
    const __m128i* h = reinterpret_cast<const __m128i*>(b + i);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                           _mm256_cvtph_ps(_mm_loadu_si128(h)), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                           _mm256_cvtph_ps(_mm_loadu_si128(h + 1)), acc1);
  }
  const __m256 acc = _mm256_add_ps(acc0, acc1);
  __m128 sum =
      _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  total = _mm_cvtss_f32(sum);
#endif
  for (; i < n; ++i) {
    total += a[i] * FromHalf(b[i]);
  }
  return total;
}

// y[i] += s * x[i], x in fp16.
static inline void AddScaledHalf(float s, const std::uint16_t* x, float* y,
                                 int n) {
  int i = 0;
#if defined(__F16C__) && defined(__FMA__)
  const __m256 scale = _mm256_set1_ps(s);
  for (; i + 8 <= n; i += 8) {
    const __m256 v = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i)));
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(scale, v, _mm256_loadu_ps(y + i)));
  }
#endif
  for (; i < n; ++i) {
    y[i] += s * FromHalf(x[i]);
  }
}

}  // namespace gufo_npu::vec
