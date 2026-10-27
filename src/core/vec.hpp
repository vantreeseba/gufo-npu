#pragma once

#include <cmath>

#if defined(__AVX2__) && defined(__FMA__)
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

}  // namespace gufo_npu::vec
