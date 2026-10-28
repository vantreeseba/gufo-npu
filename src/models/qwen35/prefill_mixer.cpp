#include "models/qwen35/prefill_mixer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "core/quant.hpp"
#include "core/simd.hpp"
#include "core/vec.hpp"

namespace gufo_npu::qwen35 {
namespace {

using vec::Dot;
using vec::RmsNorm;

// Attention: query tokens per work item, keys per online-softmax block.
constexpr int kQueryBlock = 16;
constexpr int kKeyBlock = 64;

float Sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}
float Softplus(float x) {
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

// The delta rule for one token on kDeltaColumns value columns j of a head's
// state S (dk rows of the columns, `stride` apart), with the decoder's
// arithmetic: u = (decay S)^T k, delta = beta (v - u), S' = decay S +
// k delta^T, y = S'^T q.
constexpr int kDeltaColumns = 32;
constexpr int kMaxHeadDim = 128;

void DeltaColumns(float* s, int dk, int stride, const float* k, const float* q,
                  const float* v, float decay, float beta, float* y) {
#if defined(__AVX2__) && defined(__FMA__)
  __m256 acc[4];
  for (auto& a : acc) {
    a = _mm256_setzero_ps();
  }
  for (int i = 0; i < dk; ++i) {
    const float* row = s + static_cast<std::size_t>(i) * stride;
    const __m256 ki = _mm256_set1_ps(k[i] * decay);
    for (int c = 0; c < 4; ++c) {
      acc[c] = _mm256_fmadd_ps(_mm256_loadu_ps(row + 8 * c), ki, acc[c]);
    }
  }
  __m256 delta[4];
  __m256 out[4];
  const __m256 b = _mm256_set1_ps(beta);
  for (int c = 0; c < 4; ++c) {
    delta[c] =
        _mm256_mul_ps(b, _mm256_sub_ps(_mm256_loadu_ps(v + 8 * c), acc[c]));
    out[c] = _mm256_setzero_ps();
  }
  const __m256 dc = _mm256_set1_ps(decay);
  for (int i = 0; i < dk; ++i) {
    float* row = s + static_cast<std::size_t>(i) * stride;
    const __m256 ki = _mm256_set1_ps(k[i]);
    const __m256 qi = _mm256_set1_ps(q[i]);
    for (int c = 0; c < 4; ++c) {
      const __m256 updated = _mm256_fmadd_ps(
          ki, delta[c], _mm256_mul_ps(_mm256_loadu_ps(row + 8 * c), dc));
      _mm256_storeu_ps(row + 8 * c, updated);
      out[c] = _mm256_fmadd_ps(updated, qi, out[c]);
    }
  }
  for (int c = 0; c < 4; ++c) {
    _mm256_storeu_ps(y + 8 * c, out[c]);
  }
#else
  float delta[kDeltaColumns] = {};
  for (int i = 0; i < dk; ++i) {
    const float* row = s + static_cast<std::size_t>(i) * stride;
    const float ki = k[i] * decay;
    for (int j = 0; j < kDeltaColumns; ++j) {
      delta[j] += row[j] * ki;
    }
  }
  for (int j = 0; j < kDeltaColumns; ++j) {
    delta[j] = beta * (v[j] - delta[j]);
    y[j] = 0;
  }
  for (int i = 0; i < dk; ++i) {
    float* row = s + static_cast<std::size_t>(i) * stride;
    for (int j = 0; j < kDeltaColumns; ++j) {
      row[j] = row[j] * decay + k[i] * delta[j];
      y[j] += row[j] * q[i];
    }
  }
#endif
}

// Attention kernels over one token's G = 4 query heads of a kv head (fp32
// rows hd apart) against the fp16 K/V cache (vec::ToHalf bits, rows `stride`
// halves apart), widening the cache in registers (F16C or AVX-512).
using Half = std::uint16_t;

#if defined(__AVX2__) && defined(__FMA__)
float HorizontalSum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

__m256 LoadHalf8(const Half* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}
#endif

#if defined(__x86_64__) && defined(__GNUC__)
// Zen 5 runs AVX-512 at full width: the attention kernels have 512-bit
// versions, picked at run time (the build targets AVX2).
#define GUFO_PREFILL_AVX512 1

bool HasAvx512() {
  static const bool has = __builtin_cpu_supports("avx512f");
  return has;
}

__attribute__((target("avx512f"))) __m512 LoadHalf16(const Half* p) {
  return _mm512_cvtph_ps(
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}

__attribute__((target("avx512f"))) void Scores4Avx512(const float* q, int hd,
                                                      const Half* keys,
                                                      std::size_t stride,
                                                      int count, float* scores,
                                                      int ld) {
  int j = 0;
  for (; j + 2 <= count; j += 2) {
    const Half* k0 = keys + static_cast<std::size_t>(j) * stride;
    const Half* k1 = k0 + stride;
    __m512 acc[4][2];
    for (auto& a : acc) {
      a[0] = _mm512_setzero_ps();
      a[1] = _mm512_setzero_ps();
    }
    for (int e = 0; e < hd; e += 16) {
      const __m512 x0 = LoadHalf16(k0 + e);
      const __m512 x1 = LoadHalf16(k1 + e);
      for (int g = 0; g < 4; ++g) {
        const __m512 qv = _mm512_loadu_ps(q + g * hd + e);
        acc[g][0] = _mm512_fmadd_ps(qv, x0, acc[g][0]);
        acc[g][1] = _mm512_fmadd_ps(qv, x1, acc[g][1]);
      }
    }
    for (int g = 0; g < 4; ++g) {
      scores[g * ld + j] = _mm512_reduce_add_ps(acc[g][0]);
      scores[g * ld + j + 1] = _mm512_reduce_add_ps(acc[g][1]);
    }
  }
  for (; j < count; ++j) {
    const Half* key = keys + static_cast<std::size_t>(j) * stride;
    for (int g = 0; g < 4; ++g) {
      scores[g * ld + j] = vec::DotHalf(q + g * hd, key, hd);
    }
  }
}

__attribute__((target("avx512f"))) void AccumulateValues4Avx512(
    const float* p, int ld, const Half* values, std::size_t stride, int count,
    int hd, float* out) {
  for (int e = 0; e < hd; e += 32) {
    __m512 acc[4][2];
    for (int g = 0; g < 4; ++g) {
      acc[g][0] = _mm512_loadu_ps(out + g * hd + e);
      acc[g][1] = _mm512_loadu_ps(out + g * hd + e + 16);
    }
    for (int j = 0; j < count; ++j) {
      const Half* v = values + static_cast<std::size_t>(j) * stride + e;
      const __m512 v0 = LoadHalf16(v);
      const __m512 v1 = LoadHalf16(v + 16);
      for (int g = 0; g < 4; ++g) {
        const __m512 pg = _mm512_set1_ps(p[g * ld + j]);
        acc[g][0] = _mm512_fmadd_ps(pg, v0, acc[g][0]);
        acc[g][1] = _mm512_fmadd_ps(pg, v1, acc[g][1]);
      }
    }
    for (int g = 0; g < 4; ++g) {
      _mm512_storeu_ps(out + g * hd + e, acc[g][0]);
      _mm512_storeu_ps(out + g * hd + e + 16, acc[g][1]);
    }
  }
}
#endif

// scores[g * ld + j] = q_g . key_j for G = 4 query heads and keys j in
// [0, count); hd a multiple of 32.
void Scores4(const float* q, int hd, const Half* keys, std::size_t stride,
             int count, float* scores, int ld) {
#ifdef GUFO_PREFILL_AVX512
  if (HasAvx512()) {
    Scores4Avx512(q, hd, keys, stride, count, scores, ld);
    return;
  }
#endif
  int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
  for (; j + 2 <= count; j += 2) {
    const Half* k0 = keys + static_cast<std::size_t>(j) * stride;
    const Half* k1 = k0 + stride;
    __m256 acc[4][2];
    for (auto& a : acc) {
      a[0] = _mm256_setzero_ps();
      a[1] = _mm256_setzero_ps();
    }
    for (int e = 0; e < hd; e += 8) {
      const __m256 x0 = LoadHalf8(k0 + e);
      const __m256 x1 = LoadHalf8(k1 + e);
      for (int g = 0; g < 4; ++g) {
        const __m256 qv = _mm256_loadu_ps(q + g * hd + e);
        acc[g][0] = _mm256_fmadd_ps(qv, x0, acc[g][0]);
        acc[g][1] = _mm256_fmadd_ps(qv, x1, acc[g][1]);
      }
    }
    for (int g = 0; g < 4; ++g) {
      scores[g * ld + j] = HorizontalSum(acc[g][0]);
      scores[g * ld + j + 1] = HorizontalSum(acc[g][1]);
    }
  }
#endif
  for (; j < count; ++j) {
    const Half* key = keys + static_cast<std::size_t>(j) * stride;
    for (int g = 0; g < 4; ++g) {
      scores[g * ld + j] = vec::DotHalf(q + g * hd, key, hd);
    }
  }
}

// out_g += sum_j p[g * ld + j] value_j for G = 4 rows of hd (a multiple of
// 32).
void AccumulateValues4(const float* p, int ld, const Half* values,
                       std::size_t stride, int count, int hd, float* out) {
#ifdef GUFO_PREFILL_AVX512
  if (HasAvx512()) {
    AccumulateValues4Avx512(p, ld, values, stride, count, hd, out);
    return;
  }
#endif
#if defined(__AVX2__) && defined(__FMA__)
  for (int e = 0; e < hd; e += 16) {
    __m256 acc[4][2];
    for (int g = 0; g < 4; ++g) {
      acc[g][0] = _mm256_loadu_ps(out + g * hd + e);
      acc[g][1] = _mm256_loadu_ps(out + g * hd + e + 8);
    }
    for (int j = 0; j < count; ++j) {
      const Half* v = values + static_cast<std::size_t>(j) * stride + e;
      const __m256 v0 = LoadHalf8(v);
      const __m256 v1 = LoadHalf8(v + 8);
      for (int g = 0; g < 4; ++g) {
        const __m256 pg = _mm256_broadcast_ss(p + g * ld + j);
        acc[g][0] = _mm256_fmadd_ps(pg, v0, acc[g][0]);
        acc[g][1] = _mm256_fmadd_ps(pg, v1, acc[g][1]);
      }
    }
    for (int g = 0; g < 4; ++g) {
      _mm256_storeu_ps(out + g * hd + e, acc[g][0]);
      _mm256_storeu_ps(out + g * hd + e + 8, acc[g][1]);
    }
  }
#else
  for (int j = 0; j < count; ++j) {
    const Half* v = values + static_cast<std::size_t>(j) * stride;
    for (int g = 0; g < 4; ++g) {
      vec::AddScaledHalf(p[g * ld + j], v, out + g * hd, hd);
    }
  }
#endif
}

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

}  // namespace

PrefillMixerWeights LoadPrefillMixer(const gguf::File& file,
                                     const Config& config, int il) {
  const std::string p = "blk." + std::to_string(il) + ".";
  auto t = [&](const char* name) -> const gguf::TensorInfo& {
    return file.Tensor(p + name);
  };
  PrefillMixerWeights w;
  w.attention = config.IsFullAttention(il);
  if (w.attention) {
    w.q_norm = ToFloats(t("attn_q_norm.weight"));
    w.k_norm = ToFloats(t("attn_k_norm.weight"));
    return w;
  }
  w.alpha = ToFloats(t("ssm_alpha.weight"));
  w.beta = ToFloats(t("ssm_beta.weight"));
  const auto conv = ToFloats(t("ssm_conv1d.weight"));  // [ch][kernel]
  const int kernel = config.conv_kernel;
  const int channels = config.ConvChannels();
  w.conv.resize(conv.size());
  for (int ch = 0; ch < channels; ++ch) {
    for (int s = 0; s < kernel; ++s) {
      w.conv[static_cast<std::size_t>(s) * channels + ch] =
          conv[static_cast<std::size_t>(ch) * kernel + s];
    }
  }
  w.ssm_a = ToFloats(t("ssm_a"));
  w.dt_bias = ToFloats(t("ssm_dt.bias"));
  w.ssm_norm = ToFloats(t("ssm_norm.weight"));
  return w;
}

PrefillMixer::PrefillMixer(const Config& config, BurstPool& pool,
                           int max_tokens)
    : config_(config), pool_(pool), max_tokens_(max_tokens) {
  const Config& cfg = config_;
  if (cfg.kv_heads <= 0 || cfg.heads % cfg.kv_heads != 0 ||
      (cfg.heads / cfg.kv_heads) % 4 != 0 || cfg.head_dim % 32 != 0) {
    throw std::runtime_error(
        "npu prefill: a multiple of 4 query heads per kv head and a head "
        "size that is a multiple of 32 only");
  }
  if (cfg.linear_head_dim > kMaxHeadDim ||
      cfg.linear_head_dim % kDeltaColumns != 0) {
    throw std::runtime_error("npu prefill: unsupported DeltaNet head size");
  }
  rope_inv_freq_.resize(cfg.rope_dim / 2);
  for (int i = 0; i < cfg.rope_dim / 2; ++i) {
    rope_inv_freq_[i] = static_cast<float>(
        1.0 / std::pow(static_cast<double>(cfg.rope_theta),
                       2.0 * i / static_cast<double>(cfg.rope_dim)));
  }
  const auto rows = static_cast<std::size_t>(max_tokens);
  gates_.resize(rows * 2 * cfg.linear_value_heads);
  u_.resize(rows * cfg.ConvChannels());
  queries_.resize(rows * cfg.heads * cfg.head_dim);
  rope_cos_.resize(rows * cfg.rope_dim / 2);
  rope_sin_.resize(rope_cos_.size());
}

// cos/sin of each RoPE frequency at the chunk's positions, as the decoder
// computes them.
void PrefillMixer::SetChunk(int count, int position) {
  if (count < 0 || count > max_tokens_) {
    throw std::logic_error(
        "npu prefill: chunk longer than the mixer's scratch");
  }
  n_ = count;
  position_ = position;
  const int half = config_.rope_dim / 2;
  for (int t = 0; t < n_; ++t) {
    for (int i = 0; i < half; ++i) {
      const double angle =
          static_cast<double>(position_ + t) * rope_inv_freq_[i];
      rope_cos_[t * half + i] = static_cast<float>(std::cos(angle));
      rope_sin_[t * half + i] = static_cast<float>(std::sin(angle));
    }
  }
}

void PrefillMixer::Rope(float* x, int t) const {
  const int half = config_.rope_dim / 2;
  const float* cs = rope_cos_.data() + t * half;
  const float* sn = rope_sin_.data() + t * half;
  for (int i = 0; i < half; ++i) {
    const float a0 = x[i];
    const float b0 = x[i + half];
    x[i] = a0 * cs[i] - b0 * sn[i];
    x[i + half] = a0 * sn[i] + b0 * cs[i];
  }
}

void PrefillMixer::AppendKv(const float* in, std::size_t in_stride,
                            const PrefillMixerWeights& w,
                            DecodeState::Layer& st) {
  const Config& cfg = config_;
  const int hd = cfg.head_dim;
  const std::size_t kv = static_cast<std::size_t>(cfg.kv_heads) * hd;
  const std::size_t base = st.keys.size();
  st.keys.resize(base + n_ * kv);
  st.values.resize(base + n_ * kv);
  pool_.Run(static_cast<std::size_t>(n_), [&](std::size_t t) {
    const float* k = in + t * in_stride + 2 * cfg.heads * hd;
    const float* v = k + kv;
    thread_local std::vector<float> key;
    key.resize(hd);
    Half* kd = st.keys.data() + base + t * kv;
    for (int kh = 0; kh < cfg.kv_heads; ++kh) {
      RmsNorm(k + kh * hd, w.k_norm.data(), hd, cfg.rms_eps, key.data());
      Rope(key.data(), static_cast<int>(t));
      vec::ToHalf(key.data(), kd + kh * hd, hd);
    }
    vec::ToHalf(v, st.values.data() + base + t * kv, static_cast<int>(kv));
  });
}

void PrefillMixer::Attention(const float* in, std::size_t in_stride,
                             const PrefillMixerWeights& w,
                             DecodeState::Layer& st, float* mixed) {
  const Config& cfg = config_;
  const int n = n_;
  const int hd = cfg.head_dim;
  const int heads = cfg.heads;
  const int group = heads / cfg.kv_heads;
  const std::size_t kv = static_cast<std::size_t>(cfg.kv_heads) * hd;
  // Keys cached before the chunk's tokens.
  const int past = static_cast<int>(st.keys.size() / kv);
  AppendKv(in, in_stride, w, st);
  // Queries: q_norm, RoPE, and 1/sqrt(hd) (a power of two: exact).
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  pool_.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
    for (int qh = 0; qh < heads; ++qh) {
      const float* raw = in + t * in_stride + qh * 2 * hd;
      float* q = queries_.data() + (t * heads + qh) * hd;
      RmsNorm(raw, w.q_norm.data(), hd, cfg.rms_eps, q);
      Rope(q, static_cast<int>(t));
      for (int e = 0; e < hd; ++e) {
        q[e] *= scale;
      }
    }
  });
  // Work items: (kv head, block of query tokens), each with that kv head's
  // query heads (4 at a time through the kernels), softmax online over key
  // blocks.
  const int blocks = (n + kQueryBlock - 1) / kQueryBlock;
  const int out_stride = heads * hd;
  pool_.Run(
      static_cast<std::size_t>(blocks * cfg.kv_heads), [&](std::size_t item) {
        const int g = static_cast<int>(item) % cfg.kv_heads;
        const int t0 = static_cast<int>(item) / cfg.kv_heads * kQueryBlock;
        const int t1 = std::min(n, t0 + kQueryBlock);
        const int rows = (t1 - t0) * group;
        thread_local std::vector<float> o, m, sum, p;
        o.assign(static_cast<std::size_t>(rows) * hd, 0.0f);
        m.assign(rows, -INFINITY);
        sum.assign(rows, 0.0f);
        p.resize(static_cast<std::size_t>(rows) * kKeyBlock);
        const Half* keys = st.keys.data() + g * hd;
        const Half* values = st.values.data() + g * hd;
        const int last = past + t1;  // keys any row of the item sees
        for (int jb = 0; jb < last; jb += kKeyBlock) {
          for (int t = t0; t < t1; ++t) {
            const int limit = std::min(past + t + 1, jb + kKeyBlock);
            if (limit <= jb) {
              continue;
            }
            const int count = limit - jb;
            const int r0 = (t - t0) * group;
            const float* q =
                queries_.data() +
                (static_cast<std::size_t>(t) * heads + g * group) * hd;
            float* pr = p.data() + static_cast<std::size_t>(r0) * kKeyBlock;
            for (int r4 = 0; r4 < group; r4 += 4) {
              Scores4(q + static_cast<std::size_t>(r4) * hd, hd,
                      keys + static_cast<std::size_t>(jb) * kv, kv, count,
                      pr + static_cast<std::size_t>(r4) * kKeyBlock, kKeyBlock);
            }
            for (int r = 0; r < group; ++r) {
              float* s = pr + r * kKeyBlock;
              float mx = m[r0 + r];
              for (int j = 0; j < count; ++j) {
                mx = std::max(mx, s[j]);
              }
              const float correction = std::exp(m[r0 + r] - mx);
              const float total = simd::ExpShiftSum(s, mx, s, count);
              if (correction != 1.0f) {
                float* orow = o.data() + static_cast<std::size_t>(r0 + r) * hd;
                for (int e = 0; e < hd; ++e) {
                  orow[e] *= correction;
                }
              }
              sum[r0 + r] = sum[r0 + r] * correction + total;
              m[r0 + r] = mx;
            }
            for (int r4 = 0; r4 < group; r4 += 4) {
              AccumulateValues4(
                  pr + static_cast<std::size_t>(r4) * kKeyBlock, kKeyBlock,
                  values + static_cast<std::size_t>(jb) * kv, kv, count, hd,
                  o.data() + static_cast<std::size_t>(r0 + r4) * hd);
            }
          }
        }
        for (int t = t0; t < t1; ++t) {
          for (int r = 0; r < group; ++r) {
            const int qh = g * group + r;
            const float* orow =
                o.data() + static_cast<std::size_t>((t - t0) * group + r) * hd;
            const float inv = 1.0f / sum[(t - t0) * group + r];
            const float* gate = in + t * in_stride + qh * 2 * hd + hd;
            float* out =
                mixed + static_cast<std::size_t>(t) * out_stride + qh * hd;
            for (int e = 0; e < hd; ++e) {
              out[e] = orow[e] * inv * Sigmoid(gate[e]);
            }
          }
        }
      });
}

void PrefillMixer::DeltaNet(const float* in, std::size_t in_stride,
                            const float* normed, const PrefillMixerWeights& w,
                            DecodeState::Layer& st, float* mixed) {
  const Config& cfg = config_;
  const int n = n_;
  const int d = cfg.hidden;
  const int hk = cfg.linear_key_heads;
  const int hv = cfg.linear_value_heads;
  const int dk = cfg.linear_head_dim;
  const int key_dim = cfg.LinearKeyDim();
  const int value_dim = cfg.LinearValueDim();
  const int channels = cfg.ConvChannels();
  const int window = cfg.conv_kernel - 1;
  float* u = u_.data();
  float* gates = gates_.data();

  // Per token and value head: the decay exp(g) and the write strength
  // beta, from the alpha and beta rows (fp32 on the CPU, as in decode).
  pool_.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
    const float* x = normed + t * d;
    float* out = gates + t * 2 * hv;
    for (int head = 0; head < hv; ++head) {
      const float a =
          Dot(w.alpha.data() + static_cast<std::size_t>(head) * d, x, d);
      const float b =
          Dot(w.beta.data() + static_cast<std::size_t>(head) * d, x, d);
      out[head] = std::exp(w.ssm_a[head] * Softplus(a + w.dt_bias[head]));
      out[hv + head] = Sigmoid(b);
    }
  });

  // Causal depthwise conv over [3 previous inputs, current], then SiLU:
  // blocks of channels, vectorized across them, along the tokens.
  constexpr int kConvBlock = 128;
  pool_.Run(
      static_cast<std::size_t>((channels + kConvBlock - 1) / kConvBlock),
      [&](std::size_t block) {
        const int c0 = static_cast<int>(block) * kConvBlock;
        const int width = std::min(kConvBlock, channels - c0);
        const float* x[8];
        for (int t = 0; t < n; ++t) {
          for (int s = 0; s <= window; ++s) {
            const int src = t - window + s;
            x[s] = src >= 0
                       ? in + src * in_stride + c0
                       : st.conv.data() +
                             static_cast<std::size_t>(src + window) * channels +
                             c0;
          }
          float* out = u + static_cast<std::size_t>(t) * channels + c0;
          const float* kw = w.conv.data() + c0;
          for (int i = 0; i < width; ++i) {
            float acc = kw[static_cast<std::size_t>(window) * channels + i] *
                        x[window][i];
            for (int s = 0; s < window; ++s) {
              acc += kw[static_cast<std::size_t>(s) * channels + i] * x[s][i];
            }
            out[i] = acc;
          }
          simd::SiluRow(out, out, width);
        }
      });
  // The window after these tokens: their last `window` inputs. Ascending
  // s reads older entries (src + window = n + s > s) before they are
  // overwritten.
  for (int s = 0; s < window; ++s) {
    const int src = n - window + s;
    float* dst = st.conv.data() + static_cast<std::size_t>(s) * channels;
    if (src >= 0) {
      std::copy_n(in + src * in_stride, channels, dst);
    } else {
      std::copy_n(
          st.conv.data() + static_cast<std::size_t>(src + window) * channels,
          channels, dst);
    }
  }

  // L2-normalize q and k per key head; scale q by 1/sqrt(dk).
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(dk));
  pool_.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
    float* q = u + t * channels;
    float* k = q + key_dim;
    for (int head = 0; head < hk; ++head) {
      for (float* x : {q + head * dk, k + head * dk}) {
        const float inv = 1.0f / std::sqrt(Dot(x, x, dk) + 1e-6f);
        const float sc = x == q + head * dk ? inv * q_scale : inv;
        for (int e = 0; e < dk; ++e) {
          x[e] *= sc;
        }
      }
    }
  });

  // The delta rule along the tokens. Each value column of a head's state
  // evolves on its own (given k, q, v and the gates), so the jobs are
  // (value head, block of kDeltaColumns columns), four times as many as
  // heads to balance. A job copies its 16 KB slice of the state into an
  // aligned local block (no cache line shared with another job, all in
  // L1), runs every token, and copies it back.
  const int column_blocks = dk / kDeltaColumns;
  pool_.Run(static_cast<std::size_t>(hv * column_blocks), [&](std::size_t job) {
    const int head = static_cast<int>(job) / column_blocks;
    const int j0 = static_cast<int>(job) % column_blocks * kDeltaColumns;
    const int kh = head % hk;  // GGUF order: value heads tile key heads
    float* sm = st.ssm.data() + static_cast<std::size_t>(head) * dk * dk + j0;
    alignas(64) float local[kMaxHeadDim * kDeltaColumns];
    for (int i = 0; i < dk; ++i) {
      std::copy_n(sm + static_cast<std::size_t>(i) * dk, kDeltaColumns,
                  local + i * kDeltaColumns);
    }
    for (int t = 0; t < n; ++t) {
      const float* ut = u + static_cast<std::size_t>(t) * channels;
      const float* g = gates + static_cast<std::size_t>(t) * 2 * hv;
      DeltaColumns(
          local, dk, kDeltaColumns, ut + key_dim + kh * dk, ut + kh * dk,
          ut + 2 * key_dim + head * dk + j0, g[head], g[hv + head],
          mixed + static_cast<std::size_t>(t) * value_dim + head * dk + j0);
    }
    for (int i = 0; i < dk; ++i) {
      std::copy_n(local + i * kDeltaColumns, kDeltaColumns,
                  sm + static_cast<std::size_t>(i) * dk);
    }
  });

  // Gated RMSNorm per head: RMS(o) * w * silu(z).
  pool_.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
    for (int head = 0; head < hv; ++head) {
      float* yh = mixed + t * value_dim + head * dk;
      RmsNorm(yh, w.ssm_norm.data(), dk, cfg.rms_eps, yh);
      simd::MulSiluRow(yh, in + t * in_stride + channels + head * dk, dk);
    }
  });
}

}  // namespace gufo_npu::qwen35
