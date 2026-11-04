#include "models/qwen35_2b/deltanet.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "core/quant.hpp"
#include "core/vec.hpp"

// Every expression below has the shape of its counterpart in
// qwen35::Mixer::Linear (src/models/qwen35/decoder.cpp) and the decoder's
// alpha/beta rows, and this file is built with the same flags, so the
// compiler fuses the same multiply-adds: the results are bit-identical
// (qwen35_deltanet_test). Change the two together.
namespace gufo_npu::qwen35_2b {
namespace {

using vec::Dot;
using vec::RmsNorm;

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

float Sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}
float Silu(float x) {
  return x * Sigmoid(x);
}
float Softplus(float x) {
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

// The delta rule's columns per block: a block of a head's state (128 rows of
// 32 fp32, 16 KB) stays in L1 while both tokens of a pair are applied.
constexpr int kColumns = 32;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxTokens = 2;

// One token's delta rule on columns [j0, j0 + kColumns) of a head's state
// (dk x dk, row i at i * dk): reads src, writes dst (which may be src).
// u = (decay S)^T k, delta = beta (v - u), S' = decay S + k delta^T,
// y = S'^T q, as Mixer::Linear's head loops (its passes run over all dk
// columns; each column's arithmetic is the same).
void DeltaBlock(const float* src, float* dst, int dk, int j0, const float* k,
                const float* q, const float* v, float decay, float b,
                float* yh) {
  float delta[kColumns] = {};
  for (int i = 0; i < dk; ++i) {
    const float* row = src + static_cast<std::size_t>(i) * dk + j0;
    const float ki = k[i] * decay;
    for (int j = 0; j < kColumns; ++j) {
      delta[j] += row[j] * ki;
    }
  }
  for (int j = 0; j < kColumns; ++j) {
    delta[j] = b * (v[j0 + j] - delta[j]);
  }
  float* y = yh + j0;
  std::fill_n(y, kColumns, 0.0f);
  for (int i = 0; i < dk; ++i) {
    const float* row = src + static_cast<std::size_t>(i) * dk + j0;
    float* out = dst + static_cast<std::size_t>(i) * dk + j0;
    const float ki = k[i];
    const float qi = q[i];
    for (int j = 0; j < kColumns; ++j) {
      out[j] = row[j] * decay + ki * delta[j];
      y[j] += out[j] * qi;
    }
  }
}

#if defined(__x86_64__) && defined(__GNUC__) && defined(__AVX2__) && \
    defined(__FMA__)
#define GUFO_DELTANET_AVX512 1

bool HasAvx512() {
  static const bool has = __builtin_cpu_supports("avx512f");
  return has;
}

// DeltaBlock on two 512-bit vectors per row. The fused multiply-adds are
// the ones the compiler emits for DeltaBlock and Mixer::Linear (checked in
// the disassembly: u = fma(row, k decay, u), S' = fma(row, decay,
// k delta), y = fma(S', q, y)), so each element rounds the same way. (The
// prefill's DeltaColumns fuses the other product of S', so it is not.)
__attribute__((target("avx512f"))) void DeltaBlockAvx512(
    const float* src, float* dst, int dk, int j0, const float* k,
    const float* q, const float* v, float decay, float b, float* yh) {
  __m512 acc[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
  for (int i = 0; i < dk; ++i) {
    const float* row = src + static_cast<std::size_t>(i) * dk + j0;
    const __m512 ki = _mm512_set1_ps(k[i] * decay);
    for (int c = 0; c < 2; ++c) {
      acc[c] = _mm512_fmadd_ps(_mm512_loadu_ps(row + 16 * c), ki, acc[c]);
    }
  }
  __m512 delta[2];
  __m512 out[2];
  const __m512 bv = _mm512_set1_ps(b);
  for (int c = 0; c < 2; ++c) {
    delta[c] = _mm512_mul_ps(
        bv, _mm512_sub_ps(_mm512_loadu_ps(v + j0 + 16 * c), acc[c]));
    out[c] = _mm512_setzero_ps();
  }
  const __m512 dc = _mm512_set1_ps(decay);
  for (int i = 0; i < dk; ++i) {
    const float* row = src + static_cast<std::size_t>(i) * dk + j0;
    float* o = dst + static_cast<std::size_t>(i) * dk + j0;
    const __m512 ki = _mm512_set1_ps(k[i]);
    const __m512 qi = _mm512_set1_ps(q[i]);
    for (int c = 0; c < 2; ++c) {
      const __m512 updated = _mm512_fmadd_ps(_mm512_loadu_ps(row + 16 * c), dc,
                                             _mm512_mul_ps(ki, delta[c]));
      _mm512_storeu_ps(o + 16 * c, updated);
      out[c] = _mm512_fmadd_ps(updated, qi, out[c]);
    }
  }
  for (int c = 0; c < 2; ++c) {
    _mm512_storeu_ps(yh + j0 + 16 * c, out[c]);
  }
}
#endif

}  // namespace

struct DeltaNetStep::Impl {
  struct Layer {
    std::vector<float> conv;  // [channels][kernel]
    std::vector<float> ssm_a;
    std::vector<float> dt_bias;
    std::vector<float> ssm_norm;
    std::vector<float> alpha;      // [value heads][hidden]
    std::vector<float> beta;       // [value heads][hidden]
    std::vector<float> attn_norm;  // [hidden]
  };

  Config config;
  SpinPool& pool;
  std::vector<Layer> layers;  // by layer index (attention layers empty)
  bool wide = false;
  // RMS(h) * attn_norm per token, and the next conv window.
  std::vector<float> normed;
  std::vector<float> next_conv;
  // Reference(): the Linear input [qkv | z | alpha | beta] of a token.
  std::vector<float> ref_in;

  Impl(const gguf::File& file, const Config& c, SpinPool& p)
      : config(c), pool(p) {
#ifdef GUFO_DELTANET_AVX512
    wide = HasAvx512();
#endif
    if (c.linear_head_dim > kMaxHeadDim || c.linear_head_dim % kColumns != 0) {
      throw std::runtime_error("deltanet: unsupported head dim " +
                               std::to_string(c.linear_head_dim));
    }
    layers.resize(c.layers);
    for (int il = 0; il < c.layers; ++il) {
      if (c.IsFullAttention(il)) {
        continue;
      }
      const std::string p = "blk." + std::to_string(il) + ".";
      auto t = [&](const char* name) {
        return ToFloats(file.Tensor(p + name));
      };
      Layer& l = layers[il];
      l.conv = t("ssm_conv1d.weight");
      l.ssm_a = t("ssm_a");
      l.dt_bias = t("ssm_dt.bias");
      l.ssm_norm = t("ssm_norm.weight");
      l.alpha = t("ssm_alpha.weight");
      l.beta = t("ssm_beta.weight");
      l.attn_norm = t("attn_norm.weight");
    }
    normed.resize(static_cast<std::size_t>(kMaxTokens) * c.hidden);
  }

  // Value head `head`'s whole step for `count` tokens.
  void Head(int head, const Layer& lw, int count, const float* in,
            std::size_t in_stride, DecodeState::Layer& st, float* y,
            std::size_t y_stride) {
    const Config& c = config;
    const int hk = c.linear_key_heads;
    const int dk = c.linear_head_dim;
    const int key_dim = c.LinearKeyDim();
    const int channels = c.ConvChannels();
    const int window = c.conv_kernel - 1;
    const int d = c.hidden;
    const int kh = head % hk;  // GGUF order: value heads tile key heads
    // The first value head of each key head writes its q and k channels'
    // conv window.
    const bool owns_key = head < hk;
    // This head's channels: q and k of its key head, its own v.
    const int bases[3] = {kh * dk, key_dim + kh * dk, 2 * key_dim + head * dk};

    float u[kMaxTokens][3][kMaxHeadDim];
    // Token t's conv history row r: old window rows t.., then the tokens
    // before t (raw inputs), oldest first.
    auto history = [&](int r, int t) -> const float* {
      return r + t < window
                 ? st.conv.data() + static_cast<std::size_t>(r + t) * channels
                 : in + static_cast<std::size_t>(r + t - window) * in_stride;
    };
    for (int t = 0; t < count; ++t) {
      const float* x = in + static_cast<std::size_t>(t) * in_stride;
      const float* rows[8];
      for (int r = 0; r < window; ++r) {
        rows[r] = history(r, t);
      }
      // Causal depthwise conv over [window previous inputs, current], then
      // SiLU.
      for (int part = 0; part < 3; ++part) {
        for (int e = 0; e < dk; ++e) {
          const int ch = bases[part] + e;
          const float* kw =
              &lw.conv[static_cast<std::size_t>(ch) * c.conv_kernel];
          float acc = kw[window] * x[ch];
          for (int r = 0; r < window; ++r) {
            acc += kw[r] * rows[r][ch];
          }
          u[t][part][e] = Silu(acc);
        }
      }
      // The window after token t: into conv_saved after the first of a
      // pair, else the next window (swapped in after the dispatch).
      float* next =
          (count == 2 && t == 0) ? st.conv_saved.data() : next_conv.data();
      for (int r = 0; r < window; ++r) {
        const float* from = history(r, t + 1);
        float* to = next + static_cast<std::size_t>(r) * channels;
        for (int part = owns_key ? 0 : 2; part < 3; ++part) {
          std::memcpy(to + bases[part], from + bases[part],
                      static_cast<std::size_t>(dk) * sizeof(float));
        }
      }
      // L2-normalize q and k; scale q by 1/sqrt(dk).
      float* q = u[t][0];
      float* k = u[t][1];
      const float q_scale = 1.0f / std::sqrt(static_cast<float>(dk));
      for (float* v : {q, k}) {
        const float inv = 1.0f / std::sqrt(Dot(v, v, dk) + 1e-6f);
        const float scale = v == q ? inv * q_scale : inv;
        for (int e = 0; e < dk; ++e) {
          v[e] *= scale;
        }
      }
    }

    float decay[kMaxTokens];
    float b[kMaxTokens];
    for (int t = 0; t < count; ++t) {
      const float* n = normed.data() + static_cast<std::size_t>(t) * d;
      const float alpha =
          Dot(lw.alpha.data() + static_cast<std::size_t>(head) * d, n, d);
      const float beta =
          Dot(lw.beta.data() + static_cast<std::size_t>(head) * d, n, d);
      b[t] = Sigmoid(beta);
      decay[t] = std::exp(lw.ssm_a[head] * Softplus(alpha + lw.dt_bias[head]));
    }

    // The delta rule, block by block: the first token in place, a pair's
    // second from the first's result into ssm_spec.
    const std::size_t head_offset = static_cast<std::size_t>(head) * dk * dk;
    float* sm = &st.ssm[head_offset];
    float* spec = count == 2 ? &st.ssm_spec[head_offset] : nullptr;
    for (int j0 = 0; j0 < dk; j0 += kColumns) {
      for (int t = 0; t < count; ++t) {
        float* dst = t == 0 ? sm : spec;
        float* yh = y + static_cast<std::size_t>(t) * y_stride +
                    static_cast<std::size_t>(head) * dk;
#ifdef GUFO_DELTANET_AVX512
        if (wide) {
          DeltaBlockAvx512(sm, dst, dk, j0, u[t][1], u[t][0], u[t][2], decay[t],
                           b[t], yh);
          continue;
        }
#endif
        DeltaBlock(sm, dst, dk, j0, u[t][1], u[t][0], u[t][2], decay[t], b[t],
                   yh);
      }
    }

    // Gated RMSNorm: RMS(o) * w * silu(z).
    for (int t = 0; t < count; ++t) {
      float* yh = y + static_cast<std::size_t>(t) * y_stride +
                  static_cast<std::size_t>(head) * dk;
      RmsNorm(yh, lw.ssm_norm.data(), dk, c.rms_eps, yh);
      const float* zh = in + static_cast<std::size_t>(t) * in_stride +
                        channels + static_cast<std::size_t>(head) * dk;
      for (int j = 0; j < dk; ++j) {
        yh[j] *= Silu(zh[j]);
      }
    }
  }
};

DeltaNetStep::DeltaNetStep(const gguf::File& file, const Config& config,
                           SpinPool& pool)
    : impl_(std::make_unique<Impl>(file, config, pool)) {}

DeltaNetStep::~DeltaNetStep() = default;

bool DeltaNetStep::wide() const {
  return impl_->wide;
}

void DeltaNetStep::set_wide(bool wide) {
#ifdef GUFO_DELTANET_AVX512
  impl_->wide = wide && HasAvx512();
#else
  (void)wide;
#endif
}

void DeltaNetStep::Run(int il, int count, const float* in,
                       std::size_t in_stride, const float* h,
                       std::size_t h_stride, DecodeState::Layer& state,
                       float* y, std::size_t y_stride) {
  Impl& m = *impl_;
  const Config& c = m.config;
  if (count < 1 || count > kMaxTokens) {
    throw std::invalid_argument("deltanet: 1 or 2 tokens per step");
  }
  const auto& lw = m.layers.at(static_cast<std::size_t>(il));
  if (lw.conv.empty()) {
    throw std::invalid_argument("deltanet: layer " + std::to_string(il) +
                                " is not a DeltaNet layer");
  }
  const int d = c.hidden;
  for (int t = 0; t < count; ++t) {
    RmsNorm(h + static_cast<std::size_t>(t) * h_stride, lw.attn_norm.data(), d,
            c.rms_eps, m.normed.data() + static_cast<std::size_t>(t) * d);
  }
  m.next_conv.resize(state.conv.size());
  if (count == 2) {
    state.conv_saved.resize(state.conv.size());
    state.ssm_spec.resize(state.ssm.size());
  }
  m.pool.Run(static_cast<std::size_t>(c.linear_value_heads),
             [&](std::size_t head) {
               m.Head(static_cast<int>(head), lw, count, in, in_stride, state,
                      y, y_stride);
             });
  state.conv.swap(m.next_conv);
}

void DeltaNetStep::Reference(qwen35::Mixer& mixer, int il, int count,
                             const float* in, std::size_t in_stride,
                             const float* h, std::size_t h_stride,
                             DecodeState::Layer& st, float* y,
                             std::size_t y_stride) {
  Impl& m = *impl_;
  const Config& c = m.config;
  const auto& lw = m.layers.at(static_cast<std::size_t>(il));
  const int d = c.hidden;
  const int heads = c.linear_value_heads;
  const std::size_t rows =
      static_cast<std::size_t>(c.ConvChannels()) + c.LinearValueDim();
  m.ref_in.resize(rows + 2 * static_cast<std::size_t>(heads));
  for (int t = 0; t < count; ++t) {
    float* in_t = m.ref_in.data();
    std::memcpy(in_t, in + static_cast<std::size_t>(t) * in_stride,
                rows * sizeof(float));
    RmsNorm(h + static_cast<std::size_t>(t) * h_stride, lw.attn_norm.data(), d,
            c.rms_eps, m.normed.data());
    float* ab = in_t + rows;
    for (int head = 0; head < 2 * heads; ++head) {
      const float* w = (head < heads ? lw.alpha.data() : lw.beta.data()) +
                       static_cast<std::size_t>(head % heads) * d;
      ab[head] = Dot(w, m.normed.data(), d);
    }
    float* y_t = y + static_cast<std::size_t>(t) * y_stride;
    if (count == 2 && t == 1) {
      st.conv_saved = st.conv;
      st.ssm_spec.resize(st.ssm.size());
      mixer.Linear(il, in_t, st, y_t, st.ssm_spec.data());
    } else {
      mixer.Linear(il, in_t, st, y_t);
    }
  }
}

}  // namespace gufo_npu::qwen35_2b
