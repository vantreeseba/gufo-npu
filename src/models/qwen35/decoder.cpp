#include "models/qwen35/decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/quant.hpp"
#include "core/thread_pool.hpp"
#include "core/vec.hpp"

namespace gufo_npu::qwen35 {
namespace {

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

using vec::Dot;
using vec::RmsNorm;

float Sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}
float Silu(float x) {
  return x * Sigmoid(x);
}
float Softplus(float x) {
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

void CheckDense(const Config& c) {
  if (c.experts != 0 || !c.tied_embeddings) {
    throw std::runtime_error(
        "qwen35 decoder: dense, tied-embedding models only");
  }
}

}  // namespace

struct Mixer::Weights {
  struct Layer {
    std::vector<float> conv;  // [channels][kernel]
    std::vector<float> ssm_a;
    std::vector<float> dt_bias;
    std::vector<float> ssm_norm;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
  };
  std::vector<Layer> layers;
  std::vector<float> rope_inv_freq;
};

struct Mixer::Scratch {
  std::vector<float> u;            // conv output
  std::vector<float> delta;        // DeltaNet rank-1 update, per value head
  std::vector<float> queries;      // RoPE'd query per head
  std::vector<float> head_scores;  // attention weights per job and head
  std::vector<float> partials;     // per job and head: weighted values, max,
                                   // sum
  int rope_position = -1;          // the position rope_cos/sin hold
  std::vector<float> rope_cos, rope_sin;
  // DeltaNet heads run in parallel; a job is tens of microseconds.
  SpinPool pool{std::min<std::size_t>(
      7, std::max<std::size_t>(std::thread::hardware_concurrency(), 1) - 1)};
};

Mixer::Mixer(const gguf::File& file)
    : config_(ReadConfig(file)),
      weights_(std::make_unique<Weights>()),
      scratch_(std::make_unique<Scratch>()) {
  const Config& c = config_;
  CheckDense(c);
  auto& w = *weights_;
  for (int il = 0; il < c.layers + c.mtp_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto t = [&](const char* name) { return ToFloats(file.Tensor(p + name)); };
    Weights::Layer layer;
    if (c.IsFullAttention(il)) {
      layer.q_norm = t("attn_q_norm.weight");
      layer.k_norm = t("attn_k_norm.weight");
    } else {
      layer.conv = t("ssm_conv1d.weight");
      layer.ssm_a = t("ssm_a");
      layer.dt_bias = t("ssm_dt.bias");
      layer.ssm_norm = t("ssm_norm.weight");
    }
    w.layers.push_back(std::move(layer));
  }
  w.rope_inv_freq.resize(c.rope_dim / 2);
  for (int i = 0; i < c.rope_dim / 2; ++i) {
    w.rope_inv_freq[i] = static_cast<float>(
        1.0 / std::pow(static_cast<double>(c.rope_theta),
                       2.0 * i / static_cast<double>(c.rope_dim)));
  }
  auto& s = *scratch_;
  s.u.resize(c.ConvChannels());
  s.delta.resize(static_cast<std::size_t>(c.linear_value_heads) *
                 c.linear_head_dim);
}

Mixer::~Mixer() = default;

DecodeState Mixer::NewState() const {
  const Config& c = config_;
  DecodeState state;
  state.layers.resize(c.layers + c.mtp_layers);
  for (int il = 0; il < c.layers + c.mtp_layers; ++il) {
    if (!c.IsFullAttention(il)) {
      state.layers[il].conv.assign(
          static_cast<std::size_t>(c.conv_kernel - 1) * c.ConvChannels(), 0.0f);
      state.layers[il].ssm.assign(
          static_cast<std::size_t>(c.linear_value_heads) * c.linear_head_dim *
              c.linear_head_dim,
          0.0f);
    }
  }
  return state;
}

void Mixer::Linear(int il, const float* in, DecodeState::Layer& st, float* y,
                   float* ssm_out) {
  const Config& c = config_;
  const auto& lw = weights_->layers[il];
  auto& s = *scratch_;
  const int hk = c.linear_key_heads;
  const int hv = c.linear_value_heads;
  const int dk = c.linear_head_dim;
  const int key_dim = c.LinearKeyDim();
  const int value_dim = c.LinearValueDim();
  const int channels = c.ConvChannels();
  const int window = c.conv_kernel - 1;
  const float* x = in;
  const float* z = x + channels;
  const float* alpha = z + value_dim;
  const float* beta = alpha + hv;

  // Causal depthwise conv over [3 previous inputs, current], then SiLU, in
  // chunks across the pool.
  constexpr int kConvChunk = 512;
  s.pool.Run(
      static_cast<std::size_t>((channels + kConvChunk - 1) / kConvChunk),
      [&](std::size_t chunk) {
        const int end =
            std::min(channels, (static_cast<int>(chunk) + 1) * kConvChunk);
        for (int ch = static_cast<int>(chunk) * kConvChunk; ch < end; ++ch) {
          const float* kw =
              &lw.conv[static_cast<std::size_t>(ch) * c.conv_kernel];
          float acc = kw[window] * x[ch];
          for (int t = 0; t < window; ++t) {
            acc += kw[t] * st.conv[static_cast<std::size_t>(t) * channels + ch];
          }
          s.u[ch] = Silu(acc);
        }
      });
  std::memmove(st.conv.data(), st.conv.data() + channels,
               static_cast<std::size_t>(window - 1) * channels * sizeof(float));
  std::copy_n(x, channels,
              st.conv.data() + static_cast<std::size_t>(window - 1) * channels);

  // L2-normalize q and k per key head; scale q by 1/sqrt(dk).
  float* q = s.u.data();
  float* k = q + key_dim;
  const float* v = k + key_dim;
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(dk));
  for (int head = 0; head < hk; ++head) {
    for (float* x : {q + head * dk, k + head * dk}) {
      const float inv = 1.0f / std::sqrt(Dot(x, x, dk) + 1e-6f);
      const float scale = x == q + head * dk ? inv * q_scale : inv;
      for (int e = 0; e < dk; ++e) {
        x[e] *= scale;
      }
    }
  }

  // Heads are independent: run them across the pool, each with its own
  // delta scratch.
  s.pool.Run(static_cast<std::size_t>(hv), [&](std::size_t h) {
    const int head = static_cast<int>(h);
    float* delta = s.delta.data() + static_cast<std::size_t>(head) * dk;
    const int kh = head % hk;  // GGUF order: value heads tile key heads
    const float* qh = q + kh * dk;
    const float* kvec = k + kh * dk;
    const float* vh = v + head * dk;
    const float b = Sigmoid(beta[head]);
    const float decay =
        std::exp(lw.ssm_a[head] * Softplus(alpha[head] + lw.dt_bias[head]));
    const std::size_t head_offset = static_cast<std::size_t>(head) * dk * dk;
    float* sm = &st.ssm[head_offset];
    // The updated state: in place, or into ssm_out.
    float* dst = ssm_out != nullptr ? ssm_out + head_offset : sm;
    float* yh = y + head * dk;

    // S is [i (key)][j (value)]. Pass 1: retrieve (decay S)^T k.
    std::fill_n(delta, dk, 0.0f);
    for (int i = 0; i < dk; ++i) {
      const float* row = sm + static_cast<std::size_t>(i) * dk;
      const float ki = kvec[i] * decay;
      for (int j = 0; j < dk; ++j) {
        delta[j] += row[j] * ki;
      }
    }
    for (int j = 0; j < dk; ++j) {
      delta[j] = b * (vh[j] - delta[j]);
    }
    // Pass 2: S' = decay S + k delta^T, and read out S'^T q.
    std::fill_n(yh, dk, 0.0f);
    for (int i = 0; i < dk; ++i) {
      const float* row = sm + static_cast<std::size_t>(i) * dk;
      float* out = dst + static_cast<std::size_t>(i) * dk;
      const float ki = kvec[i];
      const float qi = qh[i];
      for (int j = 0; j < dk; ++j) {
        out[j] = row[j] * decay + ki * delta[j];
        yh[j] += out[j] * qi;
      }
    }
    // Gated RMSNorm: RMS(o) * w * silu(z).
    RmsNorm(yh, lw.ssm_norm.data(), dk, c.rms_eps, yh);
    const float* zh = z + head * dk;
    for (int j = 0; j < dk; ++j) {
      yh[j] *= Silu(zh[j]);
    }
  });
}

void Mixer::Attention(int il, float* in, int position, DecodeState::Layer& st,
                      float* ctx) {
  const Config& c = config_;
  const auto& lw = weights_->layers[il];
  auto& s = *scratch_;
  const int hd = c.head_dim;
  const int kv = c.kv_heads * hd;
  const int group = c.heads / c.kv_heads;
  const int half = c.rope_dim / 2;
  float* qg = in;
  float* k = qg + 2 * c.heads * hd;
  const float* v = k + kv;

  // cos/sin for this position, computed once and shared by every head and
  // layer at the same position.
  if (s.rope_position != position) {
    s.rope_cos.resize(half);
    s.rope_sin.resize(half);
    for (int i = 0; i < half; ++i) {
      const double angle =
          static_cast<double>(position) * weights_->rope_inv_freq[i];
      s.rope_cos[i] = static_cast<float>(std::cos(angle));
      s.rope_sin[i] = static_cast<float>(std::sin(angle));
    }
    s.rope_position = position;
  }
  auto rope = [&](float* x) {
    for (int i = 0; i < half; ++i) {
      const float cs = s.rope_cos[i];
      const float sn = s.rope_sin[i];
      const float a = x[i];
      const float b = x[i + half];
      x[i] = a * cs - b * sn;
      x[i + half] = a * sn + b * cs;
    }
  };
  for (int kh = 0; kh < c.kv_heads; ++kh) {
    float* head = k + kh * hd;
    RmsNorm(head, lw.k_norm.data(), hd, c.rms_eps, head);
    rope(head);
  }
  const std::size_t cached = st.keys.size();
  st.keys.resize(cached + kv);
  st.values.resize(cached + kv);
  vec::ToHalf(k, st.keys.data() + cached, kv);
  vec::ToHalf(v, st.values.data() + cached, kv);

  // Every cached position (equals position + 1 for the decoder layers; the
  // MTP layer's cache can start later).
  const int length = static_cast<int>(st.keys.size() / kv);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  s.queries.resize(static_cast<std::size_t>(c.heads) * hd);
  for (int qh = 0; qh < c.heads; ++qh) {
    float* query = s.queries.data() + static_cast<std::size_t>(qh) * hd;
    RmsNorm(qg + static_cast<std::size_t>(qh) * 2 * hd, lw.q_norm.data(), hd,
            c.rms_eps, query);
    rope(query);
  }

  // Split-K (flash decoding): a job takes one KV head and a slice of the
  // positions, reads each key and value once for the `group` query heads
  // that share them, and leaves a partial softmax (max, sum, weighted
  // values) per query head; the slices are merged after. Reading the cache
  // is the cost at long context, and enough slices keep several cores'
  // worth of bandwidth busy.
  constexpr int kMinSlice = 128;
  const int slices =
      std::clamp(length / kMinSlice, 1, static_cast<int>(s.pool.size()));
  const int slice_len = (length + slices - 1) / slices;
  const int jobs = c.kv_heads * slices;
  const std::size_t partial = static_cast<std::size_t>(hd) + 2;  // acc, m, l
  s.head_scores.resize(static_cast<std::size_t>(jobs) * group * slice_len);
  s.partials.resize(static_cast<std::size_t>(jobs) * group * partial);
  s.pool.Run(static_cast<std::size_t>(jobs), [&](std::size_t job) {
    const int kh = static_cast<int>(job) / slices;
    const int begin = static_cast<int>(job) % slices * slice_len;
    const int end = std::min(length, begin + slice_len);
    float* scores = s.head_scores.data() + job * group * slice_len;
    float* out = s.partials.data() + job * group * partial;
    const float* queries =
        s.queries.data() + static_cast<std::size_t>(kh) * group * hd;
    for (int j = begin; j < end; ++j) {
      const std::uint16_t* key =
          &st.keys[static_cast<std::size_t>(j) * kv + kh * hd];
      for (int g = 0; g < group; ++g) {
        scores[g * slice_len + (j - begin)] =
            vec::DotHalf(queries + static_cast<std::size_t>(g) * hd, key, hd) *
            scale;
      }
    }
    for (int g = 0; g < group; ++g) {
      float* sc = scores + g * slice_len;
      float max_score = -INFINITY;
      for (int j = 0; j < end - begin; ++j) {
        max_score = std::max(max_score, sc[j]);
      }
      float total = 0;
      for (int j = 0; j < end - begin; ++j) {
        sc[j] = std::exp(sc[j] - max_score);
        total += sc[j];
      }
      float* acc = out + g * partial;
      std::fill_n(acc, hd, 0.0f);
      acc[hd] = max_score;
      acc[hd + 1] = total;
    }
    for (int j = begin; j < end; ++j) {
      const std::uint16_t* value =
          &st.values[static_cast<std::size_t>(j) * kv + kh * hd];
      for (int g = 0; g < group; ++g) {
        vec::AddScaledHalf(scores[g * slice_len + (j - begin)], value,
                           out + g * partial, hd);
      }
    }
  });

  // Merge the slices per query head, then the output gate.
  for (int qh = 0; qh < c.heads; ++qh) {
    const int kh = qh / group;
    const int g = qh % group;
    auto slice = [&](int sl) {
      return s.partials.data() +
             (static_cast<std::size_t>(kh * slices + sl) * group + g) * partial;
    };
    float max_score = -INFINITY;
    for (int sl = 0; sl < slices; ++sl) {
      max_score = std::max(max_score, slice(sl)[hd]);
    }
    float* out = ctx + static_cast<std::size_t>(qh) * hd;
    std::fill_n(out, hd, 0.0f);
    float total = 0;
    for (int sl = 0; sl < slices; ++sl) {
      const float* acc = slice(sl);
      const float w = std::exp(acc[hd] - max_score);
      total += w * acc[hd + 1];
      for (int e = 0; e < hd; ++e) {
        out[e] += w * acc[e];
      }
    }
    const float inv_total = 1.0f / total;
    const float* gate = qg + static_cast<std::size_t>(qh) * 2 * hd + hd;
    for (int e = 0; e < hd; ++e) {
      out[e] *= inv_total * Sigmoid(gate[e]);
    }
  }
}

struct Decoder::Weights {
  struct Layer {
    std::vector<float> attn_norm;
    std::vector<float> post_attention_norm;
  };
  const gguf::TensorInfo* embed = nullptr;
  std::vector<float> output_norm;
  std::vector<Layer> layers;
};

struct Decoder::Scratch {
  std::vector<float> h;       // residual stream
  std::vector<float> normed;  // hidden
  std::vector<float> in;      // a stage's fused output
  std::vector<float> mixed;   // mixer output (DeltaNet y or attention ctx)
  std::vector<float> out;     // hidden-sized projection output
  std::vector<float> ffn;     // gate | up
  std::vector<float> act;     // silu(gate) * up
  std::vector<float> logits;
};

Decoder::Decoder(const gguf::File& file, const Projections& projections)
    : mixer_(file),
      projections_(projections),
      weights_(std::make_unique<Weights>()),
      scratch_(std::make_unique<Scratch>()) {
  const Config& c = mixer_.config();
  auto& w = *weights_;
  w.embed = &file.Tensor("token_embd.weight");
  w.output_norm = ToFloats(file.Tensor("output_norm.weight"));
  for (int il = 0; il < c.layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    w.layers.push_back(
        {ToFloats(file.Tensor(p + "attn_norm.weight")),
         ToFloats(file.Tensor(p + "post_attention_norm.weight"))});
  }
  auto& s = *scratch_;
  s.h.resize(c.hidden);
  s.normed.resize(c.hidden);
  s.in.resize(std::max(StageOutputs(c, Stage::kLinearIn, 0),
                       StageOutputs(c, Stage::kAttentionIn, 3)));
  s.mixed.resize(std::max(c.LinearValueDim(), c.heads * c.head_dim));
  s.out.resize(c.hidden);
  s.ffn.resize(2 * static_cast<std::size_t>(c.ffn));
  s.act.resize(c.ffn);
  s.logits.resize(c.vocab);
}

Decoder::~Decoder() = default;

const std::vector<float>& Decoder::Step(int token, State& state) {
  const Config& c = mixer_.config();
  auto& w = *weights_;
  auto& s = *scratch_;
  const int d = c.hidden;
  if (token < 0 || token >= c.vocab) {
    throw std::out_of_range("qwen35: token id out of range");
  }
  quant::Dequantize(w.embed->type,
                    w.embed->data + static_cast<std::size_t>(token) *
                                        quant::RowBytes(w.embed->type, d),
                    s.h.data(), d);

  for (int il = 0; il < c.layers; ++il) {
    const auto& lw = w.layers[il];
    RmsNorm(s.h.data(), lw.attn_norm.data(), d, c.rms_eps, s.normed.data());
    if (c.IsFullAttention(il)) {
      projections_.Apply(Stage::kAttentionIn, il, s.normed.data(), 1,
                         s.in.data());
      mixer_.Attention(il, s.in.data(), state.position, state.layers[il],
                       s.mixed.data());
    } else {
      projections_.Apply(Stage::kLinearIn, il, s.normed.data(), 1, s.in.data());
      mixer_.Linear(il, s.in.data(), state.layers[il], s.mixed.data());
    }
    projections_.Apply(Stage::kMixerOut, il, s.mixed.data(), 1, s.out.data());
    for (int i = 0; i < d; ++i) {
      s.h[i] += s.out[i];
    }
    // SwiGLU MLP.
    RmsNorm(s.h.data(), lw.post_attention_norm.data(), d, c.rms_eps,
            s.normed.data());
    projections_.Apply(Stage::kFfnIn, il, s.normed.data(), 1, s.ffn.data());
    for (int i = 0; i < c.ffn; ++i) {
      s.act[i] = Silu(s.ffn[i]) * s.ffn[c.ffn + i];
    }
    projections_.Apply(Stage::kFfnDown, il, s.act.data(), 1, s.out.data());
    for (int i = 0; i < d; ++i) {
      s.h[i] += s.out[i];
    }
  }
  ++state.position;

  RmsNorm(s.h.data(), w.output_norm.data(), d, c.rms_eps, s.normed.data());
  projections_.Apply(Stage::kLmHead, -1, s.normed.data(), 1, s.logits.data());
  return s.logits;
}

}  // namespace gufo_npu::qwen35
