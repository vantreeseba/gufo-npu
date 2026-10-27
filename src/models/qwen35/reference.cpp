#include "models/qwen35/reference.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "core/parallel.hpp"
#include "core/quant.hpp"

namespace gufo_npu::qwen35 {
namespace {

using gguf::TensorInfo;

// The GGUF stores norm weights with the +1 of Qwen's zero-centred RMSNorm
// already added (except ssm_norm), so every norm here is RMS(x) * w.
void RmsNorm(const float* x, const float* w, int n, float eps, float* out) {
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += static_cast<double>(x[i]) * x[i];
  }
  const double inv = 1.0 / std::sqrt(sum / n + eps);
  for (int i = 0; i < n; ++i) {
    out[i] = static_cast<float>(x[i] * inv) * w[i];
  }
}

float Sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}
float Silu(float x) {
  return x * Sigmoid(x);
}
// torch.nn.functional.softplus with its default threshold of 20.
float Softplus(float x) {
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

std::vector<float> ToFloats(const TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

// y[t][r] = sum_c W[r][c] * x[t][c] for `tokens` rows of x. Each weight row
// is dequantized once and applied to every token.
void MatMul(const TensorInfo& w, const float* x, int tokens, float* y) {
  const auto in = static_cast<std::size_t>(w.RowElements());
  const auto out = static_cast<std::size_t>(w.Rows());
  const auto row_bytes = quant::RowBytes(w.type, in);
  ParallelFor(out, [&](std::size_t begin, std::size_t end) {
    std::vector<float> row(in);
    for (std::size_t r = begin; r < end; ++r) {
      quant::Dequantize(w.type, w.data + r * row_bytes, row.data(), in);
      for (int t = 0; t < tokens; ++t) {
        const float* xt = x + static_cast<std::size_t>(t) * in;
        double sum = 0;
        for (std::size_t c = 0; c < in; ++c) {
          sum += static_cast<double>(row[c]) * xt[c];
        }
        y[static_cast<std::size_t>(t) * out + r] = static_cast<float>(sum);
      }
    }
  });
}

void CheckShape(const TensorInfo& t, std::uint64_t cols, std::uint64_t rows) {
  if (t.RowElements() != cols || t.Rows() != rows) {
    throw std::runtime_error("qwen35: tensor " + t.name + " has shape [" +
                             std::to_string(t.RowElements()) + ", " +
                             std::to_string(t.Rows()) + "], expected [" +
                             std::to_string(cols) + ", " +
                             std::to_string(rows) + "]");
  }
}

}  // namespace

struct Reference::Weights {
  struct Layer {
    std::vector<float> attn_norm;
    std::vector<float> post_attention_norm;
    const TensorInfo* ffn_gate = nullptr;
    const TensorInfo* ffn_up = nullptr;
    const TensorInfo* ffn_down = nullptr;
    // Gated DeltaNet.
    const TensorInfo* qkv = nullptr;
    const TensorInfo* gate = nullptr;  // z
    const TensorInfo* alpha = nullptr;
    const TensorInfo* beta = nullptr;
    const TensorInfo* ssm_out = nullptr;
    std::vector<float> conv;      // [channels][kernel]
    std::vector<float> ssm_a;     // -exp(A_log), per value head
    std::vector<float> dt_bias;   // per value head
    std::vector<float> ssm_norm;  // per value-head dim, no +1
    // Gated full attention.
    const TensorInfo* q = nullptr;  // per head: [query, gate]
    const TensorInfo* k = nullptr;
    const TensorInfo* v = nullptr;
    const TensorInfo* o = nullptr;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
  };
  const TensorInfo* embed = nullptr;
  std::vector<float> output_norm;
  std::vector<Layer> layers;
  std::vector<float> rope_inv_freq;  // rope_dim / 2 entries
};

Reference::Reference(const gguf::File& file)
    : config_(ReadConfig(file)), weights_(std::make_unique<Weights>()) {
  const Config& c = config_;
  if (c.experts != 0) {
    throw std::runtime_error("qwen35 reference: MoE models are not supported");
  }
  auto& w = *weights_;
  w.embed = &file.Tensor("token_embd.weight");
  CheckShape(*w.embed, c.hidden, c.vocab);
  if (!c.tied_embeddings) {
    throw std::runtime_error("qwen35 reference: untied output.weight");
  }
  w.output_norm = ToFloats(file.Tensor("output_norm.weight"));

  const auto key_dim = static_cast<std::uint64_t>(c.LinearKeyDim());
  const auto value_dim = static_cast<std::uint64_t>(c.LinearValueDim());
  const auto attn_dim = static_cast<std::uint64_t>(c.heads * c.head_dim);
  const auto kv_dim = static_cast<std::uint64_t>(c.kv_heads * c.head_dim);
  for (int il = 0; il < c.layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto tensor = [&](const char* name) -> const TensorInfo& {
      return file.Tensor(p + name);
    };
    Weights::Layer layer;
    layer.attn_norm = ToFloats(tensor("attn_norm.weight"));
    layer.post_attention_norm = ToFloats(tensor("post_attention_norm.weight"));
    layer.ffn_gate = &tensor("ffn_gate.weight");
    layer.ffn_up = &tensor("ffn_up.weight");
    layer.ffn_down = &tensor("ffn_down.weight");
    CheckShape(*layer.ffn_gate, c.hidden, c.ffn);
    CheckShape(*layer.ffn_up, c.hidden, c.ffn);
    CheckShape(*layer.ffn_down, c.ffn, c.hidden);
    if (c.IsFullAttention(il)) {
      layer.q = &tensor("attn_q.weight");
      layer.k = &tensor("attn_k.weight");
      layer.v = &tensor("attn_v.weight");
      layer.o = &tensor("attn_output.weight");
      CheckShape(*layer.q, c.hidden, 2 * attn_dim);
      CheckShape(*layer.k, c.hidden, kv_dim);
      CheckShape(*layer.v, c.hidden, kv_dim);
      CheckShape(*layer.o, attn_dim, c.hidden);
      layer.q_norm = ToFloats(tensor("attn_q_norm.weight"));
      layer.k_norm = ToFloats(tensor("attn_k_norm.weight"));
    } else {
      layer.qkv = &tensor("attn_qkv.weight");
      layer.gate = &tensor("attn_gate.weight");
      layer.alpha = &tensor("ssm_alpha.weight");
      layer.beta = &tensor("ssm_beta.weight");
      layer.ssm_out = &tensor("ssm_out.weight");
      CheckShape(*layer.qkv, c.hidden, 2 * key_dim + value_dim);
      CheckShape(*layer.gate, c.hidden, value_dim);
      CheckShape(*layer.alpha, c.hidden, c.linear_value_heads);
      CheckShape(*layer.beta, c.hidden, c.linear_value_heads);
      CheckShape(*layer.ssm_out, value_dim, c.hidden);
      const auto& conv = tensor("ssm_conv1d.weight");
      CheckShape(conv, c.conv_kernel, c.ConvChannels());
      layer.conv = ToFloats(conv);
      layer.ssm_a = ToFloats(tensor("ssm_a"));
      layer.dt_bias = ToFloats(tensor("ssm_dt.bias"));
      layer.ssm_norm = ToFloats(tensor("ssm_norm.weight"));
    }
    w.layers.push_back(std::move(layer));
  }

  w.rope_inv_freq.resize(c.rope_dim / 2);
  for (int i = 0; i < c.rope_dim / 2; ++i) {
    w.rope_inv_freq[i] = static_cast<float>(
        1.0 / std::pow(static_cast<double>(c.rope_theta),
                       2.0 * i / static_cast<double>(c.rope_dim)));
  }
}

Reference::~Reference() = default;

State Reference::NewState() const {
  const Config& c = config_;
  State state;
  state.linear.resize(c.layers);
  state.attention.resize(c.layers);
  for (int il = 0; il < c.layers; ++il) {
    if (!c.IsFullAttention(il)) {
      state.linear[il].conv.assign(
          static_cast<std::size_t>(c.conv_kernel - 1) * c.ConvChannels(), 0.0f);
      state.linear[il].ssm.assign(
          static_cast<std::size_t>(c.linear_value_heads) * c.linear_head_dim *
              c.linear_head_dim,
          0.0f);
    }
  }
  return state;
}

ForwardOutput Reference::Forward(const std::vector<int>& tokens, State& state,
                                 bool want_logits, bool want_hidden) const {
  const Config& c = config_;
  const auto& w = *weights_;
  const int n = static_cast<int>(tokens.size());
  const int d = c.hidden;
  const auto nd = static_cast<std::size_t>(n) * d;
  ForwardOutput result;

  // Embeddings: row `token` of token_embd.
  std::vector<float> h(nd);
  const auto embed_row = quant::RowBytes(w.embed->type, d);
  for (int t = 0; t < n; ++t) {
    if (tokens[t] < 0 || tokens[t] >= c.vocab) {
      throw std::out_of_range("qwen35: token id out of range");
    }
    quant::Dequantize(
        w.embed->type,
        w.embed->data + static_cast<std::size_t>(tokens[t]) * embed_row,
        h.data() + static_cast<std::size_t>(t) * d, d);
  }
  if (want_hidden) {
    result.hidden.insert(result.hidden.end(), h.begin(), h.end());
  }

  std::vector<float> normed(nd);
  for (int il = 0; il < c.layers; ++il) {
    const auto& layer = w.layers[il];
    for (int t = 0; t < n; ++t) {
      RmsNorm(&h[static_cast<std::size_t>(t) * d], layer.attn_norm.data(), d,
              c.rms_eps, &normed[static_cast<std::size_t>(t) * d]);
    }
    std::vector<float> mixed(nd);

    if (c.IsFullAttention(il)) {
      const int hd = c.head_dim;
      const int kv = c.kv_heads * hd;
      const int group = c.heads / c.kv_heads;
      std::vector<float> qg(static_cast<std::size_t>(n) * 2 * c.heads * hd);
      std::vector<float> k(static_cast<std::size_t>(n) * kv);
      std::vector<float> v(static_cast<std::size_t>(n) * kv);
      MatMul(*layer.q, normed.data(), n, qg.data());
      MatMul(*layer.k, normed.data(), n, k.data());
      MatMul(*layer.v, normed.data(), n, v.data());

      auto rope = [&](float* x, int pos) {
        const int half = c.rope_dim / 2;
        for (int i = 0; i < half; ++i) {
          const double angle = static_cast<double>(pos) * w.rope_inv_freq[i];
          const auto cs = static_cast<float>(std::cos(angle));
          const auto sn = static_cast<float>(std::sin(angle));
          const float a = x[i];
          const float b = x[i + half];
          x[i] = a * cs - b * sn;
          x[i + half] = a * sn + b * cs;
        }
      };

      auto& cache = state.attention[il];
      std::vector<float> ctx(static_cast<std::size_t>(c.heads) * hd);
      std::vector<float> query(hd);
      for (int t = 0; t < n; ++t) {
        const int pos = state.position + t;
        float* kt = &k[static_cast<std::size_t>(t) * kv];
        for (int kh = 0; kh < c.kv_heads; ++kh) {
          float* head = kt + static_cast<std::size_t>(kh) * hd;
          RmsNorm(head, layer.k_norm.data(), hd, c.rms_eps, head);
          rope(head, pos);
        }
        cache.keys.insert(cache.keys.end(), kt, kt + kv);
        cache.values.insert(cache.values.end(),
                            &v[static_cast<std::size_t>(t) * kv],
                            &v[static_cast<std::size_t>(t) * kv] + kv);
        const int length = pos + 1;
        const float* qt = &qg[static_cast<std::size_t>(t) * 2 * c.heads * hd];
        for (int qh = 0; qh < c.heads; ++qh) {
          const float* raw = qt + static_cast<std::size_t>(qh) * 2 * hd;
          RmsNorm(raw, layer.q_norm.data(), hd, c.rms_eps, query.data());
          rope(query.data(), pos);
          const int kh = qh / group;
          std::vector<double> scores(length);
          double max_score = -1e300;
          for (int j = 0; j < length; ++j) {
            const float* key =
                &cache.keys[static_cast<std::size_t>(j) * kv + kh * hd];
            double dot = 0;
            for (int e = 0; e < hd; ++e) {
              dot += static_cast<double>(query[e]) * key[e];
            }
            scores[j] = dot / std::sqrt(static_cast<double>(hd));
            max_score = std::max(max_score, scores[j]);
          }
          double total = 0;
          for (auto& s : scores) {
            s = std::exp(s - max_score);
            total += s;
          }
          const float* gate = raw + hd;
          float* out = &ctx[static_cast<std::size_t>(qh) * hd];
          for (int e = 0; e < hd; ++e) {
            double acc = 0;
            for (int j = 0; j < length; ++j) {
              acc +=
                  scores[j] *
                  cache.values[static_cast<std::size_t>(j) * kv + kh * hd + e];
            }
            out[e] = static_cast<float>(acc / total) * Sigmoid(gate[e]);
          }
        }
        MatMul(*layer.o, ctx.data(), 1,
               &mixed[static_cast<std::size_t>(t) * d]);
      }
    } else {
      const int hk = c.linear_key_heads;
      const int hv = c.linear_value_heads;
      const int dk = c.linear_head_dim;
      const int key_dim = c.LinearKeyDim();
      const int value_dim = c.LinearValueDim();
      const int channels = c.ConvChannels();
      const int window = c.conv_kernel - 1;
      std::vector<float> qkv(static_cast<std::size_t>(n) * channels);
      std::vector<float> z(static_cast<std::size_t>(n) * value_dim);
      std::vector<float> alpha(static_cast<std::size_t>(n) * hv);
      std::vector<float> beta(static_cast<std::size_t>(n) * hv);
      MatMul(*layer.qkv, normed.data(), n, qkv.data());
      MatMul(*layer.gate, normed.data(), n, z.data());
      MatMul(*layer.alpha, normed.data(), n, alpha.data());
      MatMul(*layer.beta, normed.data(), n, beta.data());

      auto& st = state.linear[il];
      std::vector<float> u(channels);
      std::vector<float> y(value_dim);
      std::vector<float> retrieved(dk);
      for (int t = 0; t < n; ++t) {
        const float* xt = &qkv[static_cast<std::size_t>(t) * channels];
        // Causal depthwise conv over [3 previous inputs, current], then SiLU.
        for (int ch = 0; ch < channels; ++ch) {
          const float* kw =
              &layer.conv[static_cast<std::size_t>(ch) * c.conv_kernel];
          double acc = static_cast<double>(kw[window]) * xt[ch];
          for (int s = 0; s < window; ++s) {
            acc += static_cast<double>(kw[s]) *
                   st.conv[static_cast<std::size_t>(s) * channels + ch];
          }
          u[ch] = Silu(static_cast<float>(acc));
        }
        for (int s = 0; s + 1 < window; ++s) {
          std::copy_n(&st.conv[static_cast<std::size_t>(s + 1) * channels],
                      channels,
                      &st.conv[static_cast<std::size_t>(s) * channels]);
        }
        std::copy_n(xt, channels,
                    &st.conv[static_cast<std::size_t>(window - 1) * channels]);

        // L2-normalise q and k per key head; scale q by 1/sqrt(dk).
        float* q = u.data();
        float* k = u.data() + key_dim;
        const float* v = u.data() + 2 * key_dim;
        for (int head = 0; head < hk; ++head) {
          for (float* vec : {q + head * dk, k + head * dk}) {
            double sum = 0;
            for (int e = 0; e < dk; ++e) {
              sum += static_cast<double>(vec[e]) * vec[e];
            }
            const double inv = 1.0 / std::sqrt(sum + 1e-6);
            for (int e = 0; e < dk; ++e) {
              vec[e] = static_cast<float>(vec[e] * inv);
            }
          }
          const double scale = 1.0 / std::sqrt(static_cast<double>(dk));
          for (int e = 0; e < dk; ++e) {
            q[head * dk + e] = static_cast<float>(q[head * dk + e] * scale);
          }
        }

        for (int head = 0; head < hv; ++head) {
          const int kh = head % hk;  // GGUF order: value heads tile key heads
          const float* qh = q + kh * dk;
          const float* kvec = k + kh * dk;
          const float* vh = v + head * dk;
          const float b =
              Sigmoid(beta[static_cast<std::size_t>(t) * hv + head]);
          const float g =
              layer.ssm_a[head] *
              Softplus(alpha[static_cast<std::size_t>(t) * hv + head] +
                       layer.dt_bias[head]);
          const auto decay = static_cast<float>(std::exp(g));
          float* s = &st.ssm[static_cast<std::size_t>(head) * dk * dk];
          for (int i = 0; i < dk * dk; ++i) {
            s[i] *= decay;
          }
          // S is [i (key)][j (value)]: retrieve u = S^T k, then rank-1 update.
          for (int j = 0; j < dk; ++j) {
            double acc = 0;
            for (int i = 0; i < dk; ++i) {
              acc += static_cast<double>(s[i * dk + j]) * kvec[i];
            }
            retrieved[j] = static_cast<float>(acc);
          }
          for (int j = 0; j < dk; ++j) {
            const float delta = b * (vh[j] - retrieved[j]);
            for (int i = 0; i < dk; ++i) {
              s[i * dk + j] += kvec[i] * delta;
            }
          }
          float* yh = &y[static_cast<std::size_t>(head) * dk];
          for (int j = 0; j < dk; ++j) {
            double acc = 0;
            for (int i = 0; i < dk; ++i) {
              acc += static_cast<double>(s[i * dk + j]) * qh[i];
            }
            yh[j] = static_cast<float>(acc);
          }
          // Gated RMSNorm: RMS(o) * w * silu(z).
          RmsNorm(yh, layer.ssm_norm.data(), dk, c.rms_eps, yh);
          const float* zh =
              &z[static_cast<std::size_t>(t) * value_dim + head * dk];
          for (int j = 0; j < dk; ++j) {
            yh[j] *= Silu(zh[j]);
          }
        }
        MatMul(*layer.ssm_out, y.data(), 1,
               &mixed[static_cast<std::size_t>(t) * d]);
      }
    }

    for (std::size_t i = 0; i < nd; ++i) {
      h[i] += mixed[i];
    }

    // SwiGLU MLP.
    for (int t = 0; t < n; ++t) {
      RmsNorm(&h[static_cast<std::size_t>(t) * d],
              layer.post_attention_norm.data(), d, c.rms_eps,
              &normed[static_cast<std::size_t>(t) * d]);
    }
    const auto nf = static_cast<std::size_t>(n) * c.ffn;
    std::vector<float> gate(nf);
    std::vector<float> up(nf);
    MatMul(*layer.ffn_gate, normed.data(), n, gate.data());
    MatMul(*layer.ffn_up, normed.data(), n, up.data());
    for (std::size_t i = 0; i < nf; ++i) {
      gate[i] = Silu(gate[i]) * up[i];
    }
    std::vector<float> down(nd);
    MatMul(*layer.ffn_down, gate.data(), n, down.data());
    for (std::size_t i = 0; i < nd; ++i) {
      h[i] += down[i];
    }
    if (want_hidden) {
      result.hidden.insert(result.hidden.end(), h.begin(), h.end());
    }
  }
  state.position += n;

  if (want_logits) {
    for (int t = 0; t < n; ++t) {
      RmsNorm(&h[static_cast<std::size_t>(t) * d], w.output_norm.data(), d,
              c.rms_eps, &normed[static_cast<std::size_t>(t) * d]);
    }
    result.logits.resize(static_cast<std::size_t>(n) * c.vocab);
    MatMul(*w.embed, normed.data(), n, result.logits.data());
  }
  return result;
}

}  // namespace gufo_npu::qwen35
