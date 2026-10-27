#include "models/qwen35/reference.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/quant.hpp"
#include "models/qwen35/moe.hpp"

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

// Splits `tokens` rows of concatenated outputs into one buffer per part.
void Split(const std::vector<float>& fused, int tokens,
           std::initializer_list<std::pair<std::vector<float>*, int>> parts) {
  std::size_t stride = 0;
  for (const auto& part : parts) {
    stride += static_cast<std::size_t>(part.second);
  }
  std::size_t offset = 0;
  for (const auto& [out, width] : parts) {
    for (int t = 0; t < tokens; ++t) {
      std::copy_n(&fused[static_cast<std::size_t>(t) * stride + offset], width,
                  out->data() + static_cast<std::size_t>(t) * width);
    }
    offset += static_cast<std::size_t>(width);
  }
}

}  // namespace

struct Reference::Weights {
  struct Layer {
    std::vector<float> attn_norm;
    std::vector<float> post_attention_norm;
    // Gated DeltaNet.
    std::vector<float> conv;      // [channels][kernel]
    std::vector<float> ssm_a;     // -exp(A_log), per value head
    std::vector<float> dt_bias;   // per value head
    std::vector<float> ssm_norm;  // per value-head dim, no +1
    // Gated full attention.
    std::vector<float> q_norm;
    std::vector<float> k_norm;
  };
  const TensorInfo* embed = nullptr;
  std::vector<float> output_norm;
  std::vector<Layer> layers;  // decoder layers, then the MTP layer
  // MTP (nextn) norms.
  std::vector<float> mtp_enorm;
  std::vector<float> mtp_hnorm;
  std::vector<float> mtp_head_norm;
  std::vector<float> rope_inv_freq;  // rope_dim / 2 entries
};

Reference::Reference(const gguf::File& file)
    : config_(ReadConfig(file)),
      weights_(std::make_unique<Weights>()),
      gguf_projections_(std::make_unique<GgufProjections>(
          file, GgufProjections::Precision::kDouble)) {
  LoadWeights(file);
}

void Reference::LoadWeights(const gguf::File& file) {
  const Config& c = config_;
  auto& w = *weights_;
  w.embed = &file.Tensor("token_embd.weight");
  if (w.embed->RowElements() != static_cast<std::uint64_t>(c.hidden)) {
    throw std::runtime_error("qwen35: token_embd rows are not hidden wide");
  }
  w.output_norm = ToFloats(file.Tensor("output_norm.weight"));

  // The decoder layers, then the MTP layer (full attention) if present.
  const bool mtp = HasMtpLayer(file, c);
  for (int il = 0; il < c.layers + (mtp ? 1 : 0); ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto tensor = [&](const char* name) -> const TensorInfo& {
      return file.Tensor(p + name);
    };
    Weights::Layer layer;
    layer.attn_norm = ToFloats(tensor("attn_norm.weight"));
    layer.post_attention_norm = ToFloats(tensor("post_attention_norm.weight"));
    if (c.IsFullAttention(il)) {
      layer.q_norm = ToFloats(tensor("attn_q_norm.weight"));
      layer.k_norm = ToFloats(tensor("attn_k_norm.weight"));
    } else {
      const auto& conv = tensor("ssm_conv1d.weight");
      if (conv.shape != std::vector<std::uint64_t>{
                            static_cast<std::uint64_t>(c.conv_kernel),
                            static_cast<std::uint64_t>(c.ConvChannels())}) {
        throw std::runtime_error("qwen35: unexpected shape of " + conv.name);
      }
      layer.conv = ToFloats(conv);
      layer.ssm_a = ToFloats(tensor("ssm_a"));
      layer.dt_bias = ToFloats(tensor("ssm_dt.bias"));
      layer.ssm_norm = ToFloats(tensor("ssm_norm.weight"));
    }
    w.layers.push_back(std::move(layer));
  }
  if (mtp) {
    const std::string p = "blk." + std::to_string(c.layers) + ".nextn.";
    w.mtp_enorm = ToFloats(file.Tensor(p + "enorm.weight"));
    w.mtp_hnorm = ToFloats(file.Tensor(p + "hnorm.weight"));
    w.mtp_head_norm = ToFloats(file.Tensor(p + "shared_head_norm.weight"));
  }

  w.rope_inv_freq.resize(c.rope_dim / 2);
  for (int i = 0; i < c.rope_dim / 2; ++i) {
    w.rope_inv_freq[i] = static_cast<float>(
        1.0 / std::pow(static_cast<double>(c.rope_theta),
                       2.0 * i / static_cast<double>(c.rope_dim)));
  }
}

Reference::~Reference() = default;

const Projections& Reference::projections() const {
  return *gguf_projections_;
}

State Reference::NewState() const {
  const Config& c = config_;
  State state;
  state.linear.resize(c.layers + c.mtp_layers);
  state.attention.resize(c.layers + c.mtp_layers);
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
                                 bool want_logits, bool want_hidden,
                                 const Projections* projections) const {
  const Config& c = config_;
  const auto& w = *weights_;
  const Projections& proj =
      projections != nullptr ? *projections : *gguf_projections_;
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

  for (int il = 0; il < c.layers; ++il) {
    RunLayer(il, h, n, state, proj);
    if (want_hidden) {
      result.hidden.insert(result.hidden.end(), h.begin(), h.end());
    }
  }
  state.position += n;

  if (want_logits) {
    result.logits = Head(h, w.output_norm, proj);
  }
  return result;
}

std::vector<float> Reference::Mtp(const std::vector<float>& h,
                                  const std::vector<int>& tokens, State& state,
                                  const Projections* projections) const {
  const Config& c = config_;
  const auto& w = *weights_;
  const Projections& proj =
      projections != nullptr ? *projections : *gguf_projections_;
  const int n = static_cast<int>(tokens.size());
  const int d = c.hidden;
  if (w.mtp_enorm.empty()) {
    throw std::runtime_error("qwen35: this GGUF has no MTP layer");
  }
  if (h.size() != static_cast<std::size_t>(n) * d) {
    throw std::invalid_argument("qwen35: MTP hidden rows do not match tokens");
  }
  // [RMS(embed(token)) * enorm ; RMS(RMS(h) * output_norm) * hnorm]
  std::vector<float> eh(static_cast<std::size_t>(n) * 2 * d);
  std::vector<float> row(d);
  const auto embed_row = quant::RowBytes(w.embed->type, d);
  for (int t = 0; t < n; ++t) {
    if (tokens[t] < 0 || tokens[t] >= c.vocab) {
      throw std::out_of_range("qwen35: token id out of range");
    }
    float* out = &eh[static_cast<std::size_t>(t) * 2 * d];
    quant::Dequantize(
        w.embed->type,
        w.embed->data + static_cast<std::size_t>(tokens[t]) * embed_row,
        row.data(), d);
    RmsNorm(row.data(), w.mtp_enorm.data(), d, c.rms_eps, out);
    RmsNorm(&h[static_cast<std::size_t>(t) * d], w.output_norm.data(), d,
            c.rms_eps, row.data());
    RmsNorm(row.data(), w.mtp_hnorm.data(), d, c.rms_eps, out + d);
  }
  std::vector<float> x(static_cast<std::size_t>(n) * d);
  proj.Apply(Stage::kMtpIn, c.layers, eh.data(), n, x.data());
  RunLayer(c.layers, x, n, state, proj);
  state.position += n;
  return Head(x, w.mtp_head_norm, proj);
}

void Reference::ForwardLayer(int layer, std::vector<float>& h, State& state,
                             const Projections* projections) const {
  if (layer < 0 || layer >= config_.layers ||
      h.size() % static_cast<std::size_t>(config_.hidden) != 0) {
    throw std::invalid_argument("qwen35: bad layer or residual size");
  }
  RunLayer(layer, h, static_cast<int>(h.size() / config_.hidden), state,
           projections != nullptr ? *projections : *gguf_projections_);
}

std::vector<float> Reference::Logits(const std::vector<float>& h,
                                     const Projections* projections) const {
  return Head(h, weights_->output_norm,
              projections != nullptr ? *projections : *gguf_projections_);
}

std::vector<float> Reference::Head(const std::vector<float>& h,
                                   const std::vector<float>& norm,
                                   const Projections& proj) const {
  const Config& c = config_;
  const int d = c.hidden;
  const int n = static_cast<int>(h.size() / d);
  std::vector<float> normed(h.size());
  for (int t = 0; t < n; ++t) {
    RmsNorm(&h[static_cast<std::size_t>(t) * d], norm.data(), d, c.rms_eps,
            &normed[static_cast<std::size_t>(t) * d]);
  }
  std::vector<float> logits(static_cast<std::size_t>(n) * c.vocab);
  proj.Apply(Stage::kLmHead, -1, normed.data(), n, logits.data());
  return logits;
}

void Reference::RunLayer(int il, std::vector<float>& h, int n, State& state,
                         const Projections& proj) const {
  const Config& c = config_;
  const auto& w = *weights_;
  const auto& layer = w.layers[il];
  const int d = c.hidden;
  const auto nd = static_cast<std::size_t>(n) * d;

  std::vector<float> normed(nd);
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
    {
      std::vector<float> fused(static_cast<std::size_t>(n) *
                               StageOutputs(c, Stage::kAttentionIn, il));
      proj.Apply(Stage::kAttentionIn, il, normed.data(), n, fused.data());
      Split(fused, n, {{&qg, 2 * c.heads * hd}, {&k, kv}, {&v, kv}});
    }

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
      // Every cached position (pos + 1 for the decoder layers; the MTP
      // layer's cache may start later).
      const int length = static_cast<int>(cache.keys.size() / kv);
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
            acc += scores[j] *
                   cache.values[static_cast<std::size_t>(j) * kv + kh * hd + e];
          }
          out[e] = static_cast<float>(acc / total) * Sigmoid(gate[e]);
        }
      }
      proj.Apply(Stage::kMixerOut, il, ctx.data(), 1,
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
    {
      std::vector<float> fused(static_cast<std::size_t>(n) *
                               StageOutputs(c, Stage::kLinearIn, il));
      proj.Apply(Stage::kLinearIn, il, normed.data(), n, fused.data());
      Split(fused, n,
            {{&qkv, channels}, {&z, value_dim}, {&alpha, hv}, {&beta, hv}});
    }

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
                    channels, &st.conv[static_cast<std::size_t>(s) * channels]);
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
        const float b = Sigmoid(beta[static_cast<std::size_t>(t) * hv + head]);
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
      proj.Apply(Stage::kMixerOut, il, y.data(), 1,
                 &mixed[static_cast<std::size_t>(t) * d]);
    }
  }

  for (std::size_t i = 0; i < nd; ++i) {
    h[i] += mixed[i];
  }

  for (int t = 0; t < n; ++t) {
    RmsNorm(&h[static_cast<std::size_t>(t) * d],
            layer.post_attention_norm.data(), d, c.rms_eps,
            &normed[static_cast<std::size_t>(t) * d]);
  }
  Ffn(il, normed.data(), n, proj, mixed.data());
  for (std::size_t i = 0; i < nd; ++i) {
    h[i] += mixed[i];
  }
}

void Reference::Ffn(int il, const float* x, int n, const Projections& proj,
                    float* out) const {
  const Config& c = config_;
  const int d = c.hidden;
  const auto nd = static_cast<std::size_t>(n) * d;

  // A SwiGLU block: the dense FFN, the shared expert, or a routed expert.
  auto swiglu = [](const std::vector<float>& gate_up, int rows, int width) {
    std::vector<float> act(static_cast<std::size_t>(rows) * width);
    for (int r = 0; r < rows; ++r) {
      const float* gate = &gate_up[static_cast<std::size_t>(r) * 2 * width];
      const float* up = gate + width;
      for (int i = 0; i < width; ++i) {
        act[static_cast<std::size_t>(r) * width + i] = Silu(gate[i]) * up[i];
      }
    }
    return act;
  };

  // The dense FFN, or the shared expert.
  const int f = c.FfnWidth();
  std::vector<float> gate_up(static_cast<std::size_t>(n) * 2 * f);
  proj.Apply(Stage::kFfnIn, il, x, n, gate_up.data());
  const auto act = swiglu(gate_up, n, f);
  if (!c.IsMoe()) {
    proj.Apply(Stage::kFfnDown, il, act.data(), n, out);
    return;
  }
  std::vector<float> shared(nd);
  proj.Apply(Stage::kFfnDown, il, act.data(), n, shared.data());

  // Routing (moe.hpp); the shared expert is scaled by sigmoid of its own
  // gate logit, the router's last output.
  const int experts = c.experts;
  const int k = c.experts_used;
  std::vector<float> router(static_cast<std::size_t>(n) * (experts + 1));
  proj.Apply(Stage::kRouter, il, x, n, router.data());
  std::vector<int> chosen(static_cast<std::size_t>(n) * k);
  std::vector<float> weight(static_cast<std::size_t>(n) * k);
  std::vector<float> shared_scale(n);
  for (int t = 0; t < n; ++t) {
    const float* logits = &router[static_cast<std::size_t>(t) * (experts + 1)];
    Route(logits, experts, k, &chosen[static_cast<std::size_t>(t) * k],
          &weight[static_cast<std::size_t>(t) * k]);
    shared_scale[t] = SharedExpertScale(logits[experts]);
  }

  // Each routed expert once over the tokens that chose it.
  std::vector<std::vector<std::size_t>> assigned(experts);  // t * k + slot
  for (std::size_t i = 0; i < chosen.size(); ++i) {
    assigned[chosen[i]].push_back(i);
  }
  const int ef = c.expert_ffn;
  std::vector<float> routed(static_cast<std::size_t>(n) * k * d);
  std::vector<float> xs;
  std::vector<float> ys;
  for (int e = 0; e < experts; ++e) {
    const auto& slots = assigned[e];
    if (slots.empty()) {
      continue;
    }
    const int m = static_cast<int>(slots.size());
    xs.resize(static_cast<std::size_t>(m) * d);
    for (int r = 0; r < m; ++r) {
      std::copy_n(x + slots[r] / k * d, d,
                  &xs[static_cast<std::size_t>(r) * d]);
    }
    std::vector<float> expert_gate_up(static_cast<std::size_t>(m) * 2 * ef);
    proj.ApplyExpert(Stage::kExpertIn, il, e, xs.data(), m,
                     expert_gate_up.data());
    const auto expert_act = swiglu(expert_gate_up, m, ef);
    ys.resize(static_cast<std::size_t>(m) * d);
    proj.ApplyExpert(Stage::kExpertDown, il, e, expert_act.data(), m,
                     ys.data());
    for (int r = 0; r < m; ++r) {
      std::copy_n(&ys[static_cast<std::size_t>(r) * d], d,
                  &routed[slots[r] * d]);
    }
  }

  // Weighted sum in slot order, then the gated shared expert.
  for (int t = 0; t < n; ++t) {
    for (int i = 0; i < d; ++i) {
      double acc = 0;
      for (int slot = 0; slot < k; ++slot) {
        const std::size_t s = static_cast<std::size_t>(t) * k + slot;
        acc += static_cast<double>(weight[s]) * routed[s * d + i];
      }
      const std::size_t ti = static_cast<std::size_t>(t) * d + i;
      acc += static_cast<double>(shared_scale[t]) * shared[ti];
      out[ti] = static_cast<float>(acc);
    }
  }
}

std::size_t StageInputs(const Config& c, Stage stage, int layer) {
  switch (stage) {
    case Stage::kMixerOut:
      return static_cast<std::size_t>(
          c.IsFullAttention(layer) ? c.heads * c.head_dim : c.LinearValueDim());
    case Stage::kFfnDown:
      return static_cast<std::size_t>(c.FfnWidth());
    case Stage::kExpertDown:
      return static_cast<std::size_t>(c.expert_ffn);
    case Stage::kMtpIn:
      return static_cast<std::size_t>(2 * c.hidden);
    default:
      return static_cast<std::size_t>(c.hidden);
  }
}

std::size_t StageOutputs(const Config& c, Stage stage, int /*layer*/) {
  switch (stage) {
    case Stage::kLinearIn:
      return static_cast<std::size_t>(c.ConvChannels() + c.LinearValueDim() +
                                      2 * c.linear_value_heads);
    case Stage::kAttentionIn:
      return static_cast<std::size_t>(2 * c.heads * c.head_dim +
                                      2 * c.kv_heads * c.head_dim);
    case Stage::kMixerOut:
    case Stage::kFfnDown:
    case Stage::kExpertDown:
    case Stage::kMtpIn:
      return static_cast<std::size_t>(c.hidden);
    case Stage::kFfnIn:
      return static_cast<std::size_t>(2 * c.FfnWidth());
    case Stage::kLmHead:
      return static_cast<std::size_t>(c.vocab);
    case Stage::kRouter:
      return static_cast<std::size_t>(c.experts + 1);
    case Stage::kExpertIn:
      return static_cast<std::size_t>(2 * c.expert_ffn);
  }
  return 0;
}

}  // namespace gufo_npu::qwen35
