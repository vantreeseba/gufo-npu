#include "models/qwen35/reference.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

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

// The smallest positive normal fp16, llama.cpp's floor for the sum of the
// top-k router probabilities before renormalizing.
constexpr double kRouterWeightSumFloor = 6.103515625e-5;

std::vector<float> ToFloats(const TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

// y[t][r] = sum_c W[first + r][c] * x[t][c] for `rows` rows of W from row
// `first` and `tokens` rows of x. Each weight row is dequantized once and
// applied to every token.
void MatMulRows(const TensorInfo& w, std::size_t first, std::size_t rows,
                const float* x, int tokens, float* y) {
  const auto in = static_cast<std::size_t>(w.RowElements());
  const auto row_bytes = quant::RowBytes(w.type, in);
  ParallelFor(rows, [&](std::size_t begin, std::size_t end) {
    std::vector<float> row(in);
    for (std::size_t r = begin; r < end; ++r) {
      quant::Dequantize(w.type, w.data + (first + r) * row_bytes, row.data(),
                        in);
      for (int t = 0; t < tokens; ++t) {
        const float* xt = x + static_cast<std::size_t>(t) * in;
        double sum = 0;
        for (std::size_t c = 0; c < in; ++c) {
          sum += static_cast<double>(row[c]) * xt[c];
        }
        y[static_cast<std::size_t>(t) * rows + r] = static_cast<float>(sum);
      }
    }
  });
}

void MatMul(const TensorInfo& w, const float* x, int tokens, float* y) {
  MatMulRows(w, 0, w.Rows(), x, tokens, y);
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

// Rows [first, first + rows) of a weight matrix.
struct RowSpan {
  const TensorInfo* w;
  std::size_t first;
  std::size_t rows;
};

RowSpan All(const TensorInfo* w) {
  return {w, 0, static_cast<std::size_t>(w->Rows())};
}

// Writes `tokens` rows of W x for each of `spans` side by side, in order.
void MatMulFused(std::initializer_list<RowSpan> spans, const float* x,
                 int tokens, float* y) {
  std::size_t stride = 0;
  for (const auto& span : spans) {
    stride += span.rows;
  }
  std::size_t offset = 0;
  std::vector<float> part;
  for (const auto& span : spans) {
    part.resize(static_cast<std::size_t>(tokens) * span.rows);
    MatMulRows(*span.w, span.first, span.rows, x, tokens, part.data());
    for (int t = 0; t < tokens; ++t) {
      std::copy_n(&part[static_cast<std::size_t>(t) * span.rows], span.rows,
                  y + static_cast<std::size_t>(t) * stride + offset);
    }
    offset += span.rows;
  }
}

// `t` must be [cols, rows], [cols] when rows is 1, or, with `experts`,
// [cols, rows, experts].
void CheckShape(const TensorInfo& t, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts = 0) {
  std::vector<std::uint64_t> want = {cols, rows};
  if (experts != 0) {
    want.push_back(experts);
  } else if (rows == 1 && t.shape.size() == 1) {
    want.pop_back();
  }
  if (t.shape != want) {
    auto text = [](const std::vector<std::uint64_t>& shape) {
      std::string out;
      for (const auto dim : shape) {
        out += (out.empty() ? "" : ", ") + std::to_string(dim);
      }
      return "[" + out + "]";
    };
    throw std::runtime_error("qwen35: tensor " + t.name + " has shape " +
                             text(t.shape) + ", expected " + text(want));
  }
}

}  // namespace

struct Reference::Weights {
  struct Layer {
    std::vector<float> attn_norm;
    std::vector<float> post_attention_norm;
    // The dense FFN, or the MoE shared expert.
    const TensorInfo* ffn_gate = nullptr;
    const TensorInfo* ffn_up = nullptr;
    const TensorInfo* ffn_down = nullptr;
    // MoE: router [hidden, experts], shared-expert gate [hidden], and the
    // routed experts stacked as [cols, rows, experts].
    const TensorInfo* router = nullptr;
    const TensorInfo* shared_gate = nullptr;
    const TensorInfo* gate_exps = nullptr;
    const TensorInfo* up_exps = nullptr;
    const TensorInfo* down_exps = nullptr;
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
  const TensorInfo* lm_head = nullptr;  // `embed` when tied
  std::vector<float> output_norm;
  std::vector<Layer> layers;
  std::vector<float> rope_inv_freq;  // rope_dim / 2 entries
};

// The reference's own projections: the GGUF weights, dequantized per row.
class Reference::GgufProjections : public Projections {
public:
  GgufProjections(const Config& c, const Weights& w) : c_(c), w_(w) {}

  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override {
    if (stage == Stage::kLmHead) {
      return MatMul(*w_.lm_head, x, tokens, y);
    }
    const auto& l = w_.layers[static_cast<std::size_t>(layer)];
    switch (stage) {
      case Stage::kLinearIn:
        return MatMulFused({All(l.qkv), All(l.gate), All(l.alpha), All(l.beta)},
                           x, tokens, y);
      case Stage::kAttentionIn:
        return MatMulFused({All(l.q), All(l.k), All(l.v)}, x, tokens, y);
      case Stage::kMixerOut:
        return MatMul(l.o != nullptr ? *l.o : *l.ssm_out, x, tokens, y);
      case Stage::kFfnIn:
        return MatMulFused({All(l.ffn_gate), All(l.ffn_up)}, x, tokens, y);
      case Stage::kFfnDown:
        return MatMul(*l.ffn_down, x, tokens, y);
      case Stage::kRouter:
        return MatMulFused({All(l.router), All(l.shared_gate)}, x, tokens, y);
      case Stage::kLmHead:
      case Stage::kExpertIn:
      case Stage::kExpertDown:
        break;
    }
    throw std::logic_error("qwen35: expert stages go through ApplyExpert");
  }

  void ApplyExpert(Stage stage, int layer, int expert, const float* x,
                   int tokens, float* y) const override {
    const auto& l = w_.layers[static_cast<std::size_t>(layer)];
    const auto e = static_cast<std::size_t>(expert);
    const auto ffn = static_cast<std::size_t>(c_.expert_ffn);
    const auto hidden = static_cast<std::size_t>(c_.hidden);
    switch (stage) {
      case Stage::kExpertIn:
        return MatMulFused(
            {{l.gate_exps, e * ffn, ffn}, {l.up_exps, e * ffn, ffn}}, x, tokens,
            y);
      case Stage::kExpertDown:
        return MatMulRows(*l.down_exps, e * hidden, hidden, x, tokens, y);
      default:
        throw std::logic_error("qwen35: ApplyExpert takes expert stages only");
    }
  }

private:
  const Config& c_;
  const Weights& w_;
};

Reference::Reference(const gguf::File& file)
    : config_(ReadConfig(file)), weights_(std::make_unique<Weights>()) {
  LoadWeights(file);
  gguf_projections_ = std::make_unique<GgufProjections>(config_, *weights_);
}

void Reference::LoadWeights(const gguf::File& file) {
  const Config& c = config_;
  auto& w = *weights_;
  w.embed = &file.Tensor("token_embd.weight");
  CheckShape(*w.embed, c.hidden, c.vocab);
  w.lm_head = c.tied_embeddings ? w.embed : &file.Tensor("output.weight");
  CheckShape(*w.lm_head, c.hidden, c.vocab);
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
    if (c.IsMoe()) {
      layer.ffn_gate = &tensor("ffn_gate_shexp.weight");
      layer.ffn_up = &tensor("ffn_up_shexp.weight");
      layer.ffn_down = &tensor("ffn_down_shexp.weight");
      layer.router = &tensor("ffn_gate_inp.weight");
      layer.shared_gate = &tensor("ffn_gate_inp_shexp.weight");
      layer.gate_exps = &tensor("ffn_gate_exps.weight");
      layer.up_exps = &tensor("ffn_up_exps.weight");
      layer.down_exps = &tensor("ffn_down_exps.weight");
      CheckShape(*layer.router, c.hidden, c.experts);
      CheckShape(*layer.shared_gate, c.hidden, 1);
      CheckShape(*layer.gate_exps, c.hidden, c.expert_ffn, c.experts);
      CheckShape(*layer.up_exps, c.hidden, c.expert_ffn, c.experts);
      CheckShape(*layer.down_exps, c.expert_ffn, c.hidden, c.experts);
    } else {
      layer.ffn_gate = &tensor("ffn_gate.weight");
      layer.ffn_up = &tensor("ffn_up.weight");
      layer.ffn_down = &tensor("ffn_down.weight");
    }
    CheckShape(*layer.ffn_gate, c.hidden, c.FfnWidth());
    CheckShape(*layer.ffn_up, c.hidden, c.FfnWidth());
    CheckShape(*layer.ffn_down, c.FfnWidth(), c.hidden);
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

const Projections& Reference::projections() const {
  return *gguf_projections_;
}

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
    result.logits = Logits(h, &proj);
  }
  return result;
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
  const Config& c = config_;
  const int d = c.hidden;
  const int n = static_cast<int>(h.size() / d);
  std::vector<float> normed(h.size());
  for (int t = 0; t < n; ++t) {
    RmsNorm(&h[static_cast<std::size_t>(t) * d], weights_->output_norm.data(),
            d, c.rms_eps, &normed[static_cast<std::size_t>(t) * d]);
  }
  std::vector<float> logits(static_cast<std::size_t>(n) * c.vocab);
  (projections != nullptr ? *projections : *gguf_projections_)
      .Apply(Stage::kLmHead, -1, normed.data(), n, logits.data());
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

  // Routing: softmax over all expert logits, the top k (ties to the lower
  // index), their probabilities renormalized to sum 1. The shared expert is
  // scaled by sigmoid of its own gate logit.
  const int experts = c.experts;
  const int k = c.experts_used;
  std::vector<float> router(static_cast<std::size_t>(n) * (experts + 1));
  proj.Apply(Stage::kRouter, il, x, n, router.data());
  std::vector<int> chosen(static_cast<std::size_t>(n) * k);
  std::vector<double> weight(static_cast<std::size_t>(n) * k);
  std::vector<double> shared_scale(n);
  std::vector<double> prob(experts);
  std::vector<int> order(experts);
  for (int t = 0; t < n; ++t) {
    const float* logits = &router[static_cast<std::size_t>(t) * (experts + 1)];
    const double max_logit = *std::max_element(logits, logits + experts);
    double total = 0;
    for (int e = 0; e < experts; ++e) {
      prob[e] = std::exp(static_cast<double>(logits[e]) - max_logit);
      total += prob[e];
    }
    for (auto& p : prob) {
      p /= total;
    }
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + k, order.end(), [&](int a, int b) {
          return prob[a] > prob[b] || (prob[a] == prob[b] && a < b);
        });
    double top = 0;
    for (int slot = 0; slot < k; ++slot) {
      top += prob[order[slot]];
    }
    top = std::max(top, kRouterWeightSumFloor);
    for (int slot = 0; slot < k; ++slot) {
      chosen[static_cast<std::size_t>(t) * k + slot] = order[slot];
      weight[static_cast<std::size_t>(t) * k + slot] = prob[order[slot]] / top;
    }
    shared_scale[t] =
        1.0 / (1.0 + std::exp(-static_cast<double>(logits[experts])));
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
        acc += weight[s] * routed[s * d + i];
      }
      const std::size_t ti = static_cast<std::size_t>(t) * d + i;
      acc += shared_scale[t] * shared[ti];
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
