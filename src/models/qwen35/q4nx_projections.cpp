#include "models/qwen35/q4nx_projections.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/parallel.hpp"
#include "core/quant.hpp"

namespace gufo_npu::qwen35 {

struct Q4nxProjections::Layer {
  q4nx::Matrix in;      // qkv + z, or q + k + v
  q4nx::Matrix out;     // ssm_out or attn_output
  q4nx::Matrix ffn_in;  // gate + up: the dense FFN or the shared expert
  q4nx::Matrix down;
  std::vector<float> alpha;  // [value heads][hidden]
  std::vector<float> beta;
  // MoE: router rows, then the shared expert's gate row (fp32); the routed
  // experts' stacked tensors, and per expert the packed matrices of those
  // that are not Q4_K (empty when they are).
  std::vector<float> router;
  const gguf::TensorInfo* gate_exps = nullptr;
  const gguf::TensorInfo* up_exps = nullptr;
  const gguf::TensorInfo* down_exps = nullptr;
  std::vector<q4nx::Matrix> expert_in;
  std::vector<q4nx::Matrix> expert_down;
};

namespace {

constexpr std::size_t kQ4KBytes = 144;  // block_q4_K

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

q4nx::Segment SegmentOf(const gguf::TensorInfo& t, std::size_t first = 0,
                        std::size_t rows = 0) {
  return {t.type, t.data + first * quant::RowBytes(t.type, t.RowElements()),
          rows != 0 ? rows : static_cast<std::size_t>(t.Rows())};
}

q4nx::Matrix PackOne(const gguf::TensorInfo& t) {
  return q4nx::Pack(t.type, t.data, t.Rows(), t.RowElements());
}

// A Q4_K row as q4nx holds it: per 256 values d and dmin rounded to bf16,
// then min + step * code with step = d * sc and min = dmin * -m (q4nx.cpp's
// PackQ4KSuper, then Unpack).
void UnpackQ4KAsQ4nx(const std::uint8_t* src, float* dst, std::size_t n) {
  for (std::size_t b = 0; b < n / 256; ++b, src += kQ4KBytes, dst += 256) {
    std::uint16_t bits;
    std::memcpy(&bits, src, 2);
    const float d =
        quant::Bf16ToFloat(quant::FloatToBf16(quant::HalfToFloat(bits)));
    std::memcpy(&bits, src + 2, 2);
    const float dmin =
        quant::Bf16ToFloat(quant::FloatToBf16(quant::HalfToFloat(bits)));
    for (std::size_t g = 0; g < 8; ++g) {
      std::uint8_t sc;
      std::uint8_t m;
      quant::ScaleMinK4(g, src + 4, &sc, &m);
      const float step = d * static_cast<float>(sc);
      const float min = dmin * static_cast<float>(static_cast<std::int8_t>(-m));
      const std::uint8_t* q = src + 16 + (g / 2) * 32;
      const int shift = static_cast<int>(g % 2) * 4;
      for (std::size_t l = 0; l < 32; ++l) {
        dst[g * 32 + l] =
            min + step * static_cast<float>((q[l] >> shift) & 0xF);
      }
    }
  }
}

// y[t] = W x[t] in double for `rows` rows of Q4_K tensor `w` from row
// `first`, unpacked by q4nx's rule; y rows are `stride` apart per token.
void GemvQ4K(const gguf::TensorInfo& w, std::size_t first, std::size_t rows,
             const float* x, int tokens, float* y, std::size_t stride) {
  const auto in = static_cast<std::size_t>(w.RowElements());
  const std::size_t row_bytes = quant::RowBytes(w.type, in);
  ParallelFor(rows, [&](std::size_t begin, std::size_t end) {
    std::vector<float> row(in);
    for (std::size_t r = begin; r < end; ++r) {
      UnpackQ4KAsQ4nx(w.data + (first + r) * row_bytes, row.data(), in);
      for (int t = 0; t < tokens; ++t) {
        const float* xt = x + static_cast<std::size_t>(t) * in;
        double acc = 0;
        for (std::size_t k = 0; k < in; ++k) {
          acc += static_cast<double>(row[k]) * xt[k];
        }
        y[static_cast<std::size_t>(t) * stride + r] = static_cast<float>(acc);
      }
    }
  });
}

// y[t][r] = rows[r] . x[t] in double for fp32 rows of `in` values.
void GemvF32(const std::vector<float>& rows, std::size_t in, const float* x,
             int tokens, float* y, std::size_t stride) {
  const std::size_t n = rows.size() / in;
  for (int t = 0; t < tokens; ++t) {
    const float* xt = x + static_cast<std::size_t>(t) * in;
    for (std::size_t r = 0; r < n; ++r) {
      const float* w = rows.data() + r * in;
      double acc = 0;
      for (std::size_t k = 0; k < in; ++k) {
        acc += static_cast<double>(w[k]) * xt[k];
      }
      y[static_cast<std::size_t>(t) * stride + r] = static_cast<float>(acc);
    }
  }
}

}  // namespace

Q4nxProjections::Q4nxProjections(const gguf::File& file)
    : config_(ReadConfig(file)) {
  const Config& c = config_;
  const auto hidden = static_cast<std::size_t>(c.hidden);
  for (int il = 0; il < c.layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto t = [&](const char* name) -> const gguf::TensorInfo& {
      return file.Tensor(p + name);
    };
    auto layer = std::make_unique<Layer>();
    if (c.IsFullAttention(il)) {
      layer->in = q4nx::PackRows(
          {SegmentOf(t("attn_q.weight")), SegmentOf(t("attn_k.weight")),
           SegmentOf(t("attn_v.weight"))},
          hidden);
      layer->out = PackOne(t("attn_output.weight"));
    } else {
      layer->in = q4nx::PackRows(
          {SegmentOf(t("attn_qkv.weight")), SegmentOf(t("attn_gate.weight"))},
          hidden);
      layer->alpha = ToFloats(t("ssm_alpha.weight"));
      layer->beta = ToFloats(t("ssm_beta.weight"));
      layer->out = PackOne(t("ssm_out.weight"));
    }
    if (c.IsMoe()) {
      layer->ffn_in = q4nx::PackRows({SegmentOf(t("ffn_gate_shexp.weight")),
                                      SegmentOf(t("ffn_up_shexp.weight"))},
                                     hidden);
      layer->down = PackOne(t("ffn_down_shexp.weight"));
      layer->router = ToFloats(t("ffn_gate_inp.weight"));
      const auto shared_gate = ToFloats(t("ffn_gate_inp_shexp.weight"));
      layer->router.insert(layer->router.end(), shared_gate.begin(),
                           shared_gate.end());
      layer->gate_exps = &t("ffn_gate_exps.weight");
      layer->up_exps = &t("ffn_up_exps.weight");
      layer->down_exps = &t("ffn_down_exps.weight");
      const auto ef = static_cast<std::size_t>(c.expert_ffn);
      for (int e = 0; e < c.experts; ++e) {
        const auto first = static_cast<std::size_t>(e);
        if (layer->gate_exps->type != gguf::GgmlType::kQ4_K ||
            layer->up_exps->type != gguf::GgmlType::kQ4_K) {
          layer->expert_in.push_back(
              q4nx::PackRows({SegmentOf(*layer->gate_exps, first * ef, ef),
                              SegmentOf(*layer->up_exps, first * ef, ef)},
                             hidden));
        }
        if (layer->down_exps->type != gguf::GgmlType::kQ4_K) {
          layer->expert_down.push_back(q4nx::PackRows(
              {SegmentOf(*layer->down_exps, first * hidden, hidden)}, ef));
        }
      }
    } else {
      layer->ffn_in = q4nx::PackRows(
          {SegmentOf(t("ffn_gate.weight")), SegmentOf(t("ffn_up.weight"))},
          hidden);
      layer->down = PackOne(t("ffn_down.weight"));
    }
    layers_.push_back(std::move(layer));
  }
  const auto& head = c.tied_embeddings ? file.Tensor("token_embd.weight")
                                       : file.Tensor("output.weight");
  lm_head_ = q4nx::Pack(head.type, head.data, head.Rows(), hidden);
}

Q4nxProjections::~Q4nxProjections() = default;

void Q4nxProjections::Apply(Stage stage, int layer, const float* x, int tokens,
                            float* y) const {
  const std::size_t in = StageInputs(config_, stage, layer);
  const std::size_t out = StageOutputs(config_, stage, layer);
  if (stage == Stage::kRouter) {
    const Layer& l = *layers_.at(static_cast<std::size_t>(layer));
    if (l.router.empty()) {
      throw std::logic_error("q4nx projections: not an MoE layer");
    }
    GemvF32(l.router, in, x, tokens, y, out);
    return;
  }
  for (int t = 0; t < tokens; ++t) {
    const float* xt = x + static_cast<std::size_t>(t) * in;
    float* yt = y + static_cast<std::size_t>(t) * out;
    if (stage == Stage::kLmHead) {
      q4nx::Gemv(lm_head_, xt, yt);
      continue;
    }
    const Layer& l = *layers_.at(static_cast<std::size_t>(layer));
    switch (stage) {
      case Stage::kLinearIn: {
        q4nx::Gemv(l.in, xt, yt);
        // alpha and beta (F32 rows, one per value head) after qkv and z.
        const std::size_t heads =
            static_cast<std::size_t>(config_.linear_value_heads);
        float* ab = yt + l.in.rows;
        GemvF32(l.alpha, in, xt, 1, ab, heads);
        GemvF32(l.beta, in, xt, 1, ab + heads, heads);
        break;
      }
      case Stage::kAttentionIn:
        q4nx::Gemv(l.in, xt, yt);
        break;
      case Stage::kMixerOut:
        q4nx::Gemv(l.out, xt, yt);
        break;
      case Stage::kFfnIn:
        q4nx::Gemv(l.ffn_in, xt, yt);
        break;
      case Stage::kFfnDown:
        q4nx::Gemv(l.down, xt, yt);
        break;
      case Stage::kLmHead:
      case Stage::kRouter:
        break;
      case Stage::kExpertIn:
      case Stage::kExpertDown:
      case Stage::kMtpIn:
        throw std::logic_error(
            "q4nx projections: decoder layers, router and LM head only");
    }
  }
}

void Q4nxProjections::ApplyExpert(Stage stage, int layer, int expert,
                                  const float* x, int tokens, float* y) const {
  const Layer& l = *layers_.at(static_cast<std::size_t>(layer));
  if (l.gate_exps == nullptr || expert < 0 || expert >= config_.experts) {
    throw std::out_of_range("q4nx projections: no such expert");
  }
  const auto e = static_cast<std::size_t>(expert);
  const auto ef = static_cast<std::size_t>(config_.expert_ffn);
  const auto hidden = static_cast<std::size_t>(config_.hidden);
  const std::size_t in = stage == Stage::kExpertIn ? hidden : ef;
  const std::size_t out = stage == Stage::kExpertIn ? 2 * ef : hidden;
  const std::vector<q4nx::Matrix>& packed =
      stage == Stage::kExpertIn ? l.expert_in : l.expert_down;
  if (stage != Stage::kExpertIn && stage != Stage::kExpertDown) {
    throw std::logic_error("q4nx projections: not an expert stage");
  }
  if (!packed.empty()) {
    for (int t = 0; t < tokens; ++t) {
      q4nx::Gemv(packed[e], x + static_cast<std::size_t>(t) * in,
                 y + static_cast<std::size_t>(t) * out);
    }
    return;
  }
  if (stage == Stage::kExpertIn) {
    GemvQ4K(*l.gate_exps, e * ef, ef, x, tokens, y, out);
    GemvQ4K(*l.up_exps, e * ef, ef, x, tokens, y + ef, out);
  } else {
    GemvQ4K(*l.down_exps, e * hidden, hidden, x, tokens, y, out);
  }
}

}  // namespace gufo_npu::qwen35
