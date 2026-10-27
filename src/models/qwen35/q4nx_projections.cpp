#include "models/qwen35/q4nx_projections.hpp"

#include <string>

#include "core/quant.hpp"

namespace gufo_npu::qwen35 {

struct Q4nxProjections::Layer {
  q4nx::Matrix in;  // qkv + z, or q + k + v
  q4nx::Matrix out;
  q4nx::Matrix ffn_in;
  q4nx::Matrix down;
  std::vector<float> alpha;  // [value heads][hidden]
  std::vector<float> beta;
};

namespace {

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

q4nx::Segment SegmentOf(const gguf::TensorInfo& t) {
  return {t.type, t.data, static_cast<std::size_t>(t.Rows())};
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
      const auto& o = t("attn_output.weight");
      layer->out = q4nx::Pack(o.type, o.data, o.Rows(), o.RowElements());
    } else {
      layer->in = q4nx::PackRows(
          {SegmentOf(t("attn_qkv.weight")), SegmentOf(t("attn_gate.weight"))},
          hidden);
      layer->alpha = ToFloats(t("ssm_alpha.weight"));
      layer->beta = ToFloats(t("ssm_beta.weight"));
      const auto& o = t("ssm_out.weight");
      layer->out = q4nx::Pack(o.type, o.data, o.Rows(), o.RowElements());
    }
    layer->ffn_in = q4nx::PackRows(
        {SegmentOf(t("ffn_gate.weight")), SegmentOf(t("ffn_up.weight"))},
        hidden);
    const auto& down = t("ffn_down.weight");
    layer->down =
        q4nx::Pack(down.type, down.data, down.Rows(), down.RowElements());
    layers_.push_back(std::move(layer));
  }
  const auto& embed = file.Tensor("token_embd.weight");
  lm_head_ = q4nx::Pack(embed.type, embed.data, embed.Rows(), hidden);
}

Q4nxProjections::~Q4nxProjections() = default;

void Q4nxProjections::Apply(Stage stage, int layer, const float* x, int tokens,
                            float* y) const {
  const std::size_t in = StageInputs(config_, stage, layer);
  const std::size_t out = StageOutputs(config_, stage, layer);
  for (int t = 0; t < tokens; ++t) {
    const float* xt = x + static_cast<std::size_t>(t) * in;
    float* yt = y + static_cast<std::size_t>(t) * out;
    if (stage == Stage::kLmHead) {
      q4nx::Gemv(lm_head_, xt, yt);
      continue;
    }
    const Layer& l = *layers_[static_cast<std::size_t>(layer)];
    switch (stage) {
      case Stage::kLinearIn: {
        q4nx::Gemv(l.in, xt, yt);
        // alpha and beta: 16 F32 rows each, after qkv and z.
        const std::size_t heads =
            static_cast<std::size_t>(config_.linear_value_heads);
        float* ab = yt + l.in.rows;
        for (std::size_t h = 0; h < 2 * heads; ++h) {
          const float* w =
              (h < heads ? l.alpha.data() : l.beta.data()) + (h % heads) * in;
          double acc = 0;
          for (std::size_t k = 0; k < in; ++k) {
            acc += static_cast<double>(w[k]) * xt[k];
          }
          ab[h] = static_cast<float>(acc);
        }
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
        break;
    }
  }
}

}  // namespace gufo_npu::qwen35
