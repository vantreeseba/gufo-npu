#include "models/qwen35/q4nx_projections.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

#include "core/quant.hpp"
#include "npu/gemv.hpp"

namespace gufo_npu::qwen35 {

struct Q4nxProjections::Gemv {
  q4nx::Matrix packed;  // emptied after an NPU upload
  std::unique_ptr<npu::Q4nxGemv> npu;
  std::size_t rows = 0;
};

struct Q4nxProjections::Layer {
  Gemv in;  // qkv + z, or q + k + v
  Gemv out;
  Gemv ffn_in;
  Gemv down;
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

Q4nxProjections::Q4nxProjections(const gguf::File& file, Backend backend,
                                 const std::filesystem::path& kernel_dir)
    : config_(ReadConfig(file)), backend_(backend) {
  if (backend_ == Backend::kNpu) {
    runtime_ = std::make_unique<npu::GemvRuntime>(kernel_dir);
  }
  auto make = [&](std::vector<q4nx::Segment> segments, std::size_t dim) {
    Gemv g;
    g.packed = q4nx::PackRows(segments, dim);
    g.rows = g.packed.rows;
    if (backend_ == Backend::kNpu) {
      g.npu = std::make_unique<npu::Q4nxGemv>(*runtime_, g.packed);
      g.packed = {};
    }
    return g;
  };
  const Config& c = config_;
  const auto hidden = static_cast<std::size_t>(c.hidden);
  for (int il = 0; il < c.layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto t = [&](const char* name) -> const gguf::TensorInfo& {
      return file.Tensor(p + name);
    };
    auto layer = std::make_unique<Layer>();
    if (c.IsFullAttention(il)) {
      layer->in =
          make({SegmentOf(t("attn_q.weight")), SegmentOf(t("attn_k.weight")),
                SegmentOf(t("attn_v.weight"))},
               hidden);
      const auto& o = t("attn_output.weight");
      layer->out = make({SegmentOf(o)}, o.RowElements());
    } else {
      layer->in = make(
          {SegmentOf(t("attn_qkv.weight")), SegmentOf(t("attn_gate.weight"))},
          hidden);
      layer->alpha = ToFloats(t("ssm_alpha.weight"));
      layer->beta = ToFloats(t("ssm_beta.weight"));
      const auto& o = t("ssm_out.weight");
      layer->out = make({SegmentOf(o)}, o.RowElements());
    }
    layer->ffn_in =
        make({SegmentOf(t("ffn_gate.weight")), SegmentOf(t("ffn_up.weight"))},
             hidden);
    layer->down = make({SegmentOf(t("ffn_down.weight"))},
                       static_cast<std::size_t>(c.ffn));
    layers_.push_back(std::move(layer));
  }
  lm_head_ = std::make_unique<Gemv>(
      make({SegmentOf(file.Tensor("token_embd.weight"))}, hidden));
}

Q4nxProjections::~Q4nxProjections() = default;

void Q4nxProjections::Run(Gemv& gemv, const float* x, float* y) const {
  const auto start = std::chrono::steady_clock::now();
  if (gemv.npu) {
    gemv.npu->Run(x, y);
  } else {
    q4nx::Gemv(gemv.packed, x, y);
  }
  gemv_seconds_ +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  ++gemv_calls_;
}

void Q4nxProjections::Apply(Stage stage, int layer, const float* x, int tokens,
                            float* y) const {
  const std::size_t in = StageInputs(config_, stage, layer);
  const std::size_t out = StageOutputs(config_, stage, layer);
  for (int t = 0; t < tokens; ++t) {
    const float* xt = x + static_cast<std::size_t>(t) * in;
    float* yt = y + static_cast<std::size_t>(t) * out;
    if (stage == Stage::kLmHead) {
      Run(*lm_head_, xt, yt);
      continue;
    }
    Layer& l = *layers_[static_cast<std::size_t>(layer)];
    switch (stage) {
      case Stage::kLinearIn: {
        Run(l.in, xt, yt);
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
        Run(l.in, xt, yt);
        break;
      case Stage::kMixerOut:
        Run(l.out, xt, yt);
        break;
      case Stage::kFfnIn:
        Run(l.ffn_in, xt, yt);
        break;
      case Stage::kFfnDown:
        Run(l.down, xt, yt);
        break;
      case Stage::kLmHead:
        break;
    }
  }
}

}  // namespace gufo_npu::qwen35
