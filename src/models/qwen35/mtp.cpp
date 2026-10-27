#include "models/qwen35/mtp.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "core/q4nx.hpp"
#include "core/quant.hpp"

namespace gufo_npu::qwen35 {
namespace {

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

q4nx::Segment SegmentOf(const gguf::TensorInfo& t) {
  return {t.type, t.data, static_cast<std::size_t>(t.Rows())};
}

void RmsNorm(const float* x, const float* w, int n, float eps, float* out) {
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += x[i] * x[i];
  }
  const float inv = 1.0f / std::sqrt(sum / static_cast<float>(n) + eps);
  for (int i = 0; i < n; ++i) {
    out[i] = x[i] * inv * w[i];
  }
}

float Silu(float x) {
  return x / (1.0f + std::exp(-x));
}

}  // namespace

struct MtpHead::Impl {
  Config config;
  Options options;
  Mixer mixer;
  int layer = 0;  // blk index of the MTP block
  const gguf::TensorInfo* embed = nullptr;
  std::vector<float> enorm, hnorm, head_norm, output_norm;
  std::vector<float> attn_norm, post_norm;
  q4nx::Matrix eh_proj, in, out, ffn_in, down, lm_head;
  // Scratch.
  std::vector<float> eh, x, normed, qkv, ctx, proj, ffn, act, logits;

  Impl(const gguf::File& file, Options o)
      : config(ReadConfig(file)), options(o), mixer(file) {
    const Config& c = config;
    if (c.mtp_layers < 1) {
      throw std::runtime_error("qwen35: model has no MTP layer");
    }
    layer = c.layers;
    const std::string p = "blk." + std::to_string(layer) + ".";
    auto t = [&](const char* name) -> const gguf::TensorInfo& {
      return file.Tensor(p + name);
    };
    const auto hidden = static_cast<std::size_t>(c.hidden);
    embed = &file.Tensor("token_embd.weight");
    enorm = ToFloats(t("nextn.enorm.weight"));
    hnorm = ToFloats(t("nextn.hnorm.weight"));
    head_norm = ToFloats(t("nextn.shared_head_norm.weight"));
    output_norm = ToFloats(file.Tensor("output_norm.weight"));
    attn_norm = ToFloats(t("attn_norm.weight"));
    post_norm = ToFloats(t("post_attention_norm.weight"));
    const auto& eh_t = t("nextn.eh_proj.weight");
    eh_proj = q4nx::Pack(eh_t.type, eh_t.data, eh_t.Rows(), eh_t.RowElements());
    in = q4nx::PackRows(
        {SegmentOf(t("attn_q.weight")), SegmentOf(t("attn_k.weight")),
         SegmentOf(t("attn_v.weight"))},
        hidden);
    const auto& o_t = t("attn_output.weight");
    out = q4nx::Pack(o_t.type, o_t.data, o_t.Rows(), o_t.RowElements());
    ffn_in = q4nx::PackRows(
        {SegmentOf(t("ffn_gate.weight")), SegmentOf(t("ffn_up.weight"))},
        hidden);
    const auto& d_t = t("ffn_down.weight");
    down = q4nx::Pack(d_t.type, d_t.data, d_t.Rows(), d_t.RowElements());
    lm_head = q4nx::Pack(embed->type, embed->data, embed->Rows(), hidden);

    eh.resize(2 * hidden);
    x.resize(hidden);
    normed.resize(hidden);
    qkv.resize(in.rows);
    ctx.resize(static_cast<std::size_t>(c.heads) * c.head_dim);
    proj.resize(hidden);
    ffn.resize(ffn_in.rows);
    act.resize(c.ffn);
    logits.resize(c.vocab);
  }
};

MtpHead::MtpHead(const gguf::File& file, Options options)
    : impl_(std::make_unique<Impl>(file, options)) {}

MtpHead::~MtpHead() = default;

DecodeState MtpHead::NewState() const {
  return impl_->mixer.NewState();
}

const std::vector<float>& MtpHead::Step(const float* h, int token, int position,
                                        DecodeState& state) {
  Impl& m = *impl_;
  const Config& c = m.config;
  const int d = c.hidden;
  if (token < 0 || token >= c.vocab) {
    throw std::out_of_range("qwen35: token id out of range");
  }
  // [RMS(embed(token)) * enorm ; RMS(h) * hnorm]
  quant::Dequantize(m.embed->type,
                    m.embed->data + static_cast<std::size_t>(token) *
                                        quant::RowBytes(m.embed->type, d),
                    m.normed.data(), d);
  RmsNorm(m.normed.data(), m.enorm.data(), d, c.rms_eps, m.eh.data());
  if (m.options.hidden_after_output_norm) {
    RmsNorm(h, m.output_norm.data(), d, c.rms_eps, m.normed.data());
    RmsNorm(m.normed.data(), m.hnorm.data(), d, c.rms_eps, m.eh.data() + d);
  } else {
    RmsNorm(h, m.hnorm.data(), d, c.rms_eps, m.eh.data() + d);
  }
  q4nx::Gemv(m.eh_proj, m.eh.data(), m.x.data());

  // One full-attention block.
  const int rope_position =
      m.options.position_of_token ? position : position - 1;
  RmsNorm(m.x.data(), m.attn_norm.data(), d, c.rms_eps, m.normed.data());
  q4nx::Gemv(m.in, m.normed.data(), m.qkv.data());
  m.mixer.Attention(m.layer, m.qkv.data(), rope_position, state.layers[m.layer],
                    m.ctx.data());
  q4nx::Gemv(m.out, m.ctx.data(), m.proj.data());
  for (int i = 0; i < d; ++i) {
    m.x[i] += m.proj[i];
  }
  RmsNorm(m.x.data(), m.post_norm.data(), d, c.rms_eps, m.normed.data());
  q4nx::Gemv(m.ffn_in, m.normed.data(), m.ffn.data());
  for (int i = 0; i < c.ffn; ++i) {
    m.act[i] = Silu(m.ffn[i]) * m.ffn[c.ffn + i];
  }
  q4nx::Gemv(m.down, m.act.data(), m.proj.data());
  for (int i = 0; i < d; ++i) {
    m.x[i] += m.proj[i];
  }
  RmsNorm(m.x.data(), m.head_norm.data(), d, c.rms_eps, m.normed.data());
  q4nx::Gemv(m.lm_head, m.normed.data(), m.logits.data());
  return m.logits;
}

}  // namespace gufo_npu::qwen35
