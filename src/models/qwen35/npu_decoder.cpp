#include "models/qwen35/npu_decoder.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>

#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/gemv.hpp"

namespace gufo_npu::qwen35 {
namespace {

using npu::Buffer;
using npu::Chain;
using npu::Epilogue;
using npu::Prologue;
using npu::Q4nxGemv;

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

q4nx::Segment SegmentOf(const gguf::TensorInfo& t) {
  return {t.type, t.data, static_cast<std::size_t>(t.Rows())};
}

// A resnorm input region: token t's [H | D] at t * kGemvSlot fp32, then the
// norm weight. Outputs that feed it are written from H or D with token
// stride kGemvSlot.
constexpr std::size_t kH = 0;
constexpr std::size_t kD = npu::kGemvHidden * sizeof(float);
constexpr std::size_t kW = npu::kGemvNormOffset;
constexpr std::size_t kHiddenBytes = npu::kGemvHidden * sizeof(float);
constexpr std::size_t kSlotBytes = npu::kGemvSlot * sizeof(float);
// Two tokens' worth of a hidden vector in a slotted region.
constexpr std::size_t kSlottedBytes = kSlotBytes + kHiddenBytes;

// Writes a norm weight (bf16) into the w slot of a resnorm input region.
void SetNormWeight(Buffer& region, const gguf::TensorInfo& t) {
  const auto w = ToFloats(t);
  auto* bits = reinterpret_cast<std::uint16_t*>(region.as<std::uint8_t>() + kW);
  for (std::size_t i = 0; i < w.size(); ++i) {
    bits[i] = quant::FloatToBf16(w[i]);
  }
  region.ToDevice();
}

}  // namespace

struct NpuDecoder::Impl {
  struct Layer {
    bool attention = false;
    std::unique_ptr<Q4nxGemv> in, out, ffn, down;
    std::unique_ptr<Buffer> in_x, in_y, out_x, ffn_x, ffn_y;
    std::deque<Buffer> views;      // sub-buffers wired between calls (stable)
    std::unique_ptr<Chain> chain;  // out, ffn, down, next in (or LM head)
    std::vector<float> attn_norm;  // fp32, for alpha/beta on the CPU
    std::vector<float> alpha, beta;
  };

  Config config;
  Mixer mixer;
  npu::GemvRuntime runtime;
  const gguf::TensorInfo* embed = nullptr;
  std::vector<Layer> layers;
  std::unique_ptr<Q4nxGemv> lm;
  std::unique_ptr<Buffer> lm_x, lm_y, scratch;
  std::unique_ptr<Chain> first;  // layer 0's input projection
  std::vector<float> in, mixed, normed, logits;
  double npu_seconds = 0;
  long submissions = 0;

  Impl(const gguf::File& file, const std::filesystem::path& kernel_dir)
      : config(ReadConfig(file)), mixer(file), runtime(kernel_dir) {
    const Config& c = config;
    if (static_cast<std::size_t>(c.hidden) != npu::kGemvHidden) {
      throw std::runtime_error("npu decoder: hidden size must be " +
                               std::to_string(npu::kGemvHidden));
    }
    const auto hidden = static_cast<std::size_t>(c.hidden);
    embed = &file.Tensor("token_embd.weight");
    scratch = std::make_unique<Buffer>(runtime, npu::kGemvResidualBytes);
    layers.resize(c.layers);
    for (int il = 0; il < c.layers; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      auto t = [&](const char* name) -> const gguf::TensorInfo& {
        return file.Tensor(p + name);
      };
      Layer& l = layers[il];
      l.attention = c.IsFullAttention(il);
      q4nx::Matrix in_w;
      const gguf::TensorInfo* out_t = nullptr;
      if (l.attention) {
        in_w = q4nx::PackRows(
            {SegmentOf(t("attn_q.weight")), SegmentOf(t("attn_k.weight")),
             SegmentOf(t("attn_v.weight"))},
            hidden);
        out_t = &t("attn_output.weight");
      } else {
        in_w = q4nx::PackRows(
            {SegmentOf(t("attn_qkv.weight")), SegmentOf(t("attn_gate.weight"))},
            hidden);
        l.alpha = ToFloats(t("ssm_alpha.weight"));
        l.beta = ToFloats(t("ssm_beta.weight"));
      }
      if (out_t == nullptr) {
        out_t = &t("ssm_out.weight");
      }
      l.in = std::make_unique<Q4nxGemv>(runtime, in_w, Prologue::kResNorm,
                                        Epilogue::kStore);
      l.out = std::make_unique<Q4nxGemv>(
          runtime,
          q4nx::Pack(out_t->type, out_t->data, out_t->Rows(),
                     out_t->RowElements()),
          Prologue::kPlain, Epilogue::kStore);
      l.ffn = std::make_unique<Q4nxGemv>(
          runtime,
          q4nx::PackRows(
              {SegmentOf(t("ffn_gate.weight")), SegmentOf(t("ffn_up.weight"))},
              hidden, q4nx::SwiGluOrder(c.ffn)),
          Prologue::kResNorm, Epilogue::kSwiGlu);
      const auto& down = t("ffn_down.weight");
      l.down = std::make_unique<Q4nxGemv>(
          runtime,
          q4nx::Pack(down.type, down.data, down.Rows(), down.RowElements()),
          Prologue::kAct, Epilogue::kStore);

      l.in_x = std::make_unique<Buffer>(runtime, npu::kGemvInputBytes);
      l.in_y = std::make_unique<Buffer>(runtime, l.in->y_bytes());
      l.out_x = std::make_unique<Buffer>(runtime, npu::kGemvInputBytes);
      l.ffn_x = std::make_unique<Buffer>(runtime, npu::kGemvInputBytes);
      l.ffn_y = std::make_unique<Buffer>(runtime, l.ffn->y_bytes());
      SetNormWeight(*l.in_x, t("attn_norm.weight"));
      SetNormWeight(*l.ffn_x, t("post_attention_norm.weight"));
      l.attn_norm = ToFloats(t("attn_norm.weight"));
    }
    const auto& embed_t = *embed;
    lm = std::make_unique<Q4nxGemv>(
        runtime, q4nx::Pack(embed_t.type, embed_t.data, embed_t.Rows(), hidden),
        Prologue::kResNorm, Epilogue::kStore);
    lm_x = std::make_unique<Buffer>(runtime, npu::kGemvInputBytes);
    lm_y = std::make_unique<Buffer>(runtime, lm->y_bytes());
    SetNormWeight(*lm_x, file.Tensor("output_norm.weight"));

    // Wiring. in_l: residual -> ffn_x.H. out_l: -> ffn_x.D. ffn_l: residual
    // -> next.H. down_l: -> next.D. Layer 0's D stays zero.
    first = std::make_unique<Chain>(runtime);
    first->Add(
        *layers[0].in, *layers[0].in_x, *layers[0].in_y,
        layers[0].views.emplace_back(layers[0].ffn_x->Sub(kH, kSlottedBytes)));
    for (int il = 0; il < c.layers; ++il) {
      Layer& l = layers[il];
      const bool last = il + 1 == c.layers;
      Buffer& next_x = last ? *lm_x : *layers[il + 1].in_x;
      Buffer& ffn_d = l.views.emplace_back(l.ffn_x->Sub(kD, kSlottedBytes));
      Buffer& next_h = l.views.emplace_back(next_x.Sub(kH, kSlottedBytes));
      Buffer& next_d = l.views.emplace_back(next_x.Sub(kD, kSlottedBytes));
      l.chain = std::make_unique<Chain>(runtime);
      l.chain->Add(*l.out, *l.out_x, ffn_d, *scratch);
      l.chain->Add(*l.ffn, *l.ffn_x, *l.ffn_y, next_h);
      l.chain->Add(*l.down, *l.ffn_y, next_d, *scratch);
      if (last) {
        l.chain->Add(*lm, *lm_x, *lm_y, *scratch);
      } else {
        Layer& n = layers[il + 1];
        Buffer& n_res = n.views.emplace_back(n.ffn_x->Sub(kH, kSlottedBytes));
        l.chain->Add(*n.in, *n.in_x, *n.in_y, n_res);
      }
    }
    in.resize(std::max(StageOutputs(c, Stage::kLinearIn, 0),
                       StageOutputs(c, Stage::kAttentionIn, 3)));
    mixed.resize(std::max(c.LinearValueDim(), c.heads * c.head_dim));
    normed.resize(hidden);
    logits.resize(c.vocab);
  }

  void Submit(Chain& chain) {
    const auto start = std::chrono::steady_clock::now();
    chain.Run();
    npu_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    ++submissions;
  }
};

NpuDecoder::NpuDecoder(const gguf::File& file,
                       const std::filesystem::path& kernel_dir)
    : impl_(std::make_unique<Impl>(file, kernel_dir)) {}

NpuDecoder::~NpuDecoder() = default;

const Config& NpuDecoder::config() const {
  return impl_->config;
}

DecodeState NpuDecoder::NewState() const {
  return impl_->mixer.NewState();
}

double NpuDecoder::npu_seconds() const {
  return impl_->npu_seconds;
}

long NpuDecoder::submissions() const {
  return impl_->submissions;
}

const std::vector<float>& NpuDecoder::Step(int token, DecodeState& state) {
  Impl& m = *impl_;
  const Config& c = m.config;
  const int d = c.hidden;
  if (token < 0 || token >= c.vocab) {
    throw std::out_of_range("qwen35: token id out of range");
  }
  {
    Buffer h = m.layers[0].in_x->Sub(kH, kHiddenBytes);
    quant::Dequantize(m.embed->type,
                      m.embed->data + static_cast<std::size_t>(token) *
                                          quant::RowBytes(m.embed->type, d),
                      h.as<float>(), d);
    h.ToDevice();
  }
  m.Submit(*m.first);

  for (int il = 0; il < c.layers; ++il) {
    Impl::Layer& l = m.layers[il];
    l.in_y->FromDevice();
    const std::size_t rows = l.in->rows();
    std::memcpy(m.in.data(), l.in_y->as<float>(), rows * sizeof(float));
    Buffer out_x = l.out_x->Sub(0, kHiddenBytes);
    if (l.attention) {
      m.mixer.Attention(il, m.in.data(), state.position, state.layers[il],
                        out_x.as<float>());
    } else {
      // alpha and beta from the layer's normalized input: its residual is
      // what the input projection wrote to ffn_x.H.
      Buffer h = l.ffn_x->Sub(kH, kHiddenBytes);
      h.FromDevice();
      const float* hv = h.as<float>();
      float sum = 0;
      for (int i = 0; i < d; ++i) {
        sum += hv[i] * hv[i];
      }
      const float inv =
          1.0f / std::sqrt(sum / static_cast<float>(d) + c.rms_eps);
      for (int i = 0; i < d; ++i) {
        m.normed[i] = hv[i] * inv * l.attn_norm[i];
      }
      const int heads = c.linear_value_heads;
      float* ab = m.in.data() + rows;
      for (int head = 0; head < 2 * heads; ++head) {
        const float* w = (head < heads ? l.alpha.data() : l.beta.data()) +
                         static_cast<std::size_t>(head % heads) * d;
        float acc = 0;
        for (int i = 0; i < d; ++i) {
          acc += w[i] * m.normed[i];
        }
        ab[head] = acc;
      }
      m.mixer.Linear(il, m.in.data(), state.layers[il], out_x.as<float>());
    }
    out_x.ToDevice();
    m.Submit(*l.chain);
  }
  ++state.position;

  m.lm_y->FromDevice();
  std::memcpy(m.logits.data(), m.lm_y->as<float>(), c.vocab * sizeof(float));
  return m.logits;
}

}  // namespace gufo_npu::qwen35
