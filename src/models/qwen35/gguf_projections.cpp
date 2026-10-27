#include "models/qwen35/gguf_projections.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "core/quant.hpp"
#include "core/thread_pool.hpp"
#include "core/vec.hpp"

namespace gufo_npu::qwen35 {
namespace {

using gguf::TensorInfo;
using Rows = GgufProjections::Rows;

Rows All(const TensorInfo& w) {
  return {&w, 0, static_cast<std::size_t>(w.Rows())};
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

double DotDouble(const float* a, const float* b, std::size_t n) {
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(a[i]) * b[i];
  }
  return sum;
}

// A Q8_0 row (blocks of an fp16 scale and 32 int8) dotted with fp32 x
// without dequantizing it first: most of the 35B's bytes per token are Q8_0.
float DotQ8_0(const std::uint8_t* row, const float* x, std::size_t n) {
  constexpr std::size_t kBlock = 32;
  constexpr std::size_t kBlockBytes = 34;
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
  __m256 acc = _mm256_setzero_ps();
  for (std::size_t b = 0; b < n / kBlock; ++b, row += kBlockBytes) {
    std::uint16_t bits;
    std::memcpy(&bits, row, sizeof(bits));
    const __m256 d = _mm256_set1_ps(_cvtsh_ss(bits));
    const auto* q = reinterpret_cast<const __m128i*>(row + 2);
    const float* xb = x + b * kBlock;
    __m256 block = _mm256_setzero_ps();
    for (int j = 0; j < 2; ++j) {
      const __m128i q16 = _mm_loadu_si128(q + j);
      const __m256 lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16));
      const __m256 hi =
          _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16, 8)));
      block = _mm256_fmadd_ps(lo, _mm256_loadu_ps(xb + 16 * j), block);
      block = _mm256_fmadd_ps(hi, _mm256_loadu_ps(xb + 16 * j + 8), block);
    }
    acc = _mm256_fmadd_ps(d, block, acc);
  }
  const __m128 sum4 =
      _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  const __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
  return _mm_cvtss_f32(_mm_add_ss(sum2, _mm_movehdup_ps(sum2)));
#else
  float sum = 0;
  for (std::size_t b = 0; b < n / kBlock; ++b, row += kBlockBytes) {
    std::uint16_t bits;
    std::memcpy(&bits, row, sizeof(bits));
    const auto* q = reinterpret_cast<const std::int8_t*>(row + 2);
    float block = 0;
    for (std::size_t j = 0; j < kBlock; ++j) {
      block += static_cast<float>(q[j]) * x[b * kBlock + j];
    }
    sum += quant::HalfToFloat(bits) * block;
  }
  return sum;
#endif
}

}  // namespace

struct GgufProjections::Layer {
  std::vector<Rows> linear_in;     // attn_qkv, attn_gate, ssm_alpha, ssm_beta
  std::vector<Rows> attention_in;  // attn_q, attn_k, attn_v
  std::vector<Rows> mixer_out;     // ssm_out or attn_output
  std::vector<Rows> ffn_in;        // gate, up: dense or shared expert
  std::vector<Rows> ffn_down;
  std::vector<Rows> router;  // ffn_gate_inp, ffn_gate_inp_shexp
  std::vector<Rows> mtp_in;  // nextn.eh_proj (MTP layer)
  const TensorInfo* gate_exps = nullptr;
  const TensorInfo* up_exps = nullptr;
  const TensorInfo* down_exps = nullptr;
};

// Half the hardware threads (memory bandwidth, not cores, bounds these
// products): every Run waits for every worker, so a pool that oversubscribes
// the CPU pays a scheduler time slice per call.
struct GgufProjections::Pool {
  SpinPool pool{
      std::max<std::size_t>(std::thread::hardware_concurrency() / 2, 1) - 1};
};

GgufProjections::GgufProjections(const gguf::File& file, Precision precision)
    : config_(ReadConfig(file)),
      precision_(precision),
      pool_(std::make_unique<Pool>()) {
  const Config& c = config_;
  const auto& embed = file.Tensor("token_embd.weight");
  lm_head_ = c.tied_embeddings ? &embed : &file.Tensor("output.weight");
  CheckShape(*lm_head_, c.hidden, c.vocab);

  const auto key_dim = static_cast<std::uint64_t>(c.LinearKeyDim());
  const auto value_dim = static_cast<std::uint64_t>(c.LinearValueDim());
  const auto attn_dim = static_cast<std::uint64_t>(c.heads * c.head_dim);
  const auto kv_dim = static_cast<std::uint64_t>(c.kv_heads * c.head_dim);
  // The decoder layers, then the MTP layers (full attention).
  for (int il = 0; il < c.layers + c.mtp_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto t = [&](const char* name) -> const TensorInfo& {
      return file.Tensor(p + name);
    };
    Layer layer;
    if (il >= c.layers) {
      const auto& eh = t("nextn.eh_proj.weight");
      CheckShape(eh, 2 * static_cast<std::uint64_t>(c.hidden), c.hidden);
      layer.mtp_in = {All(eh)};
    }
    if (c.IsMoe()) {
      const auto& router = t("ffn_gate_inp.weight");
      const auto& shared_gate = t("ffn_gate_inp_shexp.weight");
      CheckShape(router, c.hidden, c.experts);
      CheckShape(shared_gate, c.hidden, 1);
      layer.router = {All(router), All(shared_gate)};
      layer.gate_exps = &t("ffn_gate_exps.weight");
      layer.up_exps = &t("ffn_up_exps.weight");
      layer.down_exps = &t("ffn_down_exps.weight");
      CheckShape(*layer.gate_exps, c.hidden, c.expert_ffn, c.experts);
      CheckShape(*layer.up_exps, c.hidden, c.expert_ffn, c.experts);
      CheckShape(*layer.down_exps, c.expert_ffn, c.hidden, c.experts);
    }
    const char* suffix = c.IsMoe() ? "_shexp.weight" : ".weight";
    const auto& gate = file.Tensor(p + "ffn_gate" + suffix);
    const auto& up = file.Tensor(p + "ffn_up" + suffix);
    const auto& down = file.Tensor(p + "ffn_down" + suffix);
    CheckShape(gate, c.hidden, c.FfnWidth());
    CheckShape(up, c.hidden, c.FfnWidth());
    CheckShape(down, c.FfnWidth(), c.hidden);
    layer.ffn_in = {All(gate), All(up)};
    layer.ffn_down = {All(down)};
    if (c.IsFullAttention(il)) {
      const auto& q = t("attn_q.weight");
      const auto& k = t("attn_k.weight");
      const auto& v = t("attn_v.weight");
      const auto& o = t("attn_output.weight");
      CheckShape(q, c.hidden, 2 * attn_dim);
      CheckShape(k, c.hidden, kv_dim);
      CheckShape(v, c.hidden, kv_dim);
      CheckShape(o, attn_dim, c.hidden);
      layer.attention_in = {All(q), All(k), All(v)};
      layer.mixer_out = {All(o)};
    } else {
      const auto& qkv = t("attn_qkv.weight");
      const auto& z = t("attn_gate.weight");
      const auto& alpha = t("ssm_alpha.weight");
      const auto& beta = t("ssm_beta.weight");
      const auto& out = t("ssm_out.weight");
      CheckShape(qkv, c.hidden, 2 * key_dim + value_dim);
      CheckShape(z, c.hidden, value_dim);
      CheckShape(alpha, c.hidden, c.linear_value_heads);
      CheckShape(beta, c.hidden, c.linear_value_heads);
      CheckShape(out, value_dim, c.hidden);
      layer.linear_in = {All(qkv), All(z), All(alpha), All(beta)};
      layer.mixer_out = {All(out)};
    }
    layers_.push_back(std::move(layer));
  }
}

GgufProjections::~GgufProjections() = default;

void GgufProjections::Apply(Stage stage, int layer, const float* x, int tokens,
                            float* y) const {
  if (stage == Stage::kLmHead) {
    return MatMul({All(*lm_head_)}, x, tokens, y);
  }
  const Layer& l = layers_.at(static_cast<std::size_t>(layer));
  const std::vector<Rows>* parts = nullptr;
  switch (stage) {
    case Stage::kLinearIn:
      parts = &l.linear_in;
      break;
    case Stage::kAttentionIn:
      parts = &l.attention_in;
      break;
    case Stage::kMixerOut:
      parts = &l.mixer_out;
      break;
    case Stage::kFfnIn:
      parts = &l.ffn_in;
      break;
    case Stage::kFfnDown:
      parts = &l.ffn_down;
      break;
    case Stage::kRouter:
      parts = &l.router;
      break;
    case Stage::kMtpIn:
      parts = &l.mtp_in;
      break;
    case Stage::kLmHead:
    case Stage::kExpertIn:
    case Stage::kExpertDown:
      break;
  }
  if (parts == nullptr || parts->empty()) {
    throw std::logic_error("qwen35: no weights for this stage and layer");
  }
  MatMul(*parts, x, tokens, y);
}

void GgufProjections::ApplyExpert(Stage stage, int layer, int expert,
                                  const float* x, int tokens, float* y) const {
  const Layer& l = layers_.at(static_cast<std::size_t>(layer));
  if (l.gate_exps == nullptr || expert < 0 || expert >= config_.experts) {
    throw std::logic_error("qwen35: no such expert");
  }
  const auto e = static_cast<std::size_t>(expert);
  const auto ffn = static_cast<std::size_t>(config_.expert_ffn);
  const auto hidden = static_cast<std::size_t>(config_.hidden);
  switch (stage) {
    case Stage::kExpertIn:
      return MatMul({{l.gate_exps, e * ffn, ffn}, {l.up_exps, e * ffn, ffn}}, x,
                    tokens, y);
    case Stage::kExpertDown:
      return MatMul({{l.down_exps, e * hidden, hidden}}, x, tokens, y);
    default:
      throw std::logic_error("qwen35: ApplyExpert takes expert stages only");
  }
}

void GgufProjections::MatMul(const std::vector<Rows>& parts, const float* x,
                             int tokens, float* y) const {
  // Jobs of kChunk rows; each dequantizes a row once and applies it to every
  // token. Every output is one dot product, so results do not depend on the
  // split or on how many tokens come together.
  constexpr std::size_t kChunk = 16;
  const auto in = static_cast<std::size_t>(parts.front().w->RowElements());
  std::size_t stride = 0;
  std::vector<std::size_t> first_job = {0};
  for (const auto& part : parts) {
    stride += part.rows;
    first_job.push_back(first_job.back() + (part.rows + kChunk - 1) / kChunk);
  }
  const bool exact = precision_ == Precision::kDouble;
  pool_->pool.Run(first_job.back(), [&](std::size_t job) {
    std::size_t p = 0;
    std::size_t offset = 0;
    while (job >= first_job[p + 1]) {
      offset += parts[p].rows;
      ++p;
    }
    const Rows& part = parts[p];
    const auto row_bytes = quant::RowBytes(part.w->type, in);
    const std::size_t begin = (job - first_job[p]) * kChunk;
    const std::size_t end = std::min(part.rows, begin + kChunk);
    thread_local std::vector<float> row;
    row.resize(in);
    for (std::size_t r = begin; r < end; ++r) {
      const std::uint8_t* data = part.w->data + (part.first + r) * row_bytes;
      if (!exact && part.w->type == gguf::GgmlType::kQ8_0) {
        for (int t = 0; t < tokens; ++t) {
          y[static_cast<std::size_t>(t) * stride + offset + r] =
              DotQ8_0(data, x + static_cast<std::size_t>(t) * in, in);
        }
        continue;
      }
      quant::Dequantize(part.w->type, data, row.data(), in);
      for (int t = 0; t < tokens; ++t) {
        const float* xt = x + static_cast<std::size_t>(t) * in;
        y[static_cast<std::size_t>(t) * stride + offset + r] =
            exact ? static_cast<float>(DotDouble(row.data(), xt, in))
                  : vec::Dot(row.data(), xt, static_cast<int>(in));
      }
    }
  });
}

}  // namespace gufo_npu::qwen35
