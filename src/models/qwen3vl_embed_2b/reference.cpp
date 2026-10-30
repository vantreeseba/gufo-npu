#include "models/qwen3vl_embed_2b/reference.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "core/parallel.hpp"
#include "core/quant.hpp"
#include "core/simd.hpp"
#include "core/vec.hpp"
#include "models/qwen3vl_embed_2b/ops.hpp"

namespace gufo_npu::qwen3vl_embed_2b {
namespace {

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

// out[t * out_stride + offset + o] = x_t . W_o for the `count` rows x
// (`dim` values each) and W's rows o; W is dequantized one row at a time.
void MatMul(const gguf::TensorInfo& w, const float* x, int count, float* out,
            int out_stride, int offset) {
  const auto dim = static_cast<std::size_t>(w.RowElements());
  const auto rows = static_cast<std::size_t>(w.Rows());
  const std::size_t row_bytes = quant::RowBytes(w.type, dim);
  ParallelFor(rows, [&](std::size_t begin, std::size_t end) {
    std::vector<float> row(dim);
    for (std::size_t o = begin; o < end; ++o) {
      quant::Dequantize(w.type, w.data + o * row_bytes, row.data(), dim);
      for (int t = 0; t < count; ++t) {
        out[static_cast<std::size_t>(t) * out_stride + offset + o] =
            vec::Dot(x + static_cast<std::size_t>(t) * dim, row.data(),
                     static_cast<int>(dim));
      }
    }
  });
}

}  // namespace

struct Reference::Impl {
  struct Layer {
    std::vector<float> attn_norm, ffn_norm, q_norm, k_norm;
    const gguf::TensorInfo* q = nullptr;
    const gguf::TensorInfo* k = nullptr;
    const gguf::TensorInfo* v = nullptr;
    const gguf::TensorInfo* o = nullptr;
    const gguf::TensorInfo* gate = nullptr;
    const gguf::TensorInfo* up = nullptr;
    const gguf::TensorInfo* down = nullptr;
  };

  Config config;
  const gguf::TensorInfo* embed = nullptr;
  std::vector<Layer> layers;
  std::vector<float> output_norm;
  RopeTable rope;

  explicit Impl(const gguf::File& file)
      : config(ReadConfig(file)), rope(config.head_dim, config.rope_theta) {
    embed = &file.Tensor("token_embd.weight");
    output_norm = ToFloats(file.Tensor("output_norm.weight"));
    for (int il = 0; il < config.layers; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      auto t = [&](const char* name) -> const gguf::TensorInfo& {
        return file.Tensor(p + name);
      };
      Layer l;
      l.attn_norm = ToFloats(t("attn_norm.weight"));
      l.ffn_norm = ToFloats(t("ffn_norm.weight"));
      l.q_norm = ToFloats(t("attn_q_norm.weight"));
      l.k_norm = ToFloats(t("attn_k_norm.weight"));
      l.q = &t("attn_q.weight");
      l.k = &t("attn_k.weight");
      l.v = &t("attn_v.weight");
      l.o = &t("attn_output.weight");
      l.gate = &t("ffn_gate.weight");
      l.up = &t("ffn_up.weight");
      l.down = &t("ffn_down.weight");
      layers.push_back(std::move(l));
    }
  }

  std::vector<std::vector<float>> Embed(
      const std::vector<std::vector<int>>& inputs) {
    const Config& cfg = config;
    const auto segments = Pack(inputs, cfg.vocab);
    if (segments.empty()) {
      return {};
    }
    const int n = segments.back().begin + segments.back().length;
    int longest = 0;
    for (const Segment& s : segments) {
      longest = std::max(longest, s.length);
    }
    rope.Reserve(longest);
    const ParallelRun run = [](std::size_t count,
                               const std::function<void(std::size_t)>& fn) {
      ParallelFor(count, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          fn(i);
        }
      });
    };

    const int d = cfg.hidden;
    const auto rows = static_cast<std::size_t>(n);
    std::vector<float> h(rows * d), x(rows * d), tmp(rows * d);
    std::vector<float> in(rows * cfg.InRows()), mixed(rows * cfg.QDim());
    std::vector<float> gate(rows * cfg.ffn), up(rows * cfg.ffn);
    {
      std::size_t row = 0;
      for (const auto& ids : inputs) {
        for (const int id : ids) {
          quant::Dequantize(embed->type,
                            embed->data + static_cast<std::size_t>(id) *
                                              quant::RowBytes(embed->type, d),
                            h.data() + row * d, d);
          ++row;
        }
      }
    }
    auto norm_rows = [&](const std::vector<float>& w) {
      run(rows, [&](std::size_t t) {
        vec::RmsNorm(h.data() + t * d, w.data(), d, cfg.rms_eps,
                     x.data() + t * d);
      });
    };
    auto add_tmp = [&] {
      for (std::size_t i = 0; i < h.size(); ++i) {
        h[i] += tmp[i];
      }
    };
    for (const Layer& l : layers) {
      norm_rows(l.attn_norm);
      MatMul(*l.q, x.data(), n, in.data(), cfg.InRows(), 0);
      MatMul(*l.k, x.data(), n, in.data(), cfg.InRows(), cfg.QDim());
      MatMul(*l.v, x.data(), n, in.data(), cfg.InRows(),
             cfg.QDim() + cfg.KvDim());
      Attention(cfg, rope, l.q_norm.data(), l.k_norm.data(), segments,
                in.data(), mixed.data(), run);
      MatMul(*l.o, mixed.data(), n, tmp.data(), d, 0);
      add_tmp();
      norm_rows(l.ffn_norm);
      MatMul(*l.gate, x.data(), n, gate.data(), cfg.ffn, 0);
      MatMul(*l.up, x.data(), n, up.data(), cfg.ffn, 0);
      for (std::size_t i = 0; i < gate.size(); ++i) {
        gate[i] = simd::Silu(gate[i]) * up[i];
      }
      MatMul(*l.down, gate.data(), n, tmp.data(), d, 0);
      add_tmp();
    }
    return Pool(cfg, output_norm.data(), segments, h.data());
  }
};

Reference::Reference(const gguf::File& file)
    : impl_(std::make_unique<Impl>(file)) {}

Reference::~Reference() = default;

const Config& Reference::config() const {
  return impl_->config;
}

std::vector<std::vector<float>> Reference::Embed(
    const std::vector<std::vector<int>>& inputs) {
  return impl_->Embed(inputs);
}

}  // namespace gufo_npu::qwen3vl_embed_2b
