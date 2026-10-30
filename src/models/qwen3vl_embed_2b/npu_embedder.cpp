#include "models/qwen3vl_embed_2b/npu_embedder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/burst_pool.hpp"
#include "core/quant.hpp"
#include "core/simd.hpp"
#include "core/vec.hpp"
#include "models/qwen3vl_embed_2b/npu/embed_gemm.hpp"
#include "models/qwen3vl_embed_2b/ops.hpp"
#include "npu/gemm.hpp"

namespace gufo_npu::qwen3vl_embed_2b {
namespace {

using gufo_npu::npu::Bfp16Gemm;
using gufo_npu::npu::GemmBuffer;
using gufo_npu::npu::GemmCall;
using gufo_npu::npu::GemmRuntime;
using Clock = std::chrono::steady_clock;
using Bits = std::uint16_t;  // bf16

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

// One GEMM call's rows of the batch: n tokens from `begin`, run as m rows
// (n rounded up to a chunk size of npu::kGemmChunks).
struct Chunk {
  int begin = 0;
  int n = 0;
  int m = 0;
};

std::vector<Chunk> Chunks(int rows) {
  std::vector<Chunk> chunks;
  const int max = static_cast<int>(gufo_npu::npu::kGemmMaxChunk);
  for (int begin = 0; begin < rows;) {
    const int n = std::min(max, rows - begin);
    int m = max;
    for (const std::size_t size : gufo_npu::npu::kGemmChunks) {
      if (static_cast<int>(size) >= n) {
        m = static_cast<int>(size);
        break;
      }
    }
    chunks.push_back({begin, n, m});
    begin += n;
  }
  return chunks;
}

}  // namespace

struct NpuEmbedder::Impl {
  struct Layer {
    std::unique_ptr<Bfp16Gemm> in, out, ffn, down;
    std::vector<float> attn_norm, ffn_norm, q_norm, k_norm;
  };

  // A GEMM call's device operands; two, so one chunk's call can run while
  // the CPU fills the next one's A.
  struct Slot {
    std::unique_ptr<GemmBuffer> a, c;
    GemmCall call;
    std::size_t c_sync = 0;  // bytes of C to read back
  };

  Config config;
  GemmRuntime runtime;
  const gguf::TensorInfo* embed = nullptr;
  std::vector<Layer> layers;
  std::vector<float> output_norm;
  RopeTable rope;
  Slot slots[2];
  // CPU workers, spinning for the length of an Embed: one per physical core
  // with the caller (half the hardware threads, for SMT), as the prefills.
  BurstPool pool{
      std::max<std::size_t>(std::thread::hardware_concurrency() / 2, 1) - 1};
  Profile profile;

  // The batch: residual stream, attention input (q | k | v), attention
  // output, SwiGLU output (bf16, the down projection's A).
  std::vector<float> h, in, mixed;
  std::vector<Bits> act;
  std::vector<Chunk> chunks;

  static void CheckShapes(const Config& c) {
    if (static_cast<std::size_t>(c.hidden) != npu::kHidden ||
        static_cast<std::size_t>(c.ffn) != npu::kFfn ||
        static_cast<std::size_t>(c.InRows()) != npu::kInRows ||
        static_cast<std::size_t>(c.QDim()) != npu::kQDim) {
      throw std::runtime_error(
          "qwen3vl npu embedder: this GGUF's shapes (hidden " +
          std::to_string(c.hidden) + ", ffn " + std::to_string(c.ffn) +
          ") are not Qwen3-VL-Embedding-2B's, which the GEMM program is "
          "built for");
    }
  }

  // W = the tensors' rows stacked, rounded to bfp16 on the device.
  std::unique_ptr<Bfp16Gemm> Gemm(
      const std::vector<const gguf::TensorInfo*>& parts) {
    const auto dim = static_cast<std::size_t>(parts.front()->RowElements());
    std::size_t rows = 0;
    for (const auto* t : parts) {
      if (t->RowElements() != dim) {
        throw std::runtime_error("qwen3vl: stacked weights differ in width");
      }
      rows += static_cast<std::size_t>(t->Rows());
    }
    std::vector<float> w(rows * dim);
    std::size_t at = 0;
    for (const auto* t : parts) {
      quant::Dequantize(t->type, t->data, w.data() + at, t->num_elements);
      at += t->num_elements;
    }
    return std::make_unique<Bfp16Gemm>(runtime, w.data(), rows, dim);
  }

  Impl(const gguf::File& file, const std::filesystem::path& kernel_dir)
      : config(ReadConfig(file)),
        runtime((CheckShapes(config), kernel_dir), npu::kGemmName),
        rope(config.head_dim, config.rope_theta) {
    embed = &file.Tensor("token_embd.weight");
    output_norm = ToFloats(file.Tensor("output_norm.weight"));
    for (int il = 0; il < config.layers; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      auto t = [&](const char* name) { return &file.Tensor(p + name); };
      Layer l;
      l.attn_norm = ToFloats(*t("attn_norm.weight"));
      l.ffn_norm = ToFloats(*t("ffn_norm.weight"));
      l.q_norm = ToFloats(*t("attn_q_norm.weight"));
      l.k_norm = ToFloats(*t("attn_k_norm.weight"));
      l.in = Gemm({t("attn_q.weight"), t("attn_k.weight"), t("attn_v.weight")});
      l.out = Gemm({t("attn_output.weight")});
      l.ffn = Gemm({t("ffn_gate.weight"), t("ffn_up.weight")});
      l.down = Gemm({t("ffn_down.weight")});
      layers.push_back(std::move(l));
    }
    std::size_t a_bytes = 0;
    std::size_t c_bytes = 0;
    for (const Layer& l : layers) {
      for (const auto* g : {&l.in, &l.out, &l.ffn, &l.down}) {
        a_bytes =
            std::max(a_bytes, (*g)->a_bytes(gufo_npu::npu::kGemmMaxChunk));
        c_bytes =
            std::max(c_bytes, (*g)->c_bytes(gufo_npu::npu::kGemmMaxChunk));
      }
    }
    for (Slot& s : slots) {
      s.a = std::make_unique<GemmBuffer>(runtime, a_bytes);
      s.c = std::make_unique<GemmBuffer>(runtime, c_bytes);
    }
  }

  void Run(std::size_t n, const std::function<void(std::size_t)>& fn) {
    pool.Run(n, fn);
  }

  // Runs `g` over every chunk of the batch, two calls in flight: chunk i+1
  // is written (prepare) and started before chunk i is waited for and read
  // (consume). prepare(chunk, A) fills A's first chunk.n rows (g.dim() bf16
  // each); consume(chunk, C) reads C's first chunk.n rows (g.rows() bf16).
  template<typename Prepare, typename Consume>
  void Pass(Bfp16Gemm& g, const Prepare& prepare, const Consume& consume) {
    const auto start = [&](std::size_t i) {
      Slot& s = slots[i % 2];
      const Chunk& c = chunks[i];
      prepare(c, s.a->as<Bits>());
      const auto sync = Clock::now();
      s.a->ToDevice(static_cast<std::size_t>(c.n) * g.dim() * sizeof(Bits));
      profile.sync_seconds += SecondsSince(sync);
      s.call = g.Start(*s.a, *s.c, static_cast<std::size_t>(c.m));
      s.c_sync = static_cast<std::size_t>(c.n) * g.rows() * sizeof(Bits);
      ++profile.gemms;
    };
    start(0);
    for (std::size_t i = 0; i < chunks.size(); ++i) {
      if (i + 1 < chunks.size()) {
        start(i + 1);
      }
      Slot& s = slots[i % 2];
      auto t = Clock::now();
      s.call.Wait();
      profile.gemm_seconds += SecondsSince(t);
      t = Clock::now();
      s.c->FromDevice(s.c_sync);
      profile.sync_seconds += SecondsSince(t);
      consume(chunks[i], static_cast<const Bits*>(s.c->as<Bits>()));
    }
  }

  // A rows = bf16(RMS(h row) * w).
  auto NormRows(const std::vector<float>& w) {
    return [this, &w](const Chunk& c, Bits* a) {
      const int d = config.hidden;
      Run(static_cast<std::size_t>(c.n), [&](std::size_t t) {
        const float* row = h.data() + (c.begin + t) * d;
        const float inv =
            1.0f / std::sqrt(vec::Dot(row, row, d) / static_cast<float>(d) +
                             config.rms_eps);
        simd::ScaleToBf16Row(row, inv, w.data(), a + t * d, d);
      });
    };
  }

  // C rows (width bf16 each) onto (add) or into fp32 rows of `dst`.
  auto ReadRows(std::vector<float>& dst, int width, bool add) {
    return [this, &dst, width, add](const Chunk& c, const Bits* cb) {
      Run(static_cast<std::size_t>(c.n), [&](std::size_t t) {
        simd::FromBf16Row(cb + t * width, dst.data() + (c.begin + t) * width,
                          width, add);
      });
    };
  }

  void RunLayer(const Layer& l, const std::vector<Segment>& segments) {
    const Config& cfg = config;
    const int q_dim = cfg.QDim();
    const int ffn = cfg.ffn;
    Pass(*l.in, NormRows(l.attn_norm), ReadRows(in, cfg.InRows(), false));
    const auto attention = Clock::now();
    Attention(
        cfg, rope, l.q_norm.data(), l.k_norm.data(), segments, in.data(),
        mixed.data(),
        [this](std::size_t n, const std::function<void(std::size_t)>& fn) {
          Run(n, fn);
        });
    profile.attention_seconds += SecondsSince(attention);
    Pass(
        *l.out,
        [&](const Chunk& c, Bits* a) {
          Run(static_cast<std::size_t>(c.n), [&](std::size_t t) {
            simd::ToBf16Row(mixed.data() + (c.begin + t) * q_dim, a + t * q_dim,
                            q_dim);
          });
        },
        ReadRows(h, cfg.hidden, true));
    Pass(*l.ffn, NormRows(l.ffn_norm), [&](const Chunk& c, const Bits* cb) {
      Run(static_cast<std::size_t>(c.n), [&](std::size_t t) {
        const Bits* gate = cb + t * 2 * ffn;
        simd::SwiGluBf16Row(gate, gate + ffn, act.data() + (c.begin + t) * ffn,
                            ffn);
      });
    });
    Pass(
        *l.down,
        [&](const Chunk& c, Bits* a) {
          std::memcpy(a, act.data() + static_cast<std::size_t>(c.begin) * ffn,
                      static_cast<std::size_t>(c.n) * ffn * sizeof(Bits));
        },
        ReadRows(h, cfg.hidden, true));
  }

  // One batch: inputs packed into rows, through every layer, pooled.
  std::vector<std::vector<float>> Batch(
      const std::vector<std::vector<int>>& inputs) {
    const Config& cfg = config;
    const auto segments = Pack(inputs, cfg.vocab);
    const int n = segments.back().begin + segments.back().length;
    int longest = 0;
    for (const Segment& s : segments) {
      longest = std::max(longest, s.length);
    }
    rope.Reserve(longest);
    const auto rows = static_cast<std::size_t>(n);
    const int d = cfg.hidden;
    h.resize(rows * d);
    in.resize(rows * cfg.InRows());
    mixed.resize(rows * cfg.QDim());
    act.resize(rows * cfg.ffn);
    chunks = Chunks(n);
    std::vector<int> ids;
    ids.reserve(rows);
    for (const auto& input : inputs) {
      ids.insert(ids.end(), input.begin(), input.end());
    }
    Run(rows, [&](std::size_t t) {
      quant::Dequantize(embed->type,
                        embed->data + static_cast<std::size_t>(ids[t]) *
                                          quant::RowBytes(embed->type, d),
                        h.data() + t * d, d);
    });
    for (const auto& l : layers) {
      RunLayer(l, segments);
    }
    profile.tokens += n;
    return Pool(cfg, output_norm.data(), segments, h.data());
  }

  std::vector<std::vector<float>> Embed(
      const std::vector<std::vector<int>>& inputs) {
    Pack(inputs, config.vocab);  // checks every input before any work
    const auto start = Clock::now();
    const BurstPool::Burst burst(pool);
    std::vector<std::vector<float>> out;
    out.reserve(inputs.size());
    std::size_t i = 0;
    while (i < inputs.size()) {
      // Inputs in order while they fit; a longer one alone.
      std::vector<std::vector<int>> batch;
      int rows = 0;
      while (i < inputs.size() &&
             (batch.empty() ||
              rows + static_cast<int>(inputs[i].size()) <= kMaxBatchRows)) {
        rows += static_cast<int>(inputs[i].size());
        batch.push_back(inputs[i++]);
      }
      for (auto& e : Batch(batch)) {
        out.push_back(std::move(e));
      }
    }
    profile.total_seconds += SecondsSince(start);
    profile.other_seconds = profile.total_seconds - profile.gemm_seconds -
                            profile.sync_seconds - profile.attention_seconds;
    return out;
  }
};

NpuEmbedder::NpuEmbedder(const gguf::File& file,
                         const std::filesystem::path& kernel_dir)
    : impl_(std::make_unique<Impl>(file, kernel_dir)) {}

NpuEmbedder::~NpuEmbedder() = default;

const Config& NpuEmbedder::config() const {
  return impl_->config;
}

std::vector<std::vector<float>> NpuEmbedder::Embed(
    const std::vector<std::vector<int>>& inputs) {
  return impl_->Embed(inputs);
}

const NpuEmbedder::Profile& NpuEmbedder::profile() const {
  return impl_->profile;
}

}  // namespace gufo_npu::qwen3vl_embed_2b
