#include "models/qwen35/npu_prefill.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "core/burst_pool.hpp"
#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "core/simd.hpp"
#include "core/vec.hpp"
#include "models/qwen35/prefill_mixer.hpp"
#include "npu/gemm.hpp"

namespace gufo_npu::qwen35 {
namespace {

using npu::Bfp16Gemm;
using npu::GemmBuffer;
using vec::Dot;
using vec::RmsNorm;

constexpr int kMaxChunk = static_cast<int>(npu::kGemmMaxChunk);

// The model the chunk plan is chosen with, measured on the 2B (BENCHMARKS.md,
// "Prefill GEMM" and "Where the time goes"): the GEMMs' throughput at each
// chunk size of npu::kGemmChunks (a 256-row call runs at ~11 TFLOPS, a
// 1,024-row one at ~17), and the CPU's time per token between them (~0.14 ms
// at 1k tokens: mixers, norms, conversions).
constexpr double kGemmTflops[] = {11.0, 14.5, 17.0};
static_assert(std::size(kGemmTflops) == std::size(npu::kGemmChunks));
constexpr double kCpuSecondsPerToken = 0.14e-3;

// One pass of a prefill: `lanes` chunks of `m` GEMM rows over `tokens`
// prompt tokens (the last chunk padded).
struct Pass {
  int m = 0;
  int lanes = 0;
  int tokens = 0;
};

// The passes over `count` tokens with the least estimated time, for GEMMs of
// `flops` per token and `cpu` seconds of CPU work per token. One lane takes
// its GEMMs' time and then its CPU time; two lanes overlap one chunk's GEMMs
// with the other's CPU work, about max(GEMMs of both, CPU of both). So 1,000
// tokens run as 512 + 488 (GEMMs ~0.2 s) rather than four chunks of 256
// (~0.25 s) or one of 1,024 without overlap.
std::vector<Pass> PlanPasses(int count, int max_lanes, double flops,
                             double cpu) {
  const auto npu = [&](std::size_t chunk) {
    return flops * static_cast<double>(npu::kGemmChunks[chunk]) /
           (kGemmTflops[chunk] * 1e12);
  };
  // best[r]: least time for the last r tokens, and the pass that starts it.
  std::vector<double> best(static_cast<std::size_t>(count) + 1, 0.0);
  std::vector<Pass> first(static_cast<std::size_t>(count) + 1);
  for (int r = 1; r <= count; ++r) {
    best[r] = INFINITY;
    for (std::size_t c = 0; c < std::size(npu::kGemmChunks); ++c) {
      const int m = static_cast<int>(npu::kGemmChunks[c]);
      for (int lanes = 1; lanes <= max_lanes; ++lanes) {
        if (lanes > 1 && r <= (lanes - 1) * m) {
          continue;  // a lane would be empty
        }
        const int tokens = std::min(r, lanes * m);
        const double t = lanes == 1 ? npu(c) + cpu * tokens
                                    : std::max(lanes * npu(c), cpu * tokens);
        if (t + best[r - tokens] < best[r]) {
          best[r] = t + best[r - tokens];
          first[r] = {m, lanes, tokens};
        }
      }
    }
  }
  std::vector<Pass> plan;
  for (int r = count; r > 0; r -= first[r].tokens) {
    plan.push_back(first[r]);
  }
  return plan;
}

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

q4nx::Segment SegmentOf(const gguf::TensorInfo& t) {
  return {t.type, t.data, static_cast<std::size_t>(t.Rows())};
}

// The weights NpuDecoder multiplies (the same q4nx packing), row-major fp32.
std::vector<float> Q4nxWeights(const std::vector<q4nx::Segment>& segments,
                               std::size_t dim) {
  const auto packed = q4nx::PackRows(segments, dim);
  auto w = q4nx::Unpack(packed);
  w.resize(packed.rows * dim);
  return w;
}

}  // namespace

struct NpuPrefill::Impl {
  struct Layer {
    PrefillMixerWeights mix;
    std::unique_ptr<Bfp16Gemm> in, out, ffn, down;
    std::vector<float> attn_norm, post_norm;
  };

  Config config;
  npu::GemmRuntime runtime;
  const gguf::TensorInfo* embed = nullptr;
  std::vector<Layer> layers;

  // MTP: eh_proj, then the block's input projection (its keys and values).
  bool has_mtp = false;
  Layer mtp;
  std::unique_ptr<Bfp16Gemm> eh_proj;
  std::vector<float> enorm, hnorm, output_norm;

  // One chunk of prompt tokens in flight: its GEMM operands (A: rows x dim
  // bf16, C: rows x out bf16) and its CPU scratch. Two lanes run two
  // consecutive chunks through the layers together, so one lane's GEMMs run
  // on the NPU while the CPU works on the other's.
  struct Lane {
    int n = 0;         // tokens
    int m = 0;         // GEMM rows (n padded to a chunk size)
    int position = 0;  // of the first
    std::unique_ptr<GemmBuffer> a, c;
    npu::GemmCall call;
    std::size_t c_sync = 0;  // bytes of C to sync once the call is done
    Floats h;                // residual stream
    Floats normed;           // RMS(h) * attn_norm, for the DeltaNet gates
    Floats in;               // input projection, fp32
    Floats mixed;            // mixer output
    std::unique_ptr<PrefillMixer> mixer;  // its scratch and RoPE angles
  };
  Lane lanes[2];
  // CPU workers, spinning for the length of a Run or FillMtp: one per
  // physical core with the caller (half the hardware threads, for SMT).
  // Measured on the 16-core box at 1k tokens: 15 workers 0.61 s, 7 workers
  // 0.75 s, 23 workers 0.78 s, 31 workers 1.4 s (spinners on every hardware
  // thread delay the NPU calls' completions).
  BurstPool pool{
      std::max<std::size_t>(std::thread::hardware_concurrency() / 2, 1) - 1};

  Profile profile;
  // GEMM work per token of a prefill pass and of the MTP cache fill.
  double flops_per_token = 0;
  double mtp_flops_per_token = 0;

  Layer MakeLayer(const gguf::File& file, int il, bool full) {
    const Config& cfg = config;
    const auto hidden = static_cast<std::size_t>(cfg.hidden);
    const std::string p = "blk." + std::to_string(il) + ".";
    auto t = [&](const char* name) -> const gguf::TensorInfo& {
      return file.Tensor(p + name);
    };
    auto gemm = [&](const std::vector<q4nx::Segment>& segments,
                    std::size_t dim) {
      const auto w = Q4nxWeights(segments, dim);
      std::size_t rows = 0;
      for (const auto& s : segments) {
        rows += s.rows;
      }
      return std::make_unique<Bfp16Gemm>(runtime, w.data(), rows, dim);
    };
    Layer l;
    l.mix = LoadPrefillMixer(file, cfg, il);
    l.attn_norm = ToFloats(t("attn_norm.weight"));
    if (l.mix.attention) {
      l.in = gemm({SegmentOf(t("attn_q.weight")), SegmentOf(t("attn_k.weight")),
                   SegmentOf(t("attn_v.weight"))},
                  hidden);
    } else {
      l.in = gemm(
          {SegmentOf(t("attn_qkv.weight")), SegmentOf(t("attn_gate.weight"))},
          hidden);
    }
    if (!full) {
      return l;  // the MTP block: only its input projection is needed
    }
    const auto& out =
        t(l.mix.attention ? "attn_output.weight" : "ssm_out.weight");
    l.out = gemm({SegmentOf(out)}, static_cast<std::size_t>(out.RowElements()));
    l.ffn =
        gemm({SegmentOf(t("ffn_gate.weight")), SegmentOf(t("ffn_up.weight"))},
             hidden);
    const auto& down = t("ffn_down.weight");
    l.down =
        gemm({SegmentOf(down)}, static_cast<std::size_t>(down.RowElements()));
    l.post_norm = ToFloats(t("post_attention_norm.weight"));
    return l;
  }

  Impl(const gguf::File& file, const std::filesystem::path& kernel_dir)
      : config(ReadConfig(file)), runtime(kernel_dir) {
    const Config& cfg = config;
    if (cfg.experts != 0 || !cfg.tied_embeddings) {
      throw std::runtime_error(
          "npu prefill: dense, tied-embedding models only");
    }
    embed = &file.Tensor("token_embd.weight");
    for (int il = 0; il < cfg.layers; ++il) {
      layers.push_back(MakeLayer(file, il, true));
    }
    if (cfg.mtp_layers > 0) {
      has_mtp = true;
      const std::string p = "blk." + std::to_string(cfg.layers) + ".";
      mtp = MakeLayer(file, cfg.layers, false);
      const auto& eh = file.Tensor(p + "nextn.eh_proj.weight");
      const auto w = Q4nxWeights({SegmentOf(eh)},
                                 static_cast<std::size_t>(eh.RowElements()));
      eh_proj = std::make_unique<Bfp16Gemm>(
          runtime, w.data(), static_cast<std::size_t>(eh.Rows()),
          static_cast<std::size_t>(eh.RowElements()));
      enorm = ToFloats(file.Tensor(p + "nextn.enorm.weight"));
      hnorm = ToFloats(file.Tensor(p + "nextn.hnorm.weight"));
      output_norm = ToFloats(file.Tensor("output_norm.weight"));
    }
    auto flops = [](const std::unique_ptr<Bfp16Gemm>& g) {
      return g ? 2.0 * static_cast<double>(g->rows() * g->dim()) : 0.0;
    };
    for (const auto& l : layers) {
      flops_per_token +=
          flops(l.in) + flops(l.out) + flops(l.ffn) + flops(l.down);
    }
    mtp_flops_per_token = flops(eh_proj) + flops(mtp.in);

    std::size_t a_bytes = 0;
    std::size_t c_bytes = 0;
    auto grow = [&](const std::unique_ptr<Bfp16Gemm>& g) {
      if (g) {
        a_bytes = std::max(a_bytes, g->a_bytes(npu::kGemmMaxChunk));
        c_bytes = std::max(c_bytes, g->c_bytes(npu::kGemmMaxChunk));
      }
    };
    std::size_t in_rows = 0;
    for (const auto& l : layers) {
      for (const auto* g : {&l.in, &l.out, &l.ffn, &l.down}) {
        grow(*g);
      }
      in_rows = std::max(in_rows, l.in->rows());
    }
    grow(mtp.in);
    grow(eh_proj);
    if (mtp.in) {
      in_rows = std::max(in_rows, mtp.in->rows());  // FillMtpChunk's `in`
    }
    const auto rows = static_cast<std::size_t>(kMaxChunk);
    const auto d = static_cast<std::size_t>(cfg.hidden);
    for (Lane& lane : lanes) {
      lane.a = std::make_unique<GemmBuffer>(runtime, a_bytes);
      lane.c = std::make_unique<GemmBuffer>(runtime, c_bytes);
      lane.h.resize(rows * d);
      lane.normed.resize(rows * d);
      lane.in.resize(rows * in_rows);
      lane.mixed.resize(
          rows * std::max(cfg.LinearValueDim(), cfg.heads * cfg.head_dim));
      lane.mixer = std::make_unique<PrefillMixer>(cfg, pool, kMaxChunk);
    }
  }

  // Starts `g` over the lane's first n rows of A into C.
  void Start(Bfp16Gemm& g, Lane& lane) {
    const auto start = Clock::now();
    lane.a->ToDevice(static_cast<std::size_t>(lane.n) * g.dim() *
                     sizeof(std::uint16_t));
    profile.sync_seconds += SecondsSince(start);
    lane.call = g.Start(*lane.a, *lane.c, static_cast<std::size_t>(lane.m));
    lane.c_sync =
        static_cast<std::size_t>(lane.n) * g.rows() * sizeof(std::uint16_t);
    ++profile.gemms;
  }

  // Waits for the lane's call and makes its C visible.
  void Finish(Lane& lane) {
    auto start = Clock::now();
    lane.call.Wait();
    profile.gemm_seconds += SecondsSince(start);
    start = Clock::now();
    lane.c->FromDevice(lane.c_sync);
    profile.sync_seconds += SecondsSince(start);
  }

  // A row t = bf16(RMS(x_t) * w), x rows `stride` apart; with `keep`, also
  // the fp32 result.
  void NormToA(Lane& lane, const float* x, std::size_t stride,
               const std::vector<float>& w, float* keep) {
    const int d = config.hidden;
    auto* bits = lane.a->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      const float* row = x + t * stride;
      const float inv =
          1.0f /
          std::sqrt(Dot(row, row, d) / static_cast<float>(d) + config.rms_eps);
      simd::ScaleToBf16Row(row, inv, w.data(), bits + t * d, d);
      if (keep != nullptr) {
        RmsNorm(row, w.data(), d, config.rms_eps, keep + t * d);
      }
    });
  }

  // A row t = bf16(x_t), `width` values.
  void ToA(Lane& lane, const float* x, int width) {
    auto* bits = lane.a->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      simd::ToBf16Row(x + t * width, bits + t * width, width);
    });
  }

  // C rows (`width` bf16 each) to fp32 `out`; with `add`, onto it.
  void FromC(Lane& lane, int width, float* out, bool add) {
    const auto* bits = lane.c->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      simd::FromBf16Row(bits + t * width, out + t * width, width, add);
    });
  }

  // C row = [gate | up] (2 ffn bf16) -> A row = silu(gate) * up.
  void SwiGlu(Lane& lane, int ffn) {
    const auto* cb = lane.c->as<std::uint16_t>();
    auto* bits = lane.a->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      const std::uint16_t* gate = cb + t * 2 * ffn;
      simd::SwiGluBf16Row(gate, gate + ffn, bits + t * ffn, ffn);
    });
  }

  // Throws before any work (as the decoder does) when an id is not in the
  // vocabulary; the jobs index the embedding table with them.
  void CheckTokens(const int* tokens, int count) const {
    for (int t = 0; t < count; ++t) {
      if (tokens[t] < 0 || tokens[t] >= config.vocab) {
        throw std::out_of_range("qwen35: token id out of range");
      }
    }
  }

  void Embed(Lane& lane, const int* tokens) {
    const int d = config.hidden;
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      quant::Dequantize(embed->type,
                        embed->data + static_cast<std::size_t>(tokens[t]) *
                                          quant::RowBytes(embed->type, d),
                        lane.h.data() + t * d, d);
    });
    lane.mixer->SetChunk(lane.n, lane.position);
  }

  // Seconds profiled so far, to charge the rest of a call to `other`.
  double Profiled() const {
    return profile.gemm_seconds + profile.sync_seconds +
           profile.deltanet_seconds + profile.attention_seconds;
  }

  // Runs the first `count` lanes (consecutive chunks, embedded) through
  // every layer. Each lane has at most one GEMM in flight; calls run in
  // start order, so while the CPU finishes one lane's step the NPU runs the
  // other lane's GEMM.
  void RunLanes(int count, DecodeState& state) {
    const Config& cfg = config;
    const int d = cfg.hidden;
    for (int il = 0; il < cfg.layers; ++il) {
      Layer& l = layers[il];
      auto& st = state.layers[il];
      for (int i = 0; i < count; ++i) {
        Lane& lane = lanes[i];
        if (il > 0) {
          Finish(lane);  // the previous layer's down projection
          FromC(lane, d, lane.h.data(), true);
        }
        NormToA(lane, lane.h.data(), d, l.attn_norm,
                l.mix.attention ? nullptr : lane.normed.data());
        Start(*l.in, lane);
      }
      for (int i = 0; i < count; ++i) {
        Lane& lane = lanes[i];
        Finish(lane);
        FromC(lane, static_cast<int>(l.in->rows()), lane.in.data(), false);
        const auto mixer_start = Clock::now();
        if (l.mix.attention) {
          lane.mixer->Attention(lane.in.data(), l.in->rows(), l.mix, st,
                                lane.mixed.data());
          profile.attention_seconds += SecondsSince(mixer_start);
        } else {
          lane.mixer->DeltaNet(lane.in.data(), l.in->rows(), lane.normed.data(),
                               l.mix, st, lane.mixed.data());
          profile.deltanet_seconds += SecondsSince(mixer_start);
        }
        ToA(lane, lane.mixed.data(), static_cast<int>(l.out->dim()));
        Start(*l.out, lane);
      }
      for (int i = 0; i < count; ++i) {
        Lane& lane = lanes[i];
        Finish(lane);
        FromC(lane, d, lane.h.data(), true);
        NormToA(lane, lane.h.data(), d, l.post_norm, nullptr);
        Start(*l.ffn, lane);
      }
      for (int i = 0; i < count; ++i) {
        Lane& lane = lanes[i];
        Finish(lane);
        SwiGlu(lane, static_cast<int>(l.ffn->rows() / 2));
        Start(*l.down, lane);
      }
    }
    for (int i = 0; i < count; ++i) {
      Finish(lanes[i]);
      FromC(lanes[i], d, lanes[i].h.data(), true);
    }
  }

  // Runs `count` tokens, up to two chunks at a time, in the passes
  // PlanPasses picks.
  void Run(const int* tokens, int count, DecodeState& state, float* hidden) {
    const int d = config.hidden;
    CheckTokens(tokens, count);
    int done = 0;
    for (const Pass& pass :
         PlanPasses(count, 2, flops_per_token, kCpuSecondsPerToken)) {
      const auto start = Clock::now();
      const double before = Profiled();
      int used = 0;
      int taken = 0;
      for (int i = 0; i < pass.lanes; ++i) {
        Lane& lane = lanes[i];
        lane.m = pass.m;
        lane.n = std::min(pass.m, pass.tokens - taken);
        lane.position = state.position + taken;
        Embed(lane, tokens + done + taken);
        taken += lane.n;
        ++used;
      }
      RunLanes(used, state);
      state.position += taken;
      if (hidden != nullptr) {
        int row = done;
        for (int i = 0; i < used; ++i) {
          std::copy_n(lanes[i].h.data(),
                      static_cast<std::size_t>(lanes[i].n) * d,
                      hidden + static_cast<std::size_t>(row) * d);
          row += lanes[i].n;
        }
      }
      done += taken;
      profile.other_seconds += SecondsSince(start) - (Profiled() - before);
    }
  }

  void FillMtpChunk(const float* hidden, const int* tokens, const Pass& pass,
                    int position, DecodeState& state) {
    const Config& cfg = config;
    const int d = cfg.hidden;
    const int n = pass.tokens;
    const auto start = Clock::now();
    const double before = Profiled();
    Lane& lane = lanes[0];
    lane.n = n;
    lane.m = pass.m;
    lane.position = position;
    // [RMS(embed(token)) * enorm ; RMS(RMS(h) * output_norm) * hnorm].
    auto* bits = lane.a->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
      thread_local std::vector<float> x, y;
      x.resize(d);
      y.resize(d);
      quant::Dequantize(embed->type,
                        embed->data + static_cast<std::size_t>(tokens[t]) *
                                          quant::RowBytes(embed->type, d),
                        x.data(), d);
      RmsNorm(x.data(), enorm.data(), d, cfg.rms_eps, y.data());
      std::uint16_t* row = bits + t * 2 * d;
      simd::ToBf16Row(y.data(), row, d);
      RmsNorm(hidden + t * d, output_norm.data(), d, cfg.rms_eps, x.data());
      RmsNorm(x.data(), hnorm.data(), d, cfg.rms_eps, y.data());
      simd::ToBf16Row(y.data(), row + d, d);
    });
    Start(*eh_proj, lane);
    Finish(lane);
    FromC(lane, d, lane.h.data(), false);
    NormToA(lane, lane.h.data(), d, mtp.attn_norm, nullptr);
    Start(*mtp.in, lane);
    Finish(lane);
    FromC(lane, static_cast<int>(mtp.in->rows()), lane.in.data(), false);
    lane.mixer->SetChunk(lane.n, lane.position);
    lane.mixer->AppendKv(lane.in.data(), mtp.in->rows(), mtp.mix,
                         state.layers[cfg.layers]);
    profile.other_seconds += SecondsSince(start) - (Profiled() - before);
  }
};

NpuPrefill::NpuPrefill(const gguf::File& file,
                       const std::filesystem::path& kernel_dir)
    : impl_(std::make_unique<Impl>(file, kernel_dir)) {}

NpuPrefill::~NpuPrefill() = default;

const Config& NpuPrefill::config() const {
  return impl_->config;
}

void NpuPrefill::Run(const int* tokens, int count, DecodeState& state,
                     float* hidden) {
  const BurstPool::Burst burst(impl_->pool);
  impl_->Run(tokens, count, state, hidden);
}

void NpuPrefill::FillMtp(const float* hidden, const int* tokens, int count,
                         int position, DecodeState& mtp) {
  if (!impl_->has_mtp) {
    throw std::runtime_error("qwen35: model has no MTP layer");
  }
  const int d = impl_->config.hidden;
  impl_->CheckTokens(tokens, count);
  const BurstPool::Burst burst(impl_->pool);
  // One lane, and almost no CPU work between its two GEMMs.
  int done = 0;
  for (const Pass& pass :
       PlanPasses(count, 1, impl_->mtp_flops_per_token, 0.0)) {
    impl_->FillMtpChunk(hidden + static_cast<std::size_t>(done) * d,
                        tokens + done, pass, position + done, mtp);
    done += pass.tokens;
  }
}

const NpuPrefill::Profile& NpuPrefill::profile() const {
  return impl_->profile;
}

PromptPrefill PrefillPrompt(NpuPrefill& prefill) {
  return [&prefill](const std::vector<int>& prompt, DecodeState& state,
                    DecodeState* mtp, float* hidden) {
    // The whole prompt goes through one prefill: the decoder then runs only
    // the LM head on its last row (NpuDecoder::Logits). A prefill pass costs
    // ~0.07 s however short the prompt (98 GEMM calls of at least 256 rows
    // each), the decode path ~28 ms per two tokens. Shorter prompts go
    // through the decoder alone (crossover measured in BENCHMARKS.md).
    constexpr int kMinPrefillPrompt = 7;
    const int n = static_cast<int>(prompt.size());
    if (n < kMinPrefillPrompt) {
      return 0;
    }
    // Every id, including the last (the MTP cache reads prompt[k]), before
    // any state changes.
    for (const int id : prompt) {
      if (id < 0 || id >= prefill.config().vocab) {
        throw std::out_of_range("qwen35: token id out of range");
      }
    }
    prefill.Run(prompt.data(), n, state, hidden);
    if (mtp != nullptr) {
      // Pairs (h[p - 1], prompt[p]) at p = 1 .. n - 1; the pair at n (the
      // first generated token) is the decoder's first draft.
      prefill.FillMtp(hidden, prompt.data() + 1, n - 1, 1, *mtp);
    }
    return n;
  };
}

}  // namespace gufo_npu::qwen35
