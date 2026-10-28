#include "models/qwen36_35b_a3b/npu_prefill.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/burst_pool.hpp"
#include "core/quant.hpp"
#include "core/simd.hpp"
#include "core/vec.hpp"
#include "models/qwen35/moe.hpp"
#include "models/qwen35/prefill_mixer.hpp"
#include "models/qwen36_35b_a3b/expert_step.hpp"
#include "models/qwen36_35b_a3b/npu_experts.hpp"
#include "models/qwen36_35b_a3b/prefill_device.hpp"
#include "npu/program.hpp"

namespace gufo_npu::qwen36_35b_a3b {
namespace {

using npu::Batch;
using npu::Buffer;
using qwen35::Config;
using qwen35::DecodeState;
using qwen35::Floats;
using qwen35::PrefillMixer;
using qwen35::PrefillMixerWeights;
using DeviceMatrix = qwen35::NpuDecoder::DeviceMatrix;
using vec::Dot;
using vec::RmsNorm;

constexpr std::size_t kMixerOut = 4096;  // ssm_out / attn_output inputs
constexpr std::size_t kMaxInRows = 12288;
constexpr std::size_t kDenseOutOffset = BfpBytes(kMaxInRows, kHidden);
static_assert(kDenseOutOffset % 64 == 0, "buffer windows start on 64 bytes");

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

}  // namespace

struct NpuPrefill::Impl {
  struct Layer {
    PrefillMixerWeights mix;
    std::vector<float> attn_norm, post_norm;
    // The routed experts' router rows, then the shared expert's gate.
    std::vector<float> router;
    std::optional<DeviceMatrix> in, out;
  };

  Profile profile;
  Config config;
  const gguf::TensorInfo* embed = nullptr;
  std::vector<Layer> layers;
  // The MTP head: eh_proj, then its block's input projection (keys, values).
  bool has_mtp = false;
  PrefillMixerWeights mtp_mix;
  std::vector<float> mtp_attn_norm, enorm, hnorm, output_norm;
  std::optional<DeviceMatrix> eh_proj, mtp_in;

  PrefillDevice device;

  // Device scratch: bfp16 weights converted per pass, and GEMM operands.
  std::optional<Buffer> dense_b;  // a layer's input, then output projection
  std::optional<Buffer> mtp_b;    // eh_proj, then the MTP block's input
  std::optional<Buffer> a_dense, c_dense;
  int dense_layer = -1;  // whose matrices dense_b holds
  bool mtp_ready = false;

  // CPU workers, spinning for the length of a Run or FillMtp: one per
  // physical core with the caller (as the 2B's prefill).
  BurstPool pool{
      std::max<std::size_t>(std::thread::hardware_concurrency() / 2, 1) - 1};
  std::optional<PrefillMixer> mixer;
  std::unique_ptr<ExpertStep> experts;

  // Host scratch of a pass: the residual, the post-attention norm (fp32 for
  // the router, bf16 for the expert GEMMs), router logits and choices; of a
  // dense block: its normalized input, input projection and mixer output.
  Floats h, x;
  std::vector<std::uint16_t> xb;
  std::vector<float> logits;
  std::vector<int> chosen;
  std::vector<float> weights;
  std::vector<float> shared_scale;
  Floats normed, in, mixed;
  // Tokens per expert (the shared one last) and their weights, of a layer.
  std::vector<std::vector<int>> expert_tokens;
  std::vector<std::vector<float>> expert_weights;

  const std::vector<int>* follow = nullptr;
  int follow_base = 0;  // token index of the current pass in `follow`
  int differences = 0;
  double gap = 0;

  Impl(const gguf::File& file, qwen35::NpuDecoder& decoder,
       const std::filesystem::path& kernel_dir)
      : config(decoder.config()), device(kernel_dir, profile) {
    const Config& c = config;
    if (!c.IsMoe() || static_cast<std::size_t>(c.hidden) != kHidden ||
        static_cast<std::size_t>(c.expert_ffn) != kExpertFfn ||
        static_cast<std::size_t>(c.shared_expert_ffn) != kExpertFfn ||
        static_cast<std::size_t>(c.heads * c.head_dim) != kMixerOut ||
        static_cast<std::size_t>(c.LinearValueDim()) != kMixerOut) {
      throw std::runtime_error(
          "qwen36_35b_a3b prefill: Qwen3.6-35B-A3B's shapes only");
    }
    mixer.emplace(c, pool, static_cast<int>(kMaxRows));
    embed = &file.Tensor("token_embd.weight");
    auto weights_on_device = decoder.device_weights();
    if (weights_on_device.layers.size() != static_cast<std::size_t>(c.layers)) {
      throw std::logic_error("qwen36_35b_a3b prefill: decoder layer count");
    }
    std::vector<Buffer> arenas;
    for (int il = 0; il < c.layers; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      Layer& l = layers.emplace_back();
      l.mix = qwen35::LoadPrefillMixer(file, c, il);
      l.attn_norm = ToFloats(file.Tensor(p + "attn_norm.weight"));
      l.post_norm = ToFloats(file.Tensor(p + "post_attention_norm.weight"));
      l.router = ToFloats(file.Tensor(p + "ffn_gate_inp.weight"));
      const auto shared =
          ToFloats(file.Tensor(p + "ffn_gate_inp_shexp.weight"));
      l.router.insert(l.router.end(), shared.begin(), shared.end());
      auto& w = weights_on_device.layers[il];
      if (!w.arena || !w.in || !w.out || w.in->dim != kHidden ||
          w.in->rows > kMaxInRows || w.out->dim != kMixerOut ||
          w.out->rows != kHidden) {
        throw std::runtime_error("qwen36_35b_a3b prefill: unexpected layer " +
                                 std::to_string(il) + " on the device");
      }
      l.in = w.in;
      l.out = w.out;
      arenas.push_back(*w.arena);
    }
    if (c.mtp_layers > 0 && weights_on_device.eh_proj &&
        weights_on_device.mtp_in) {
      has_mtp = true;
      const std::string p = "blk." + std::to_string(c.layers) + ".";
      mtp_mix = qwen35::LoadPrefillMixer(file, c, c.layers);
      mtp_attn_norm = ToFloats(file.Tensor(p + "attn_norm.weight"));
      enorm = ToFloats(file.Tensor(p + "nextn.enorm.weight"));
      hnorm = ToFloats(file.Tensor(p + "nextn.hnorm.weight"));
      output_norm = ToFloats(file.Tensor("output_norm.weight"));
      eh_proj = weights_on_device.eh_proj;
      mtp_in = weights_on_device.mtp_in;
      mtp_b.emplace(device.gemm, BfpBytes(eh_proj->rows, eh_proj->dim) +
                                     BfpBytes(mtp_in->rows, mtp_in->dim));
    }

    const std::size_t arena_experts = static_cast<std::size_t>(c.experts) + 1;
    experts = MakeNpuExperts(device, pool, std::move(arenas), arena_experts,
                             static_cast<std::size_t>(kMaxPromptChunk));
    dense_b.emplace(device.gemm,
                    kDenseOutOffset + BfpBytes(kHidden, kMixerOut));
    a_dense.emplace(device.gemm, kMaxRows * kMixerOut * sizeof(std::uint16_t));
    c_dense.emplace(device.gemm, kMaxRows * kMaxInRows * sizeof(std::uint16_t));
    normed.resize(kMaxRows * kHidden);
    in.resize(kMaxRows * kMaxInRows);
    mixed.resize(kMaxRows * kMixerOut);
    expert_tokens.resize(arena_experts);
    expert_weights.resize(arena_experts);
  }

  // Dequant calls converting `m` (q4nx) into `out` at `offset`.
  void AddDequant(Batch& batch, const DeviceMatrix& m, Buffer& out,
                  std::size_t offset) {
    const Buffer& window = device.Window(out, offset, BfpBytes(m.rows, m.dim));
    batch.Add(DequantSequence(m.dim, m.rows), {&m.weights, &window});
  }

  void AddDense(Batch& batch, int il) {
    AddDequant(batch, *layers[il].in, *dense_b, 0);
    AddDequant(batch, *layers[il].out, *dense_b, kDenseOutOffset);
  }

  void AddMtp(Batch& batch) {
    AddDequant(batch, *eh_proj, *mtp_b, 0);
    AddDequant(batch, *mtp_in, *mtp_b, BfpBytes(eh_proj->rows, eh_proj->dim));
  }

  // One GEMM call: C (n of m rows) = A W^T, A n rows of k bf16 already in
  // `a`, W `rows` x k bfp16 at `b_offset` of `b`; syncs A and C.
  void Gemm(std::size_t n, std::size_t k, std::size_t rows, Buffer& a,
            Buffer& b, std::size_t b_offset, Buffer& c) {
    const std::size_t m = PaddedRows(n);
    device.Sync(a, n * k * sizeof(std::uint16_t), true);
    const auto setup = Clock::now();
    Batch batch(device.gemm);
    const Buffer& aw = device.Window(a, 0, m * k * sizeof(std::uint16_t));
    const Buffer& bw = device.Window(b, b_offset, BfpBytes(rows, k));
    const Buffer& cw = device.Window(c, 0, m * rows * sizeof(std::uint16_t));
    batch.Add(GemmSequence(m, k, rows), {&aw, &bw, &cw});
    profile.setup_seconds += SecondsSince(setup);
    device.Run(batch, device.gemm);
    device.Sync(c, n * rows * sizeof(std::uint16_t), false);
  }

  // A row t = bf16(RMS(x_t) * w) for n rows of hidden values; with `keep`,
  // also the fp32 result.
  void NormToA(const float* x_rows, int n, const std::vector<float>& w,
               float* keep) {
    const int d = config.hidden;
    auto* bits = a_dense->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
      const float* row = x_rows + t * d;
      const float inv =
          1.0f /
          std::sqrt(Dot(row, row, d) / static_cast<float>(d) + config.rms_eps);
      simd::ScaleToBf16Row(row, inv, w.data(), bits + t * d, d);
      if (keep != nullptr) {
        RmsNorm(row, w.data(), d, config.rms_eps, keep + t * d);
      }
    });
  }

  void ToA(const float* x_rows, int n, int width) {
    auto* bits = a_dense->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
      simd::ToBf16Row(x_rows + t * width, bits + t * width, width);
    });
  }

  void FromC(int n, int width, float* out, bool add) {
    const auto* bits = c_dense->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
      simd::FromBf16Row(bits + t * width, out + t * width, width, add);
    });
  }

  void CheckTokens(const int* tokens, int count) const {
    for (int t = 0; t < count; ++t) {
      if (tokens[t] < 0 || tokens[t] >= config.vocab) {
        throw std::out_of_range("qwen35: token id out of range");
      }
    }
  }

  // Step 1: the input projection, the mixer and the output projection over
  // the pass's tokens, in blocks of up to kMaxRows.
  void Mixers(int il, int count, DecodeState& state) {
    const Layer& l = layers[il];
    const int d = config.hidden;
    const std::size_t in_rows = l.in->rows;
    auto& st = state.layers[il];
    for (int s = 0; s < count; s += static_cast<int>(kMaxRows)) {
      const int n = std::min(count - s, static_cast<int>(kMaxRows));
      float* hs = h.data() + static_cast<std::size_t>(s) * d;
      auto start = Clock::now();
      NormToA(hs, n, l.attn_norm, l.mix.attention ? nullptr : normed.data());
      profile.other_seconds += SecondsSince(start);
      Gemm(n, kHidden, in_rows, *a_dense, *dense_b, 0, *c_dense);
      start = Clock::now();
      FromC(n, static_cast<int>(in_rows), in.data(), false);
      profile.other_seconds += SecondsSince(start);
      start = Clock::now();
      mixer->SetChunk(n, state.position + s);
      if (l.mix.attention) {
        mixer->Attention(in.data(), in_rows, l.mix, st, mixed.data());
        profile.attention_seconds += SecondsSince(start);
      } else {
        mixer->DeltaNet(in.data(), in_rows, normed.data(), l.mix, st,
                        mixed.data());
        profile.deltanet_seconds += SecondsSince(start);
      }
      start = Clock::now();
      ToA(mixed.data(), n, static_cast<int>(kMixerOut));
      profile.other_seconds += SecondsSince(start);
      Gemm(n, kMixerOut, kHidden, *a_dense, *dense_b, kDenseOutOffset,
           *c_dense);
      start = Clock::now();
      FromC(n, d, hs, true);
      profile.other_seconds += SecondsSince(start);
    }
  }

  // Step 2: the post-attention norm and the routing of every token.
  void Route(int il, int count) {
    const auto start = Clock::now();
    const Layer& l = layers[il];
    const Config& c = config;
    const int d = c.hidden;
    const int k = c.experts_used;
    const int experts = c.experts;
    const int cols = experts + 1;
    pool.Run(static_cast<std::size_t>(count), [&](std::size_t t) {
      float* xt = x.data() + t * d;
      RmsNorm(h.data() + t * d, l.post_norm.data(), d, c.rms_eps, xt);
      simd::ToBf16Row(xt, xb.data() + t * d, d);
      float* y = logits.data() + t * cols;
      for (int e = 0; e < cols; ++e) {
        y[e] = Dot(l.router.data() + static_cast<std::size_t>(e) * d, xt, d);
      }
    });
    for (int t = 0; t < count; ++t) {
      float* y = logits.data() + static_cast<std::size_t>(t) * cols;
      int* ids = chosen.data() + static_cast<std::size_t>(t) * k;
      float* w = weights.data() + static_cast<std::size_t>(t) * k;
      qwen35::Route(y, experts, k, ids, w);
      if (follow != nullptr) {
        Follow(il, t, y, ids, w);
      }
      shared_scale[t] = qwen35::SharedExpertScale(y[experts]);
    }
    for (int e = 0; e <= experts; ++e) {
      expert_tokens[e].clear();
      expert_weights[e].clear();
    }
    for (int t = 0; t < count; ++t) {
      for (int i = 0; i < k; ++i) {
        const auto j = static_cast<std::size_t>(t) * k + i;
        expert_tokens[chosen[j]].push_back(t);
        expert_weights[chosen[j]].push_back(weights[j]);
      }
      expert_tokens[experts].push_back(t);
      expert_weights[experts].push_back(shared_scale[t]);
    }
    profile.router_seconds += SecondsSince(start);
  }

  // Routes token t of layer il to the followed experts instead (weights from
  // its own logits, renormalized over them), keeping the near-tie record.
  void Follow(int il, int t, float* y, int* ids, float* w) {
    const Config& c = config;
    const int k = c.experts_used;
    const auto at = (static_cast<std::size_t>(follow_base + t) * c.layers +
                     static_cast<std::size_t>(il)) *
                    k;
    if (at + k > follow->size()) {
      throw std::logic_error(
          "qwen36_35b_a3b prefill: routing to follow is "
          "shorter than the prompt");
    }
    std::vector<int> want(
        follow->begin() + static_cast<std::ptrdiff_t>(at),
        follow->begin() + static_cast<std::ptrdiff_t>(at + k));
    std::vector<int> own(ids, ids + k);
    std::sort(want.begin(), want.end());
    std::sort(own.begin(), own.end());
    if (own == want) {
      return;
    }
    ++differences;
    float lowest_want = 1e30f;
    float highest_own = -1e30f;
    for (const int e : want) {
      if (!std::binary_search(own.begin(), own.end(), e)) {
        lowest_want = std::min(lowest_want, y[e]);
      }
    }
    for (const int e : own) {
      if (!std::binary_search(want.begin(), want.end(), e)) {
        highest_own = std::max(highest_own, y[e]);
      }
    }
    gap = std::max(
        gap, 1.0 - std::exp(static_cast<double>(lowest_want) - highest_own));
    float floor = 1e30f;
    for (const int e : want) {
      floor = std::min(floor, y[e]);
    }
    std::vector<float> masked(y, y + c.experts);
    for (int e = 0; e < c.experts; ++e) {
      if (!std::binary_search(want.begin(), want.end(), e)) {
        masked[e] = std::min(masked[e], floor - 64.0f);
      }
    }
    qwen35::Route(masked.data(), c.experts, k, ids, w);
  }

  // Step 3: whatever the expert step converts for layer il, and the next
  // matrices the GEMMs need: the next layer's dense ones, or after the last
  // layer the first layer's (for the next pass) and the MTP head's (once),
  // in one visit to the dequant program.
  void Convert(int il, const ExpertRouting& routing) {
    Batch batch(device.dequant);
    experts->Prepare(il, routing, batch);
    const auto setup = Clock::now();
    const int next = il + 1 < config.layers ? il + 1 : 0;
    AddDense(batch, next);
    if (next == 0 && has_mtp && !mtp_ready) {
      AddMtp(batch);
      mtp_ready = true;
    }
    profile.setup_seconds += SecondsSince(setup);
    device.Run(batch, device.dequant);
    dense_layer = next;
  }

  // One pass: `count` tokens (up to kMaxPromptChunk) through every layer.
  void Pass(const int* tokens, int count, DecodeState& state, float* hidden) {
    const Config& c = config;
    const int d = c.hidden;
    const auto start = Clock::now();
    auto t0 = Clock::now();
    pool.Run(static_cast<std::size_t>(count), [&](std::size_t t) {
      quant::Dequantize(embed->type,
                        embed->data + static_cast<std::size_t>(tokens[t]) *
                                          quant::RowBytes(embed->type, d),
                        h.data() + t * d, d);
    });
    profile.other_seconds += SecondsSince(t0);
    if (dense_layer != 0) {
      Batch batch(device.dequant);
      AddDense(batch, 0);
      device.Run(batch, device.dequant);
      dense_layer = 0;
    }
    const ExpertRouting routing{expert_tokens, expert_weights};
    for (int il = 0; il < c.layers; ++il) {
      Mixers(il, count, state);
      Route(il, count);
      Convert(il, routing);
      // Step 4: the routed experts and the shared one, into h.
      experts->Run(il, routing, xb.data(), h.data());
    }
    state.position += count;
    if (hidden != nullptr) {
      std::copy_n(h.data(), static_cast<std::size_t>(count) * d, hidden);
    }
    ++profile.passes;
    (void)start;
  }

  void Run(const int* tokens, int count, DecodeState& state, float* hidden) {
    const Config& c = config;
    const int d = c.hidden;
    CheckTokens(tokens, count);
    differences = 0;
    gap = 0;
    const int largest = std::min(count, kMaxPromptChunk);
    const auto rows = static_cast<std::size_t>(largest);
    h.resize(std::max(h.size(), rows * d));
    x.resize(std::max(x.size(), rows * d));
    xb.resize(std::max(xb.size(), rows * d));
    logits.resize(std::max(logits.size(), rows * (c.experts + 1)));
    chosen.resize(std::max(chosen.size(), rows * c.experts_used));
    weights.resize(std::max(weights.size(), rows * c.experts_used));
    shared_scale.resize(std::max(shared_scale.size(), rows));
    for (int done = 0; done < count;) {
      const int n = std::min(count - done, kMaxPromptChunk);
      follow_base = done;
      Pass(tokens + done, n, state,
           hidden != nullptr ? hidden + static_cast<std::size_t>(done) * d
                             : nullptr);
      done += n;
    }
  }

  void FillMtp(const float* hidden, const int* tokens, int count, int position,
               DecodeState& mtp) {
    const Config& c = config;
    const int d = c.hidden;
    if (!mtp_ready) {
      Batch batch(device.dequant);
      AddMtp(batch);
      device.Run(batch, device.dequant);
      mtp_ready = true;
    }
    const std::size_t eh_bytes = BfpBytes(eh_proj->rows, eh_proj->dim);
    for (int s = 0; s < count; s += static_cast<int>(kMaxRows)) {
      const int n = std::min(count - s, static_cast<int>(kMaxRows));
      // [RMS(embed(token)) * enorm ; RMS(RMS(h) * output_norm) * hnorm].
      auto start = Clock::now();
      auto* bits = a_dense->as<std::uint16_t>();
      pool.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
        thread_local std::vector<float> e, y;
        e.resize(d);
        y.resize(d);
        quant::Dequantize(
            embed->type,
            embed->data + static_cast<std::size_t>(tokens[s + t]) *
                              quant::RowBytes(embed->type, d),
            e.data(), d);
        RmsNorm(e.data(), enorm.data(), d, c.rms_eps, y.data());
        std::uint16_t* row = bits + t * 2 * d;
        simd::ToBf16Row(y.data(), row, d);
        RmsNorm(hidden + (static_cast<std::size_t>(s) + t) * d,
                output_norm.data(), d, c.rms_eps, e.data());
        RmsNorm(e.data(), hnorm.data(), d, c.rms_eps, y.data());
        simd::ToBf16Row(y.data(), row + d, d);
      });
      profile.other_seconds += SecondsSince(start);
      Gemm(n, eh_proj->dim, eh_proj->rows, *a_dense, *mtp_b, 0, *c_dense);
      start = Clock::now();
      FromC(n, d, in.data(), false);
      NormToA(in.data(), n, mtp_attn_norm, nullptr);
      profile.other_seconds += SecondsSince(start);
      Gemm(n, mtp_in->dim, mtp_in->rows, *a_dense, *mtp_b, eh_bytes, *c_dense);
      start = Clock::now();
      FromC(n, static_cast<int>(mtp_in->rows), in.data(), false);
      mixer->SetChunk(n, position + s);
      mixer->AppendKv(in.data(), mtp_in->rows, mtp_mix, mtp.layers[c.layers]);
      profile.other_seconds += SecondsSince(start);
    }
  }
};

NpuPrefill::NpuPrefill(const gguf::File& file, qwen35::NpuDecoder& decoder,
                       const std::filesystem::path& kernel_dir)
    : impl_(std::make_unique<Impl>(file, decoder, kernel_dir)) {}

NpuPrefill::~NpuPrefill() = default;

const qwen35::Config& NpuPrefill::config() const {
  return impl_->config;
}

void NpuPrefill::Run(const int* tokens, int count, qwen35::DecodeState& state,
                     float* hidden) {
  const BurstPool::Burst burst(impl_->pool);
  impl_->Run(tokens, count, state, hidden);
}

void NpuPrefill::FillMtp(const float* hidden, const int* tokens, int count,
                         int position, qwen35::DecodeState& mtp) {
  if (!impl_->has_mtp) {
    throw std::runtime_error("qwen35: model has no MTP layer");
  }
  impl_->CheckTokens(tokens, count);
  const BurstPool::Burst burst(impl_->pool);
  impl_->FillMtp(hidden, tokens, count, position, mtp);
}

void NpuPrefill::FollowRouting(const std::vector<int>* experts) {
  impl_->follow = experts;
}

int NpuPrefill::routing_differences() const {
  return impl_->differences;
}

double NpuPrefill::routing_gap() const {
  return impl_->gap;
}

const NpuPrefill::Profile& NpuPrefill::profile() const {
  return impl_->profile;
}

qwen35::PromptPrefill PrefillPrompt(NpuPrefill& prefill) {
  return [&prefill](const std::vector<int>& prompt, qwen35::DecodeState& state,
                    qwen35::DecodeState* mtp, float* hidden) {
    const int k = static_cast<int>(prompt.size()) - 1;
    if (static_cast<int>(prompt.size()) < kMinPrefillPrompt) {
      return 0;
    }
    // Every id, including the last (the MTP cache reads prompt[k]), before
    // any state changes.
    for (const int id : prompt) {
      if (id < 0 || id >= prefill.config().vocab) {
        throw std::out_of_range("qwen35: token id out of range");
      }
    }
    prefill.Run(prompt.data(), k, state, hidden);
    if (mtp != nullptr) {
      // Pairs (h[p - 1], prompt[p]) at p = 1 .. k.
      prefill.FillMtp(hidden, prompt.data() + 1, k, 1, *mtp);
    }
    return k;
  };
}

}  // namespace gufo_npu::qwen36_35b_a3b
