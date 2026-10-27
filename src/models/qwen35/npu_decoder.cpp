#include "models/qwen35/npu_decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>

#include "core/cpus.hpp"
#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "core/thread_pool.hpp"
#include "core/vec.hpp"
#include "models/qwen35/draft_vocab.hpp"
#include "models/qwen35/moe.hpp"
#include "npu/gemv.hpp"

namespace gufo_npu::qwen35 {
namespace {

using npu::Buffer;
using npu::Chain;
using npu::Epilogue;
using npu::ExpertArena;
using npu::Prologue;
using npu::Q4nxGather;
using npu::Q4nxGemv;
using vec::Dot;
using vec::RmsNorm;

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

// `rows` rows of `t` from row `first` (an expert of a stacked 3D tensor).
q4nx::Segment RowsOf(const gguf::TensorInfo& t, std::size_t first,
                     std::size_t rows) {
  return {t.type, t.data + first * quant::RowBytes(t.type, t.RowElements()),
          rows};
}

// The shapes the NPU path supports; throws otherwise.
Config CheckShapes(Config c) {
  if (static_cast<std::size_t>(c.hidden) != npu::kGemvHidden) {
    throw std::runtime_error("npu decoder: hidden size must be " +
                             std::to_string(npu::kGemvHidden));
  }
  if (c.IsMoe() &&
      (static_cast<std::size_t>(c.experts_used) + 1 != npu::kGatherExperts ||
       static_cast<std::size_t>(c.expert_ffn) != npu::kExpertFfn ||
       static_cast<std::size_t>(c.shared_expert_ffn) != npu::kExpertFfn)) {
    throw std::runtime_error("npu decoder: MoE layers need " +
                             std::to_string(npu::kGatherExperts - 1) +
                             " routed experts and a shared expert of width " +
                             std::to_string(npu::kExpertFfn) +
                             " (the gather design)");
  }
  return c;
}

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

q4nx::Segment SegmentOf(const gguf::TensorInfo& t) {
  return {t.type, t.data, static_cast<std::size_t>(t.Rows())};
}

q4nx::Matrix PackOne(const gguf::TensorInfo& t) {
  return q4nx::Pack(t.type, t.data, t.Rows(), t.RowElements());
}

// A GEMV input region: token t's slot at t * kGemvSlot fp32 ([H | D] for a
// resnorm input), then the norm weight. Outputs that feed it are written
// from H or D with token stride kGemvSlot.
constexpr std::size_t kH = 0;
constexpr std::size_t kD = npu::kGemvHidden * sizeof(float);
constexpr std::size_t kW = npu::kGemvNormOffset;
constexpr std::size_t kHiddenBytes = npu::kGemvHidden * sizeof(float);
constexpr std::size_t kSlotBytes = npu::kGemvSlot * sizeof(float);
// Both tokens' worth of a hidden vector in a slotted region.
constexpr std::size_t kSlottedBytes = kSlotBytes + kHiddenBytes;
constexpr int kMaxTokens = static_cast<int>(npu::kGemvTokens);

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
  // One transformer block's GEMVs and the regions between them. ffn_x is the
  // FFN's resnorm input (H: the residual, D: the mixer output) and ffn_y its
  // SwiGLU output, dense or MoE.
  struct Block {
    bool attention = false;
    std::unique_ptr<Q4nxGemv> in, out, ffn, down;  // ffn, down: dense
    std::unique_ptr<Buffer> in_x, in_y, out_x, ffn_x, ffn_y;
    std::deque<Buffer> views;      // sub-buffers wired between calls (stable)
    std::vector<float> attn_norm;  // fp32, for alpha/beta on the CPU
    std::vector<float> alpha, beta;
    // MoE: the routed experts and then the shared one in one arena, and for
    // the CPU router post_attention_norm and the router rows (experts, then
    // the shared expert's gate) in fp32.
    std::unique_ptr<Buffer> arena;
    std::vector<float> post_norm;
    std::vector<float> router;
    std::vector<int> chosen;  // the last step's first token's experts
    // Per token count. Dense: out, ffn, down, then the next block's in (or
    // the LM head). MoE: mixer_out runs out alone; after routing,
    // moe[n - kGatherExperts] runs the two gathers over n experts, then the
    // next block's in.
    std::unique_ptr<Chain> chain[kMaxTokens];
    std::unique_ptr<Chain> mixer_out[kMaxTokens];
    std::vector<std::unique_ptr<Chain>> moe[kMaxTokens];
  };

  // MoE gathers per token count and expert count (n - kGatherExperts): one
  // token reads its 8 experts and the shared one; two read their union (8
  // to 16) and the shared one. Every block's chains use the same gathers,
  // which routing re-points just before each submission.
  struct Gathers {
    std::unique_ptr<Q4nxGather> gate_up, down;
  };
  std::vector<Gathers> gathers[kMaxTokens];

  Config config;
  Mixer mixer;
  npu::GemvRuntime runtime;
  // MoE routing: per token the router logits (experts, then the shared
  // gate), its normed input, and its experts and weights; the union of the
  // step's experts and their per-token scales.
  SpinPool pool{std::min<std::size_t>(7, HalfCpuWorkers())};
  std::vector<float> router_logits[kMaxTokens], router_x[kMaxTokens];
  std::vector<int> route_ids[kMaxTokens];
  std::vector<float> route_weights[kMaxTokens];
  std::vector<int> union_ids;
  std::vector<float> union_scales;
  double router_seconds = 0;
  const gguf::TensorInfo* embed = nullptr;
  std::vector<Block> layers;
  std::unique_ptr<Q4nxGemv> lm;
  std::unique_ptr<Buffer> lm_x, lm_y, lm_r, scratch;
  std::unique_ptr<Chain> first[kMaxTokens];  // layer 0's input projection

  // The MTP head: eh_proj into its block's input, the block, then an LM
  // head over the draft vocabulary (draft_ids, ascending) with
  // shared_head_norm.
  bool has_mtp = false;
  Block mtp;
  std::vector<float> enorm, hnorm, output_norm;
  std::unique_ptr<Q4nxGemv> eh_proj, draft_lm;
  std::vector<int> draft_ids;
  std::unique_ptr<Buffer> eh_x, mtp_lm_x, mtp_lm_y;
  std::unique_ptr<Chain> mtp_first[kMaxTokens];  // eh_proj, block in

  std::vector<float> in, normed, logits, final_hidden;
  std::vector<float> one_logits;  // Step(token)'s result
  double npu_seconds = 0;
  long submissions = 0;
  double cpu_seconds = 0;

  Block MakeBlock(const gguf::File& file, int il) {
    const Config& c = config;
    const auto hidden = static_cast<std::size_t>(c.hidden);
    const std::string p = "blk." + std::to_string(il) + ".";
    auto t = [&](const char* name) -> const gguf::TensorInfo& {
      return file.Tensor(p + name);
    };
    Block b;
    b.attention = c.IsFullAttention(il);
    q4nx::Matrix in_w;
    const gguf::TensorInfo* out_t = nullptr;
    if (b.attention) {
      in_w = q4nx::PackRows(
          {SegmentOf(t("attn_q.weight")), SegmentOf(t("attn_k.weight")),
           SegmentOf(t("attn_v.weight"))},
          hidden);
      out_t = &t("attn_output.weight");
    } else {
      in_w = q4nx::PackRows(
          {SegmentOf(t("attn_qkv.weight")), SegmentOf(t("attn_gate.weight"))},
          hidden);
      b.alpha = ToFloats(t("ssm_alpha.weight"));
      b.beta = ToFloats(t("ssm_beta.weight"));
      out_t = &t("ssm_out.weight");
    }
    b.in = std::make_unique<Q4nxGemv>(runtime, in_w, Prologue::kResNorm,
                                      Epilogue::kStore);
    b.out = std::make_unique<Q4nxGemv>(runtime, PackOne(*out_t),
                                       Prologue::kPlain, Epilogue::kStore);
    std::size_t ffn_y_bytes = 0;
    if (c.IsMoe()) {
      MakeExperts(b, file, p);
      ffn_y_bytes = gathers[0].front().gate_up->y_bytes();
    } else {
      b.ffn = std::make_unique<Q4nxGemv>(
          runtime,
          q4nx::PackRows(
              {SegmentOf(t("ffn_gate.weight")), SegmentOf(t("ffn_up.weight"))},
              hidden, q4nx::SwiGluOrder(c.ffn)),
          Prologue::kResNorm, Epilogue::kSwiGlu);
      b.down =
          std::make_unique<Q4nxGemv>(runtime, PackOne(t("ffn_down.weight")),
                                     Prologue::kAct, Epilogue::kStore);
      ffn_y_bytes = b.ffn->y_bytes();
    }
    b.in_x = std::make_unique<Buffer>(runtime, npu::kGemvInputBytes);
    b.in_y = std::make_unique<Buffer>(runtime, b.in->y_bytes());
    b.out_x = std::make_unique<Buffer>(runtime, npu::kGemvInputBytes);
    b.ffn_x = std::make_unique<Buffer>(runtime, npu::kGemvInputBytes);
    b.ffn_y = std::make_unique<Buffer>(runtime, ffn_y_bytes);
    SetNormWeight(*b.in_x, t("attn_norm.weight"));
    SetNormWeight(*b.ffn_x, t("post_attention_norm.weight"));
    b.attn_norm = ToFloats(t("attn_norm.weight"));
    return b;
  }

  // An MoE block's expert arena (every routed expert, then the shared one)
  // and the router for the CPU.
  void MakeExperts(Block& b, const gguf::File& file, const std::string& p) {
    const Config& c = config;
    const auto hidden = static_cast<std::size_t>(c.hidden);
    const auto ef = static_cast<std::size_t>(c.expert_ffn);
    auto t = [&](const char* name) -> const gguf::TensorInfo& {
      return file.Tensor(p + name);
    };
    const ExpertArena arena{static_cast<std::size_t>(c.experts) + 1};
    b.arena = std::make_unique<Buffer>(runtime, arena.bytes());
    auto* base = b.arena->as<std::uint8_t>();
    const auto order = q4nx::SwiGluOrder(ef);
    auto put = [&](std::size_t e, const q4nx::Matrix& up,
                   const q4nx::Matrix& down) {
      if (up.bytes.size() != ExpertArena::kUpBytes ||
          down.bytes.size() != ExpertArena::kDownBytes) {
        throw std::logic_error("npu decoder: expert packs to the wrong size");
      }
      std::memcpy(base + arena.UpOffset(e), up.bytes.data(), up.bytes.size());
      std::memcpy(base + arena.DownOffset(e), down.bytes.data(),
                  down.bytes.size());
    };
    const auto& gate = t("ffn_gate_exps.weight");
    const auto& up = t("ffn_up_exps.weight");
    const auto& down = t("ffn_down_exps.weight");
    for (std::size_t e = 0; e < static_cast<std::size_t>(c.experts); ++e) {
      put(e,
          q4nx::PackRows({RowsOf(gate, e * ef, ef), RowsOf(up, e * ef, ef)},
                         hidden, order),
          q4nx::PackRows({RowsOf(down, e * hidden, hidden)}, ef));
    }
    put(static_cast<std::size_t>(c.experts),
        q4nx::PackRows({SegmentOf(t("ffn_gate_shexp.weight")),
                        SegmentOf(t("ffn_up_shexp.weight"))},
                       hidden, order),
        PackOne(t("ffn_down_shexp.weight")));
    b.arena->ToDevice();
    b.post_norm = ToFloats(t("post_attention_norm.weight"));
    b.router = ToFloats(t("ffn_gate_inp.weight"));
    const auto shared_gate = ToFloats(t("ffn_gate_inp_shexp.weight"));
    if (b.router.size() != static_cast<std::size_t>(c.experts) * hidden ||
        shared_gate.size() != hidden) {
      throw std::runtime_error("npu decoder: unexpected router shape in " + p);
    }
    b.router.insert(b.router.end(), shared_gate.begin(), shared_gate.end());
    b.chosen.assign(c.experts_used, 0);
  }

  // b's chain for `tokens`: out -> ffn_x.D, ffn (residual -> next.H) ->
  // ffn_y, down -> next.D, then `next` with its own buffers. MoE: out alone
  // in mixer_out, and the gathers in place of ffn and down.
  void WireBlock(Block& b, Buffer& next_x, int tokens, Q4nxGemv& next,
                 Buffer& next_y, Buffer& next_r) {
    Buffer& ffn_d = b.views.emplace_back(b.ffn_x->Sub(kD, kSlottedBytes));
    Buffer& next_h = b.views.emplace_back(next_x.Sub(kH, kSlottedBytes));
    Buffer& next_d = b.views.emplace_back(next_x.Sub(kD, kSlottedBytes));
    auto& chain = b.chain[tokens - 1];
    chain = std::make_unique<Chain>(runtime);
    if (b.arena) {
      chain.reset();
      auto& out = b.mixer_out[tokens - 1];
      out = std::make_unique<Chain>(runtime);
      out->Add(*b.out, *b.out_x, ffn_d, *scratch, tokens);
      auto& chains = b.moe[tokens - 1];
      chains.clear();
      for (auto& g : gathers[tokens - 1]) {
        auto& moe = chains.emplace_back(std::make_unique<Chain>(runtime));
        moe->Add(*g.gate_up, *b.ffn_x, *b.arena, *b.ffn_y, next_h);
        moe->Add(*g.down, *b.ffn_y, *b.arena, next_d, *scratch);
        moe->Add(next, next_x, next_y, next_r, tokens);
      }
      return;
    }
    chain->Add(*b.out, *b.out_x, ffn_d, *scratch, tokens);
    chain->Add(*b.ffn, *b.ffn_x, *b.ffn_y, next_h, tokens);
    chain->Add(*b.down, *b.ffn_y, next_d, *scratch, tokens);
    chain->Add(next, next_x, next_y, next_r, tokens);
  }

  // Runs block b's FFN for `count` tokens and the call chained after it (the
  // next block's input projection, or an LM head): the chain, or for MoE the
  // mixer's output projection, routing, then the gathers.
  void SubmitFfn(Block& b, int count) {
    if (!b.arena) {
      Submit(*b.chain[count - 1]);
      return;
    }
    Submit(*b.mixer_out[count - 1]);
    Submit(*b.moe[count - 1][RouteExperts(b, count)]);
  }

  Impl(const gguf::File& file, const std::filesystem::path& kernel_dir)
      : config(CheckShapes(ReadConfig(file))),
        mixer(file),
        // The gathers rewrite their instruction streams per call. Dense
        // models keep the ELF flow, ~2% faster on Qwen3.5-2B decode.
        runtime(kernel_dir, config.IsMoe() ? npu::Flow::kInstructionBuffer
                                           : npu::Flow::kElf) {
    const Config& c = config;
    const auto hidden = static_cast<std::size_t>(c.hidden);
    embed = &file.Tensor("token_embd.weight");
    scratch = std::make_unique<Buffer>(runtime, npu::kGemvResidualBytes);
    if (c.IsMoe()) {
      const ExpertArena arena{static_cast<std::size_t>(c.experts) + 1};
      for (std::size_t tokens = 1; tokens <= npu::kGemvTokens; ++tokens) {
        const std::size_t most =
            tokens == 1 ? npu::kGatherExperts : npu::kMaxGatherExperts;
        for (std::size_t n = npu::kGatherExperts; n <= most; ++n) {
          gathers[tokens - 1].push_back(
              {std::make_unique<Q4nxGather>(runtime, Q4nxGather::Kind::kGateUp,
                                            arena, n, tokens),
               std::make_unique<Q4nxGather>(runtime, Q4nxGather::Kind::kDown,
                                            arena, n, tokens)});
        }
      }
      for (int t = 0; t < kMaxTokens; ++t) {
        router_logits[t].resize(static_cast<std::size_t>(c.experts) + 1);
        router_x[t].resize(hidden);
        route_ids[t].resize(c.experts_used);
        route_weights[t].resize(c.experts_used);
      }
    }
    for (int il = 0; il < c.layers; ++il) {
      layers.push_back(MakeBlock(file, il));
    }
    const auto& head =
        c.tied_embeddings ? *embed : file.Tensor("output.weight");
    lm = std::make_unique<Q4nxGemv>(
        runtime, q4nx::Pack(head.type, head.data, head.Rows(), hidden),
        Prologue::kResNorm, Epilogue::kStore);
    lm_x = std::make_unique<Buffer>(runtime, npu::kGemvInputBytes);
    lm_y = std::make_unique<Buffer>(runtime, lm->y_bytes());
    lm_r = std::make_unique<Buffer>(runtime, npu::kGemvResidualBytes);
    output_norm = ToFloats(file.Tensor("output_norm.weight"));
    SetNormWeight(*lm_x, file.Tensor("output_norm.weight"));

    // Wiring. in_l: residual -> ffn_x.H. out_l: -> ffn_x.D. ffn_l: residual
    // -> next.H. down_l: -> next.D. Layer 0's D stays zero.
    for (int tokens = 1; tokens <= kMaxTokens; ++tokens) {
      Block& l0 = layers[0];
      first[tokens - 1] = std::make_unique<Chain>(runtime);
      first[tokens - 1]->Add(
          *l0.in, *l0.in_x, *l0.in_y,
          l0.views.emplace_back(l0.ffn_x->Sub(kH, kSlottedBytes)), tokens);
      for (int il = 0; il < c.layers; ++il) {
        Block& l = layers[il];
        if (il + 1 == c.layers) {
          WireBlock(l, *lm_x, tokens, *lm, *lm_y, *lm_r);
        } else {
          Block& n = layers[il + 1];
          Buffer& n_res = n.views.emplace_back(n.ffn_x->Sub(kH, kSlottedBytes));
          WireBlock(l, *n.in_x, tokens, *n.in, *n.in_y, n_res);
        }
      }
    }

    if (HasMtpLayer(file, c)) {
      has_mtp = true;
      const std::string p = "blk." + std::to_string(c.layers) + ".";
      enorm = ToFloats(file.Tensor(p + "nextn.enorm.weight"));
      hnorm = ToFloats(file.Tensor(p + "nextn.hnorm.weight"));
      mtp = MakeBlock(file, c.layers);
      eh_proj = std::make_unique<Q4nxGemv>(
          runtime, PackOne(file.Tensor(p + "nextn.eh_proj.weight")),
          Prologue::kPlain, Epilogue::kStore);
      eh_x = std::make_unique<Buffer>(runtime, npu::kGemvInputBytes);
      draft_ids = DraftVocab(c.vocab);
      // The draft vocabulary's rows of the LM head.
      std::vector<q4nx::Segment> draft_rows;
      draft_rows.reserve(draft_ids.size());
      const std::size_t row_bytes = quant::RowBytes(head.type, hidden);
      for (const int id : draft_ids) {
        draft_rows.push_back(
            {head.type, head.data + static_cast<std::size_t>(id) * row_bytes,
             1});
      }
      draft_lm = std::make_unique<Q4nxGemv>(
          runtime, q4nx::PackRows(draft_rows, hidden), Prologue::kResNorm,
          Epilogue::kStore);
      mtp_lm_x = std::make_unique<Buffer>(runtime, npu::kGemvInputBytes);
      mtp_lm_y = std::make_unique<Buffer>(runtime, draft_lm->y_bytes());
      SetNormWeight(*mtp_lm_x,
                    file.Tensor(p + "nextn.shared_head_norm.weight"));
      // eh_proj -> block in_x.H (its D stays zero); in: residual ->
      // ffn_x.H; the block chains into the draft LM head on its own input.
      for (int tokens = 1; tokens <= kMaxTokens; ++tokens) {
        auto& chain = mtp_first[tokens - 1];
        chain = std::make_unique<Chain>(runtime);
        chain->Add(*eh_proj, *eh_x,
                   mtp.views.emplace_back(mtp.in_x->Sub(kH, kSlottedBytes)),
                   *scratch, tokens);
        chain->Add(*mtp.in, *mtp.in_x, *mtp.in_y,
                   mtp.views.emplace_back(mtp.ffn_x->Sub(kH, kSlottedBytes)),
                   tokens);
        WireBlock(mtp, *mtp_lm_x, tokens, *draft_lm, *mtp_lm_y, *scratch);
      }
    }

    in.resize(kMaxTokens * std::max(StageOutputs(c, Stage::kLinearIn, 0),
                                    StageOutputs(c, Stage::kAttentionIn, 3)));
    normed.resize(hidden);
    logits.resize(static_cast<std::size_t>(kMaxTokens) * c.vocab);
    one_logits.resize(c.vocab);
    final_hidden.resize(kMaxTokens * hidden);
  }

  // MoE block b, `count` tokens: per token the router in fp32 on the
  // residual after the mixer (ffn_x: H + D), as the CPU decoder routes; then
  // the gathers pointed at the union of the tokens' experts in ascending id
  // order, and the shared expert last. A token weighs the experts only the
  // other token chose 0, and they add exact zeros, so its result does not
  // depend on what the other token chose. Returns the index of the chain for
  // the union's size (n - kGatherExperts).
  std::size_t RouteExperts(Block& b, int count) {
    const auto start = std::chrono::steady_clock::now();
    const Config& c = config;
    const int d = c.hidden;
    Buffer slots = b.ffn_x->Sub(0, count * kSlotBytes);
    slots.FromDevice();
    for (int t = 0; t < count; ++t) {
      const float* h = slots.as<float>() + t * npu::kGemvSlot;
      auto& x = router_x[t];
      for (int i = 0; i < d; ++i) {
        x[i] = h[i] + h[d + i];
      }
      RmsNorm(x.data(), b.post_norm.data(), d, c.rms_eps, x.data());
    }
    // Router rows, then the shared expert's gate, in chunks across the pool;
    // each row is read once for both tokens.
    constexpr std::size_t kRows = 32;
    const std::size_t rows = router_logits[0].size();
    pool.Run((rows + kRows - 1) / kRows, [&](std::size_t job) {
      const std::size_t end = std::min(rows, (job + 1) * kRows);
      for (std::size_t r = job * kRows; r < end; ++r) {
        for (int t = 0; t < count; ++t) {
          router_logits[t][r] =
              Dot(b.router.data() + r * d, router_x[t].data(), d);
        }
      }
    });
    auto& ids = union_ids;
    ids.clear();
    for (int t = 0; t < count; ++t) {
      Route(router_logits[t].data(), c.experts, c.experts_used,
            route_ids[t].data(), route_weights[t].data());
      ids.insert(ids.end(), route_ids[t].begin(), route_ids[t].end());
    }
    b.chosen = route_ids[0];
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    const auto routed = static_cast<std::ptrdiff_t>(ids.size());
    ids.push_back(c.experts);  // the shared expert's arena slot
    union_scales.assign(ids.size() * count, 0.0f);
    for (int t = 0; t < count; ++t) {
      for (int k = 0; k < c.experts_used; ++k) {
        const auto i = static_cast<std::size_t>(
            std::lower_bound(ids.begin(), ids.begin() + routed,
                             route_ids[t][k]) -
            ids.begin());
        union_scales[i * count + t] = route_weights[t][k];
      }
      union_scales[(ids.size() - 1) * count + t] =
          SharedExpertScale(router_logits[t][c.experts]);
    }
    const std::size_t index = ids.size() - npu::kGatherExperts;
    auto& g = gathers[count - 1].at(index);
    g.gate_up->Select(ids.data(), union_scales.data());
    g.down->Select(ids.data());
    router_seconds += SecondsSince(start);
    return index;
  }

  void Submit(Chain& chain) {
    const auto start = std::chrono::steady_clock::now();
    chain.Run();
    npu_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    ++submissions;
  }

  // The CPU part of block `b` (blk `il`) for `count` tokens: reads its input
  // projection, runs the mixer per token, writes the mixer outputs into the
  // out projection's slots. With `speculative`, the second token's DeltaNet
  // state goes to ssm_spec and the conv window before it to conv_saved.
  void Mix(Block& b, int il, int count, int position, DecodeState::Layer& st,
           bool speculative) {
    const auto start = std::chrono::steady_clock::now();
    MixTimed(b, il, count, position, st, speculative);
    cpu_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
  }

  void MixTimed(Block& b, int il, int count, int position,
                DecodeState::Layer& st, bool speculative) {
    const Config& c = config;
    const int d = c.hidden;
    b.in_y->FromDevice();
    const std::size_t rows = b.in->rows();
    const std::size_t in_stride = StageOutputs(
        c, b.attention ? Stage::kAttentionIn : Stage::kLinearIn, il);
    // The mixer writes up to a whole slot per token (Qwen3.6-35B-A3B's
    // DeltaNet value dim and attention width are 4096): sync both slots.
    Buffer out_x = b.out_x->Sub(0, kMaxTokens * kSlotBytes);
    std::optional<Buffer> h;
    if (!b.attention) {
      h.emplace(b.ffn_x->Sub(kH, kSlottedBytes));
      h->FromDevice();
    }
    for (int t = 0; t < count; ++t) {
      float* in_t = in.data() + t * in_stride;
      std::memcpy(in_t, b.in_y->as<float>() + t * b.in->y_token_stride(),
                  rows * sizeof(float));
      float* y = out_x.as<float>() + t * npu::kGemvSlot;
      if (b.attention) {
        mixer.Attention(il, in_t, position + t, st, y);
        continue;
      }
      // alpha and beta from the token's normalized input: its residual is
      // what the input projection wrote to ffn_x.H.
      const float* hv = h->as<float>() + t * npu::kGemvSlot;
      RmsNorm(hv, b.attn_norm.data(), d, c.rms_eps, normed.data());
      const int heads = c.linear_value_heads;
      float* ab = in_t + rows;
      for (int head = 0; head < 2 * heads; ++head) {
        const float* w = (head < heads ? b.alpha.data() : b.beta.data()) +
                         static_cast<std::size_t>(head % heads) * d;
        ab[head] = Dot(w, normed.data(), d);
      }
      if (speculative && t == 1) {
        st.conv_saved = st.conv;
        st.ssm_spec.resize(st.ssm.size());
        mixer.Linear(il, in_t, st, y, st.ssm_spec.data());
      } else {
        mixer.Linear(il, in_t, st, y);
      }
    }
    out_x.ToDevice();
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

DecodeState NpuDecoder::NewMtpState() const {
  return impl_->mixer.NewState();
}

double NpuDecoder::npu_seconds() const {
  return impl_->npu_seconds;
}

long NpuDecoder::submissions() const {
  return impl_->submissions;
}

double NpuDecoder::cpu_seconds() const {
  return impl_->cpu_seconds;
}

bool NpuDecoder::can_draft() const {
  return impl_->has_mtp;
}

const int* NpuDecoder::experts(int layer) const {
  const auto& chosen = impl_->layers.at(static_cast<std::size_t>(layer)).chosen;
  if (chosen.empty()) {
    throw std::logic_error("npu decoder: not an MoE layer");
  }
  return chosen.data();
}

double NpuDecoder::router_seconds() const {
  return impl_->router_seconds;
}

const float* NpuDecoder::hidden(int t) const {
  return impl_->final_hidden.data() +
         static_cast<std::size_t>(t) * impl_->config.hidden;
}

const std::vector<float>& NpuDecoder::Step(int token, DecodeState& state) {
  const float* logits = Step(&token, 1, state);
  std::memcpy(impl_->one_logits.data(), logits,
              impl_->one_logits.size() * sizeof(float));
  return impl_->one_logits;
}

const float* NpuDecoder::Step(const int* tokens, int count,
                              DecodeState& state) {
  Impl& m = *impl_;
  const Config& c = m.config;
  const int d = c.hidden;
  if (count < 1 || count > kMaxTokens) {
    throw std::invalid_argument("npu decoder: 1 or 2 tokens per step");
  }
  {
    Buffer h = m.layers[0].in_x->Sub(kH, kSlottedBytes);
    for (int t = 0; t < count; ++t) {
      if (tokens[t] < 0 || tokens[t] >= c.vocab) {
        throw std::out_of_range("qwen35: token id out of range");
      }
      quant::Dequantize(m.embed->type,
                        m.embed->data + static_cast<std::size_t>(tokens[t]) *
                                            quant::RowBytes(m.embed->type, d),
                        h.as<float>() + t * npu::kGemvSlot, d);
    }
    h.ToDevice();
  }
  m.Submit(*m.first[count - 1]);
  for (int il = 0; il < c.layers; ++il) {
    auto& layer = m.layers[il];
    m.Mix(layer, il, count, state.position, state.layers[il], count == 2);
    m.SubmitFfn(layer, count);
  }
  state.position += count;

  m.lm_y->FromDevice();
  m.lm_r->FromDevice();
  for (int t = 0; t < count; ++t) {
    std::memcpy(m.logits.data() + static_cast<std::size_t>(t) * c.vocab,
                m.lm_y->as<float>() + t * m.lm->y_token_stride(),
                c.vocab * sizeof(float));
    std::memcpy(m.final_hidden.data() + static_cast<std::size_t>(t) * d,
                m.lm_r->as<float>() + t * npu::kGemvSlot, d * sizeof(float));
  }
  return m.logits.data();
}

void NpuDecoder::Resolve(DecodeState& state, bool keep_second) {
  const Config& c = impl_->config;
  for (int il = 0; il < c.layers; ++il) {
    auto& st = state.layers[il];
    if (c.IsFullAttention(il)) {
      if (!keep_second) {
        const std::size_t kv =
            static_cast<std::size_t>(c.kv_heads) * c.head_dim;
        st.keys.resize(st.keys.size() - kv);
        st.values.resize(st.values.size() - kv);
      }
    } else if (keep_second) {
      st.ssm.swap(st.ssm_spec);
    } else {
      st.conv.swap(st.conv_saved);
    }
  }
  if (!keep_second) {
    --state.position;
  }
}

void NpuDecoder::FillMtp(const float* const* hidden, const int* tokens,
                         int count, int position, DecodeState& state) {
  Impl& m = *impl_;
  const Config& c = m.config;
  const int d = c.hidden;
  if (!m.has_mtp) {
    throw std::runtime_error("qwen35: model has no MTP layer");
  }
  if (count < 1 || count > kMaxTokens) {
    throw std::invalid_argument("npu decoder: 1 or 2 MTP positions per draft");
  }
  // [RMS(embed(token)) * enorm ; RMS(RMS(h) * output_norm) * hnorm] per
  // position, into eh_proj's input slots.
  // eh_proj's input is 2 * hidden per token: both slots in full.
  Buffer x = m.eh_x->Sub(0, kMaxTokens * kSlotBytes);
  for (int t = 0; t < count; ++t) {
    float* slot = x.as<float>() + t * npu::kGemvSlot;
    quant::Dequantize(m.embed->type,
                      m.embed->data + static_cast<std::size_t>(tokens[t]) *
                                          quant::RowBytes(m.embed->type, d),
                      m.normed.data(), d);
    RmsNorm(m.normed.data(), m.enorm.data(), d, c.rms_eps, slot);
    RmsNorm(hidden[t], m.output_norm.data(), d, c.rms_eps, m.normed.data());
    RmsNorm(m.normed.data(), m.hnorm.data(), d, c.rms_eps, slot + d);
  }
  x.ToDevice();
  // eh_proj and the block's input projection; the attention appends to the
  // cache.
  m.Submit(*m.mtp_first[count - 1]);
  m.Mix(m.mtp, c.layers, count, position, state.layers[c.layers], false);
}

int NpuDecoder::Draft(const float* const* hidden, const int* tokens, int count,
                      int position, DecodeState& state) {
  Impl& m = *impl_;
  FillMtp(hidden, tokens, count, position, state);
  m.SubmitFfn(m.mtp, count);
  m.mtp_lm_y->FromDevice();
  const float* scores =
      m.mtp_lm_y->as<float>() +
      static_cast<std::size_t>(count - 1) * m.draft_lm->y_token_stride();
  const auto best =
      std::max_element(scores, scores + m.draft_ids.size()) - scores;
  return m.draft_ids[static_cast<std::size_t>(best)];
}

namespace {

int Argmax(const float* logits, int vocab) {
  return static_cast<int>(std::max_element(logits, logits + vocab) - logits);
}

}  // namespace

std::vector<int> SpeculativeGenerate(NpuDecoder& model,
                                     const std::vector<int>& prompt,
                                     int max_tokens,
                                     const std::vector<int>& stop, bool draft,
                                     const std::function<void(int)>& on_token,
                                     SpeculativeStats* stats,
                                     const PromptPrefill& prefill) {
  if (prompt.empty()) {
    throw std::invalid_argument("generate: empty prompt");
  }
  if (draft && !model.can_draft()) {
    throw std::invalid_argument(
        "generate: this model has no MTP head on the NPU; decode without a "
        "draft");
  }
  const Config& c = model.config();
  const int vocab = c.vocab;
  const auto d = static_cast<std::size_t>(c.hidden);
  SpeculativeStats local;
  local.prompt_tokens = static_cast<int>(prompt.size());
  auto state = model.NewState();
  auto mtp = model.NewMtpState();
  std::vector<int> out;
  auto is_stop = [&](int id) {
    return std::find(stop.begin(), stop.end(), id) != stop.end();
  };
  // Emits a decided token; false once generation is over.
  auto emit = [&](int id) {
    out.push_back(id);
    if (on_token) {
      on_token(id);
    }
    return !is_stop(id) && static_cast<int>(out.size()) < max_tokens;
  };

  // Prompt: `prefill`'s part, then two tokens per pass (both kept).
  // hidden[p]: final residual after the token at p, for the MTP cache.
  auto start = std::chrono::steady_clock::now();
  const int n = static_cast<int>(prompt.size());
  std::vector<float> hidden(static_cast<std::size_t>(n) * d);
  const float* logits = nullptr;
  const int prefilled =
      prefill ? prefill(prompt, state, draft ? &mtp : nullptr, hidden.data())
              : 0;
  // The decoder runs at least the last prompt token: its logits start the
  // output.
  if (prefilled < 0 || prefilled > n - 1) {
    throw std::logic_error("generate: prompt prefill ran " +
                           std::to_string(prefilled) + " of " +
                           std::to_string(n) + " tokens");
  }
  for (int p = prefilled; p < n; p += 2) {
    const int count = std::min(2, n - p);
    logits = model.Step(&prompt[p], count, state);
    if (count == 2) {
      model.Resolve(state, true);
    }
    for (int t = 0; t < count; ++t) {
      std::copy_n(model.hidden(t), d, hidden.data() + (p + t) * d);
    }
    logits += static_cast<std::size_t>(count - 1) * vocab;
  }
  int y1 = Argmax(logits, vocab);
  int y2 = -1;
  if (draft) {
    // The MTP cache over the prompt: pair (h[p - 1], token p) at p.
    for (int p = prefilled + 1; p < n; p += 2) {
      const int count = std::min(2, n - p);
      const float* hs[2] = {hidden.data() + (p - 1) * d, hidden.data() + p * d};
      model.FillMtp(hs, &prompt[p], count, p, mtp);
    }
    const float* hs[1] = {hidden.data() + (n - 1) * d};
    y2 = model.Draft(hs, &y1, 1, n, mtp);
  }
  local.prefill_seconds = SecondsSince(start);

  start = std::chrono::steady_clock::now();
  bool going = emit(y1);
  int position = n;  // of y1
  std::vector<float> h1(d);
  std::vector<float> h2(d);
  while (going) {
    if (!draft) {
      y1 = Argmax(model.Step(&y1, 1, state), vocab);
      ++local.steps;
      going = emit(y1);
      continue;
    }
    const int pair[2] = {y1, y2};
    const float* both = model.Step(pair, 2, state);
    ++local.steps;
    const int z1 = Argmax(both, vocab);
    const int z2 = Argmax(both + vocab, vocab);
    std::copy_n(model.hidden(0), d, h1.data());
    std::copy_n(model.hidden(1), d, h2.data());
    if (z1 == y2) {
      model.Resolve(state, true);
      ++local.accepted;
      going = emit(y2) && emit(z2);
      if (!going) {
        break;
      }
      // Pairs (h after y1, y2) at position + 1 and (h after y2, z2) at
      // position + 2; the second drafts the token after z2.
      const float* hs[2] = {h1.data(), h2.data()};
      const int toks[2] = {y2, z2};
      y2 = model.Draft(hs, toks, 2, position + 1, mtp);
      y1 = z2;
      position += 2;
    } else {
      model.Resolve(state, false);
      going = emit(z1);
      if (!going) {
        break;
      }
      const float* hs[1] = {h1.data()};
      y2 = model.Draft(hs, &z1, 1, position + 1, mtp);
      y1 = z1;
      position += 1;
    }
  }
  local.decode_seconds = SecondsSince(start);
  local.generated_tokens = static_cast<int>(out.size());
  if (stats != nullptr) {
    *stats = local;
  }
  return out;
}

}  // namespace gufo_npu::qwen35
