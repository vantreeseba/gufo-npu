#include "models/qwen35_2b/npu_decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>

#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "core/vec.hpp"
#include "models/qwen35/draft_vocab.hpp"
#include "models/qwen35_2b/deltanet.hpp"
#include "models/qwen35_2b/npu/gemv.hpp"
#include "npu/gemv.hpp"

namespace gufo_npu::qwen35_2b {
namespace {

using qwen35::DraftVocab;
using qwen35::HasMtpLayer;
using qwen35::Mixer;
using qwen35::Q4nxFit;
using qwen35::ReadConfig;
using qwen35::Stage;
using qwen35::StageOutputs;
using qwen35::StatePool;

using gufo_npu::npu::Buffer;
using gufo_npu::npu::Chain;
using gufo_npu::npu::Epilogue;
using gufo_npu::npu::Prologue;
using gufo_npu::npu::Q4nxGemv;
using vec::RmsNorm;

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

// The shapes the 2B's decode program supports; throws otherwise.
Config CheckShapes(Config c) {
  if (static_cast<std::size_t>(c.hidden) != npu::kHidden) {
    throw std::runtime_error("npu decoder: hidden size must be " +
                             std::to_string(npu::kHidden));
  }
  if (c.IsMoe()) {
    throw std::runtime_error(
        "npu decoder: Qwen3.5-2B's decoder runs dense models only");
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

q4nx::Matrix PackOne(const gguf::TensorInfo& t, q4nx::Fit fit) {
  return q4nx::Pack(t.type, t.data, t.Rows(), t.RowElements(), fit);
}

// A GEMV input region: token t's slot at t * npu::kSlot fp32 ([H | D] for a
// resnorm input), then the norm weight. Outputs that feed it are written
// from H or D with token stride npu::kSlot (the chain sequences write D).
constexpr std::size_t kH = 0;
constexpr std::size_t kW = npu::kNormOffset;
constexpr std::size_t kHiddenBytes = npu::kHidden * sizeof(float);
constexpr std::size_t kSlotBytes = npu::kSlot * sizeof(float);
// Both tokens' worth of a hidden vector in a slotted region.
constexpr std::size_t kSlottedBytes = kSlotBytes + kHiddenBytes;
constexpr int kMaxTokens = static_cast<int>(npu::kTokens);

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
  // SwiGLU output.
  struct Block {
    bool attention = false;
    // Where the block's regions start in `acts` (npu::kBlock*).
    std::size_t base = 0;
    std::unique_ptr<Q4nxGemv> in, out, ffn, down;
    // out, ffn and down's weights, back to back (a chain's second argument).
    std::unique_ptr<Buffer> ffn_weights;
    std::unique_ptr<Buffer> in_x, in_y, out_x, ffn_x, ffn_y;
    std::deque<Buffer> views;  // sub-buffers wired between calls (stable)
    // Per token count: out, ffn, down, then the next block's in (or the LM
    // head), as one chain sequence.
    std::unique_ptr<Chain> chain[kMaxTokens];
    // The last block's chains without the LM head after them (prompt
    // passes whose logits nobody reads); the residual stays in lm_x.
    std::unique_ptr<Chain> bare_chain[kMaxTokens];
  };

  Config config;
  q4nx::Fit fit;  // Q4nxFit(config)
  Mixer mixer;
  // The DeltaNet layers' step, on the mixer's workers.
  DeltaNetStep deltanet;
  // Reused across generations (AcquireState, PromptHidden).
  StatePool states{[this] { return mixer.NewState(); }};
  std::vector<float> prompt_hidden;
  gufo_npu::npu::GemvRuntime runtime;
  const gguf::TensorInfo* embed = nullptr;
  // Every region the chains read and write, in the chains' layout: the
  // layers' blocks, the LM head's, then the MTP block and its draft head's.
  std::unique_ptr<Buffer> acts;
  std::vector<Block> layers;
  std::unique_ptr<Q4nxGemv> lm;
  std::unique_ptr<Buffer> lm_x, lm_y, lm_r, scratch;
  std::unique_ptr<Chain> first[kMaxTokens];  // layer 0's input projection
  std::unique_ptr<Chain> lm_only;            // the LM head on lm_x alone

  // The MTP head: eh_proj into its block's input, the block, then an LM
  // head over the draft vocabulary (draft_ids, ascending) with
  // shared_head_norm.
  bool has_mtp = false;
  Block mtp;
  std::vector<float> enorm, hnorm, output_norm, shared_head_norm;
  // The MTP block's final residual per token of the last draft
  // (mtp_hidden()).
  std::vector<float> mtp_residual;
  std::unique_ptr<Q4nxGemv> eh_proj, draft_lm;
  std::vector<int> draft_ids;
  std::unique_ptr<Buffer> eh_x, mtp_lm_x, mtp_lm_y;
  std::unique_ptr<Chain> mtp_first[kMaxTokens];  // eh_proj, block in

  std::vector<float> in, normed, logits, final_hidden;
  std::vector<float> one_logits;  // Step(token)'s result
  double npu_seconds = 0;
  long submissions = 0;
  double cpu_seconds = 0;
  NpuDecoder::Timeline timeline;

  // `bytes` of `acts` at `offset`, as a buffer of its own.
  std::unique_ptr<Buffer> Region(std::size_t offset, std::size_t bytes) {
    return std::make_unique<Buffer>(acts->Sub(offset, bytes));
  }

  // Block `il` with its regions at `base` in `acts`.
  Block MakeBlock(const gguf::File& file, int il, std::size_t base) {
    const Config& c = config;
    const auto hidden = static_cast<std::size_t>(c.hidden);
    const std::string p = "blk." + std::to_string(il) + ".";
    auto t = [&](const char* name) -> const gguf::TensorInfo& {
      return file.Tensor(p + name);
    };
    Block b;
    b.attention = c.IsFullAttention(il);
    b.base = base;
    q4nx::Matrix in_w;
    const gguf::TensorInfo* out_t = nullptr;
    if (b.attention) {
      in_w = q4nx::PackRows(
          {SegmentOf(t("attn_q.weight")), SegmentOf(t("attn_k.weight")),
           SegmentOf(t("attn_v.weight"))},
          hidden, fit);
      out_t = &t("attn_output.weight");
    } else {
      in_w = q4nx::PackRows(
          {SegmentOf(t("attn_qkv.weight")), SegmentOf(t("attn_gate.weight"))},
          hidden, fit);
      out_t = &t("ssm_out.weight");
    }
    b.in = std::make_unique<Q4nxGemv>(runtime, in_w, Prologue::kResNorm,
                                      Epilogue::kStore);
    // out, gate + up and down in one buffer, in the chain's order.
    const q4nx::Matrix out_w = PackOne(*out_t, fit);
    const q4nx::Matrix ffn_w = q4nx::PackRows(
        {SegmentOf(t("ffn_gate.weight")), SegmentOf(t("ffn_up.weight"))},
        hidden, fit, q4nx::SwiGluOrder(c.ffn));
    const q4nx::Matrix down_w = PackOne(t("ffn_down.weight"), fit);
    const std::size_t out_bytes = out_w.bytes.size();
    const std::size_t ffn_bytes = ffn_w.bytes.size();
    const std::size_t down_bytes = down_w.bytes.size();
    b.ffn_weights =
        std::make_unique<Buffer>(runtime, out_bytes + ffn_bytes + down_bytes);
    b.out = std::make_unique<Q4nxGemv>(runtime, out_w, Prologue::kPlain,
                                       Epilogue::kStore,
                                       b.ffn_weights->Sub(0, out_bytes));
    b.ffn = std::make_unique<Q4nxGemv>(
        runtime, ffn_w, Prologue::kResNorm, Epilogue::kSwiGlu,
        b.ffn_weights->Sub(out_bytes, ffn_bytes));
    b.down = std::make_unique<Q4nxGemv>(
        runtime, down_w, Prologue::kAct, Epilogue::kStore,
        b.ffn_weights->Sub(out_bytes + ffn_bytes, down_bytes));
    if (b.in->y_bytes() > npu::kInYBytes ||
        b.ffn->y_bytes() != npu::kFfnYBytes) {
      throw std::logic_error(
          "npu decoder: a block's outputs do not fit the "
          "chains' layout");
    }
    b.in_x = Region(base + npu::kBlockInX, npu::kInputBytes);
    b.ffn_x = Region(base + npu::kBlockFfnX, npu::kInputBytes);
    b.in_y = Region(base + npu::kBlockInY, b.in->y_bytes());
    b.out_x = Region(base + npu::kBlockOutX, npu::kInputBytes);
    b.ffn_y = Region(base + npu::kBlockFfnY, b.ffn->y_bytes());
    SetNormWeight(*b.in_x, t("attn_norm.weight"));
    SetNormWeight(*b.ffn_x, t("post_attention_norm.weight"));
    return b;
  }

  // b's chain for `tokens` (one sequence): out -> ffn_x.D, ffn (residual ->
  // the next block's or head's x.H) -> ffn_y, down -> that x's D, then the
  // `next` call (`next_gemv`) on the regions after b in `acts`. With `bare`,
  // the chain goes to bare_chain and stops before the next call (its input
  // region still gets the residual).
  void WireBlock(Block& b, npu::ChainNext next, Q4nxGemv& next_gemv, int tokens,
                 bool bare = false) {
    if (next_gemv.padded_rows() != npu::ChainNextRows(next)) {
      throw std::logic_error(
          "npu decoder: the next call does not match the "
          "chain's");
    }
    const npu::ChainNext kind = bare ? npu::ChainNext::kNone : next;
    Buffer& window = b.views.emplace_back(
        acts->Sub(b.base + npu::kBlockFfnX, npu::ChainWindowBytes(kind)));
    Buffer& next_w = b.views.emplace_back(next_gemv.weights());
    auto& chain = bare ? b.bare_chain[tokens - 1] : b.chain[tokens - 1];
    chain = std::make_unique<Chain>(runtime);
    chain->Add(npu::ChainSequence(kind, static_cast<std::size_t>(tokens)),
               window, *b.ffn_weights, next_w, *scratch);
  }

  // Runs block b's FFN for `count` tokens and the call chained after it (the
  // next block's input projection, or an LM head).
  // With `bare`, the last block's chain without the LM head.
  void SubmitFfn(Block& b, int count, NpuDecoder::Timeline::Submissions& kind,
                 bool bare = false) {
    Submit(bare ? *b.bare_chain[count - 1] : *b.chain[count - 1], kind);
  }

  Impl(const gguf::File& file, const std::filesystem::path& kernel_dir)
      : config(CheckShapes(ReadConfig(file))),
        fit(Q4nxFit(config)),
        mixer(file),
        deltanet(file, config, mixer.pool()),
        runtime(kernel_dir, gufo_npu::npu::Flow::kElf, npu::DecodeDesign()) {
    const Config& c = config;
    const auto hidden = static_cast<std::size_t>(c.hidden);
    embed = &file.Tensor("token_embd.weight");
    scratch = std::make_unique<Buffer>(runtime, npu::kResidualBytes);
    has_mtp = HasMtpLayer(file, c);
    // The arena: the layers' blocks, the LM head's regions right after the
    // last (its chain runs into them), then the MTP block and its draft
    // head's.
    const auto head_bytes = [](npu::ChainNext next) {
      return npu::kHeadY +
             npu::kTokens * npu::ChainNextRows(next) * sizeof(float);
    };
    const std::size_t lm_base =
        static_cast<std::size_t>(c.layers) * npu::kBlockBytes;
    const std::size_t mtp_base = lm_base + head_bytes(npu::ChainNext::kLmHead);
    const std::size_t draft_base = mtp_base + npu::kBlockBytes;
    acts = std::make_unique<Buffer>(
        runtime, has_mtp ? draft_base + head_bytes(npu::ChainNext::kDraftHead)
                         : mtp_base);
    for (int il = 0; il < c.layers; ++il) {
      layers.push_back(
          MakeBlock(file, il, static_cast<std::size_t>(il) * npu::kBlockBytes));
    }
    const auto& head =
        c.tied_embeddings ? *embed : file.Tensor("output.weight");
    lm = std::make_unique<Q4nxGemv>(
        runtime, q4nx::Pack(head.type, head.data, head.Rows(), hidden, fit),
        Prologue::kResNorm, Epilogue::kStore);
    lm_x = Region(lm_base + npu::kHeadX, npu::kInputBytes);
    lm_r = Region(lm_base + npu::kHeadR, npu::kResidualBytes);
    lm_y = Region(lm_base + npu::kHeadY, lm->y_bytes());
    output_norm = ToFloats(file.Tensor("output_norm.weight"));
    SetNormWeight(*lm_x, file.Tensor("output_norm.weight"));
    lm_only = std::make_unique<Chain>(runtime);
    lm_only->Add(*lm, *lm_x, *lm_y, *lm_r, 1);

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
          WireBlock(l, npu::ChainNext::kLmHead, *lm, tokens);
          WireBlock(l, npu::ChainNext::kLmHead, *lm, tokens, true);
        } else {
          Block& n = layers[il + 1];
          WireBlock(l,
                    n.attention ? npu::ChainNext::kAttention
                                : npu::ChainNext::kDeltaNet,
                    *n.in, tokens);
        }
      }
    }

    if (has_mtp) {
      const std::string p = "blk." + std::to_string(c.layers) + ".";
      enorm = ToFloats(file.Tensor(p + "nextn.enorm.weight"));
      hnorm = ToFloats(file.Tensor(p + "nextn.hnorm.weight"));
      mtp = MakeBlock(file, c.layers, mtp_base);
      eh_proj = std::make_unique<Q4nxGemv>(
          runtime, PackOne(file.Tensor(p + "nextn.eh_proj.weight"), fit),
          Prologue::kPlain, Epilogue::kStore);
      eh_x = std::make_unique<Buffer>(runtime, npu::kInputBytes);
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
          runtime, q4nx::PackRows(draft_rows, hidden, fit), Prologue::kResNorm,
          Epilogue::kStore);
      mtp_lm_x = Region(draft_base + npu::kHeadX, npu::kInputBytes);
      mtp_lm_y = Region(draft_base + npu::kHeadY, draft_lm->y_bytes());
      SetNormWeight(*mtp_lm_x,
                    file.Tensor(p + "nextn.shared_head_norm.weight"));
      shared_head_norm =
          ToFloats(file.Tensor(p + "nextn.shared_head_norm.weight"));
      mtp_residual.resize(kMaxTokens * hidden);
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
        WireBlock(mtp, npu::ChainNext::kDraftHead, *draft_lm, tokens);
      }
    }

    in.resize(kMaxTokens * std::max(StageOutputs(c, Stage::kLinearIn, 0),
                                    StageOutputs(c, Stage::kAttentionIn, 3)));
    normed.resize(hidden);
    logits.resize(static_cast<std::size_t>(kMaxTokens) * c.vocab);
    one_logits.resize(c.vocab);
    final_hidden.resize(kMaxTokens * hidden);
  }

  // Whether Step() is running: its host work goes to timeline.in_step.
  bool stepping = false;

  void Submit(Chain& chain, NpuDecoder::Timeline::Submissions& kind) {
    const auto start = std::chrono::steady_clock::now();
    chain.Start();
    const auto submitted = std::chrono::steady_clock::now();
    chain.Wait();
    const double seconds = SecondsSince(start);
    npu_seconds += seconds;
    ++submissions;
    kind.seconds += seconds;
    kind.submit_seconds +=
        std::chrono::duration<double>(submitted - start).count();
    ++kind.count;
  }

  // Host <-> device syncs, timed for the timeline.
  void Synced(NpuDecoder::Timeline::Syncs& syncs, const Buffer& b,
              std::chrono::steady_clock::time_point start) {
    const double seconds = SecondsSince(start);
    timeline.sync_seconds += seconds;
    if (stepping) {
      ++syncs.count;
      syncs.seconds += seconds;
      syncs.bytes += static_cast<double>(b.size());
    }
  }
  void ToDevice(Buffer& b) {
    const auto start = std::chrono::steady_clock::now();
    b.ToDevice();
    Synced(timeline.in_step.to_device, b, start);
  }
  void FromDevice(Buffer& b) {
    const auto start = std::chrono::steady_clock::now();
    b.FromDevice();
    Synced(timeline.in_step.from_device, b, start);
  }
  // Adds the time since `start` to a StepHost field while Step() runs.
  void StepWork(double& seconds, std::chrono::steady_clock::time_point start) {
    if (stepping) {
      seconds += SecondsSince(start);
    }
  }

  // The CPU part of block `b` (blk `il`) for `count` tokens: reads its input
  // projection, runs the mixer, writes the mixer outputs into the out
  // projection's slots. A DeltaNet pair's second token stays speculative
  // (DeltaNetStep::Run).
  void Mix(Block& b, int il, int count, int position, DecodeState::Layer& st) {
    const auto start = std::chrono::steady_clock::now();
    MixTimed(b, il, count, position, st);
    cpu_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
  }

  void MixTimed(Block& b, int il, int count, int position,
                DecodeState::Layer& st) {
    const Config& c = config;
    FromDevice(*b.in_y);
    // The mixer writes up to a whole slot per token: sync both slots.
    Buffer out_x = b.out_x->Sub(0, kMaxTokens * kSlotBytes);
    if (!b.attention) {
      // alpha and beta come from the token's residual, which the input
      // projection wrote to ffn_x.H.
      Buffer h = b.ffn_x->Sub(kH, kSlottedBytes);
      FromDevice(h);
      const auto mix_start = std::chrono::steady_clock::now();
      deltanet.Run(il, count, b.in_y->as<float>(), b.in->y_token_stride(),
                   h.as<float>(), npu::kSlot, st, out_x.as<float>(),
                   npu::kSlot);
      StepWork(timeline.in_step.deltanet_seconds, mix_start);
      ToDevice(out_x);
      return;
    }
    const std::size_t rows = b.in->rows();
    const std::size_t in_stride = StageOutputs(c, Stage::kAttentionIn, il);
    for (int t = 0; t < count; ++t) {
      float* in_t = in.data() + t * in_stride;
      std::memcpy(in_t, b.in_y->as<float>() + t * b.in->y_token_stride(),
                  rows * sizeof(float));
      const auto mix_start = std::chrono::steady_clock::now();
      mixer.Attention(il, in_t, position + t, st,
                      out_x.as<float>() + t * npu::kSlot);
      StepWork(timeline.in_step.attention_seconds, mix_start);
    }
    ToDevice(out_x);
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

DecodeState NpuDecoder::AcquireState() {
  return impl_->states.Acquire();
}

void NpuDecoder::ReleaseState(DecodeState&& state) {
  impl_->states.Release(std::move(state));
}

float* NpuDecoder::PromptHidden(std::size_t floats) {
  auto& h = impl_->prompt_hidden;
  if (h.size() < floats) {
    h.resize(floats);
  }
  std::fill_n(h.begin(), floats, 0.0f);
  return h.data();
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

const NpuDecoder::Timeline& NpuDecoder::timeline() const {
  return impl_->timeline;
}

bool NpuDecoder::can_draft() const {
  return impl_->has_mtp;
}

const float* NpuDecoder::mtp_hidden(int t) {
  Impl& m = *impl_;
  const int d = m.config.hidden;
  m.FromDevice(*m.mtp_lm_x);
  const float* slot = m.mtp_lm_x->as<float>() + t * npu::kSlot;
  float* out = m.mtp_residual.data() + static_cast<std::size_t>(t) * d;
  for (int i = 0; i < d; ++i) {
    out[i] = slot[i] + slot[npu::kHidden + i];
  }
  return out;
}

const float* NpuDecoder::hidden(int t) const {
  return impl_->final_hidden.data() +
         static_cast<std::size_t>(t) * impl_->config.hidden;
}

const float* NpuDecoder::Logits(const float* hidden) {
  Impl& m = *impl_;
  const Config& c = m.config;
  const int d = c.hidden;
  // Token 0's slot of the LM head's input: H = hidden, D = 0, so its input
  // stage normalizes `hidden` itself. The next Step rewrites the region.
  Buffer slot = m.lm_x->Sub(0, kSlotBytes);
  float* x = slot.as<float>();
  std::memcpy(x, hidden, static_cast<std::size_t>(d) * sizeof(float));
  std::fill(x + npu::kHidden, x + npu::kHidden + d, 0.0f);
  m.ToDevice(slot);
  m.Submit(*m.lm_only, m.timeline.head);
  m.FromDevice(*m.lm_y);
  std::memcpy(m.logits.data(), m.lm_y->as<float>(), c.vocab * sizeof(float));
  std::memcpy(m.final_hidden.data(), hidden,
              static_cast<std::size_t>(d) * sizeof(float));
  return m.logits.data();
}

const std::vector<float>& NpuDecoder::Step(int token, DecodeState& state) {
  const float* logits = Step(&token, 1, state);
  std::memcpy(impl_->one_logits.data(), logits,
              impl_->one_logits.size() * sizeof(float));
  return impl_->one_logits;
}

const float* NpuDecoder::Step(const int* tokens, int count, DecodeState& state,
                              bool logits) {
  Impl& m = *impl_;
  const Config& c = m.config;
  const int d = c.hidden;
  if (count < 1 || count > kMaxTokens) {
    throw std::invalid_argument("npu decoder: 1 or 2 tokens per step");
  }
  // Counts the step's wall time however it returns.
  struct StepTimer {
    NpuDecoder::Timeline& timeline;
    bool& stepping;
    std::chrono::steady_clock::time_point start;
    ~StepTimer() {
      stepping = false;
      ++timeline.steps;
      timeline.step_seconds += SecondsSince(start);
    }
  } step_timer{m.timeline, m.stepping, std::chrono::steady_clock::now()};
  m.stepping = true;
  {
    const auto embed_start = std::chrono::steady_clock::now();
    Buffer h = m.layers[0].in_x->Sub(kH, kSlottedBytes);
    for (int t = 0; t < count; ++t) {
      if (tokens[t] < 0 || tokens[t] >= c.vocab) {
        throw std::out_of_range("qwen35: token id out of range");
      }
      quant::Dequantize(m.embed->type,
                        m.embed->data + static_cast<std::size_t>(tokens[t]) *
                                            quant::RowBytes(m.embed->type, d),
                        h.as<float>() + t * npu::kSlot, d);
    }
    m.StepWork(m.timeline.in_step.embed_seconds, embed_start);
    m.ToDevice(h);
  }
  m.Submit(*m.first[count - 1], m.timeline.first);
  for (int il = 0; il < c.layers; ++il) {
    auto& layer = m.layers[il];
    m.Mix(layer, il, count, state.position, state.layers[il]);
    m.SubmitFfn(layer, count, m.timeline.layers, !logits && il + 1 == c.layers);
  }
  state.position += count;

  if (!logits) {
    // The final residual H + D from the LM head's input region, the same
    // fp32 sum its input stage writes to lm_r.
    m.FromDevice(*m.lm_x);
    const auto copy_start = std::chrono::steady_clock::now();
    for (int t = 0; t < count; ++t) {
      const float* slot = m.lm_x->as<float>() + t * npu::kSlot;
      float* out = m.final_hidden.data() + static_cast<std::size_t>(t) * d;
      for (int i = 0; i < d; ++i) {
        out[i] = slot[i] + slot[npu::kHidden + i];
      }
    }
    m.StepWork(m.timeline.in_step.logits_seconds, copy_start);
    return nullptr;
  }
  m.FromDevice(*m.lm_y);
  m.FromDevice(*m.lm_r);
  const auto copy_start = std::chrono::steady_clock::now();
  for (int t = 0; t < count; ++t) {
    std::memcpy(m.logits.data() + static_cast<std::size_t>(t) * c.vocab,
                m.lm_y->as<float>() + t * m.lm->y_token_stride(),
                c.vocab * sizeof(float));
    std::memcpy(m.final_hidden.data() + static_cast<std::size_t>(t) * d,
                m.lm_r->as<float>() + t * npu::kSlot, d * sizeof(float));
  }
  m.StepWork(m.timeline.in_step.logits_seconds, copy_start);
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
                         int count, int position, DecodeState& state,
                         bool from_mtp) {
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
    float* slot = x.as<float>() + t * npu::kSlot;
    quant::Dequantize(m.embed->type,
                      m.embed->data + static_cast<std::size_t>(tokens[t]) *
                                          quant::RowBytes(m.embed->type, d),
                      m.normed.data(), d);
    RmsNorm(m.normed.data(), m.enorm.data(), d, c.rms_eps, slot);
    RmsNorm(hidden[t], (from_mtp ? m.shared_head_norm : m.output_norm).data(),
            d, c.rms_eps, m.normed.data());
    RmsNorm(m.normed.data(), m.hnorm.data(), d, c.rms_eps, slot + d);
  }
  m.ToDevice(x);
  // eh_proj and the block's input projection; the attention appends to the
  // cache.
  m.Submit(*m.mtp_first[count - 1], m.timeline.mtp);
  m.Mix(m.mtp, c.layers, count, position, state.layers[c.layers]);
}

int NpuDecoder::Draft(const float* const* hidden, const int* tokens, int count,
                      int position, DecodeState& state, bool from_mtp) {
  Impl& m = *impl_;
  FillMtp(hidden, tokens, count, position, state, from_mtp);
  m.SubmitFfn(m.mtp, count, m.timeline.mtp);
  m.FromDevice(*m.mtp_lm_y);
  const float* scores =
      m.mtp_lm_y->as<float>() +
      static_cast<std::size_t>(count - 1) * m.draft_lm->y_token_stride();
  const auto best =
      std::max_element(scores, scores + m.draft_ids.size()) - scores;
  return m.draft_ids[static_cast<std::size_t>(best)];
}

std::vector<int> SpeculativeGenerate(
    NpuDecoder& model, const std::vector<int>& prompt, int max_tokens,
    const std::vector<int>& stop, bool draft,
    const std::function<bool(int)>& on_token, SpeculativeStats* stats,
    const PromptPrefill& prefill, const SamplingParams& sampling) {
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
  // The decoder's pooled states, handed back however the generation ends.
  struct Lease {
    NpuDecoder& model;
    DecodeState state;
    ~Lease() {
      try {
        model.ReleaseState(std::move(state));
      } catch (...) {
      }
    }
  };
  Lease main_lease{model, model.AcquireState()};
  Lease mtp_lease{model, model.AcquireState()};
  DecodeState& state = main_lease.state;
  DecodeState& mtp = mtp_lease.state;
  std::vector<int> out;
  Sampler sampler(sampling);
  auto is_stop = [&](int id) {
    return std::find(stop.begin(), stop.end(), id) != stop.end();
  };
  // Emits a decided token; false once generation is over.
  auto emit = [&](int id) {
    out.push_back(id);
    sampler.Accept(id);
    const bool more = on_token ? on_token(id) : true;
    return more && !is_stop(id) && static_cast<int>(out.size()) < max_tokens;
  };

  // Prompt: `prefill`'s part, then two tokens per pass (both kept).
  // hidden[p]: final residual after the token at p, for the MTP cache.
  auto start = std::chrono::steady_clock::now();
  const int n = static_cast<int>(prompt.size());
  float* hidden = model.PromptHidden(static_cast<std::size_t>(n) * d);
  const float* logits = nullptr;
  const int prefilled =
      prefill ? prefill(prompt, state, draft ? &mtp : nullptr, hidden) : 0;
  if (prefilled < 0 || prefilled > n) {
    throw std::logic_error("generate: prompt prefill ran " +
                           std::to_string(prefilled) + " of " +
                           std::to_string(n) + " tokens");
  }
  // A prefill of the whole prompt leaves only the LM head on its last row:
  // its logits start the output.
  if (prefilled == n) {
    logits = model.Logits(hidden + static_cast<std::size_t>(n - 1) * d);
  }
  for (int p = prefilled; p < n; p += 2) {
    const int count = std::min(2, n - p);
    // Only the last pass's logits are read: the others skip the LM head.
    const bool last = p + count == n;
    logits = model.Step(&prompt[p], count, state, last);
    if (count == 2) {
      model.Resolve(state, true);
    }
    for (int t = 0; t < count; ++t) {
      std::copy_n(model.hidden(t), d, hidden + (p + t) * d);
    }
    if (last) {
      logits += static_cast<std::size_t>(count - 1) * vocab;
    }
  }
  int y1 = sampler.Sample(logits, vocab);
  int y2 = -1;
  if (draft) {
    // The MTP cache over the prompt: pair (h[p - 1], token p) at p.
    for (int p = prefilled + 1; p < n; p += 2) {
      const int count = std::min(2, n - p);
      const float* hs[2] = {hidden + (p - 1) * d, hidden + p * d};
      model.FillMtp(hs, &prompt[p], count, p, mtp);
    }
    const float* hs[1] = {hidden + (n - 1) * d};
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
      y1 = sampler.Sample(model.Step(&y1, 1, state), vocab);
      ++local.steps;
      going = emit(y1);
      continue;
    }
    const int pair[2] = {y1, y2};
    const float* both = model.Step(pair, 2, state);
    ++local.steps;
    // z1 from the main model's distribution; it equals the draft with
    // probability p(y2), which is the draft's acceptance (see the header).
    const int z1 = sampler.Sample(both, vocab);
    std::copy_n(model.hidden(0), d, h1.data());
    std::copy_n(model.hidden(1), d, h2.data());
    if (z1 == y2) {
      model.Resolve(state, true);
      ++local.accepted;
      going = emit(y2);
      if (!going) {
        break;
      }
      // The token after the draft, with the draft in the penalties' history.
      const int z2 = sampler.Sample(both + vocab, vocab);
      going = emit(z2);
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

}  // namespace gufo_npu::qwen35_2b
