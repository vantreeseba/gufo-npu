#include "models/qwen35/rope.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/burst_pool.hpp"
#include "core/gguf.hpp"
#include "core/vec.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/gguf_projections.hpp"
#include "models/qwen35/prefill_mixer.hpp"
#include "models/qwen35/reference.hpp"

// The family's RoPE positions without a model:
//  - interleaved MRoPE angles against HF transformers 5.21
//    (Qwen3_5TextRotaryEmbedding on Qwen3.5-2B's config) and get_rope_index;
//  - equal axes are plain RoPE, bit for bit (the reference, the prefill
//    mixer's tables);
//  - DecodeState::rope_delta: a tiny random Qwen3.5 GGUF runs an image-like
//    prompt (distinct t, h, w) through the reference, then the CPU decoder
//    continues from the reference's caches at index + rope_delta and must
//    match the reference's own continuation.
namespace {

namespace qwen35 = gufo_npu::qwen35;
using qwen35::RopePosition;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

// Qwen3.5-2B's text RoPE: head_dim 256, partial_rotary_factor 0.25, theta
// 1e7, mrope_section [11, 11, 10].
qwen35::Config Rope2b() {
  qwen35::Config c;
  c.head_dim = 256;
  c.rope_dim = 64;
  c.rope_theta = 1e7f;
  c.rope_sections = {11, 11, 10};
  return c;
}

// Printed by HF transformers 5.21: Qwen3_5TextRotaryEmbedding(config =
// Qwen3.5-2B's text_config)(x, position_ids of shape (3, 1, 4)), cos and sin
// of frequencies kIndex (their second half repeats the first).
constexpr int kIndex[] = {0, 1, 2, 3, 4, 5, 29, 30, 31};
struct HfAngles {
  RopePosition p;
  float cos[9];
  float sin[9];
};
constexpr HfAngles kHf[] = {
    {{5, 7, 11},
     {0.2836622f, -0.463830531f, -0.640748799f, 0.450592875f, 0.595052779f,
      0.632184982f, 1, 1, 1},
     {-0.958924294f, -0.885923922f, -0.767750621f, 0.892729521f, 0.803686619f,
      0.774817467f, 4.98474219e-06f, 1.36920994e-06f, 1.15837202e-06f}},
    {{100, 103, 120},
     {0.862318873f, 0.831315458f, 0.987003088f, -0.997098923f, 0.391163498f,
      -0.970057726f, 1, 1, 1},
     {-0.506365657f, -0.555800915f, -0.160701156f, -0.0761166736f, 0.920321226f,
      -0.242874488f, 5.43790047e-05f, 2.73841979e-05f, 1.70446165e-05f}},
    {{0, 3, 9},
     {1, -0.239734888f, -0.989509583f, 1, 0.921039045f, 0.748328328f, 1, 1, 1},
     {0, 0.970838368f, -0.14446725f, 0, 0.389470309f, 0.663328469f,
      4.07842526e-06f, 0, 4.96445125e-07f}},
    {{37, 37, 37},
     {0.765414059f, -0.933114111f, 0.585684121f, -0.305948377f, 0.219830379f,
      -0.987230957f, 1, 1, 1},
     {-0.643538117f, -0.359580457f, 0.810539424f, 0.952048123f, -0.975538135f,
      0.15929538f, 1.67668604e-05f, 1.0132153e-05f, 6.12282338e-06f}},
};

void TestAngles() {
  const auto c = Rope2b();
  const auto inv = qwen35::RopeInvFreq(c);
  // The axes of the 32 frequencies: h at 1, 4, ..., 31; w at 2, 5, ..., 29;
  // t elsewhere (HF's recomposition_frequencies).
  for (int i = 0; i < 32; ++i) {
    const int want = i % 3 == 1 ? 1 : i % 3 == 2 ? 2 : 0;
    Expect(qwen35::MropeAxis(c.rope_sections, i) == want,
           "axis of frequency " + std::to_string(i));
  }
  // HF computes the angles in fp32 (position * fp32 inv_freq); ours round a
  // double angle once, so they differ by a few fp32 ulps of the angle.
  std::vector<float> cs(32), sn(32);
  double worst = 0;
  for (const auto& hf : kHf) {
    qwen35::RopeAngles(c, inv, hf.p, cs.data(), sn.data());
    for (int k = 0; k < 9; ++k) {
      worst = std::max({worst, std::fabs(double{cs[kIndex[k]]} - hf.cos[k]),
                        std::fabs(double{sn[kIndex[k]]} - hf.sin[k])});
    }
  }
  Expect(worst < 2e-5,
         "angles match HF (max diff " + std::to_string(worst) + ")");

  // Equal axes are plain RoPE, bit for bit: the angle every mixer computed
  // before MRoPE, double(p) * inv_freq[i].
  bool plain = true;
  for (const int p : {0, 1, 17, 4095, 262143}) {
    qwen35::RopeAngles(c, inv, qwen35::TextPosition(p), cs.data(), sn.data());
    for (int i = 0; i < 32; ++i) {
      const double angle = static_cast<double>(p) * inv[i];
      plain = plain && cs[i] == static_cast<float>(std::cos(angle)) &&
              sn[i] == static_cast<float>(std::sin(angle));
    }
  }
  Expect(plain, "t == h == w is plain RoPE, bit for bit");
  // Without sections (a GGUF without rope.dimension_sections) every
  // frequency follows t.
  auto none = c;
  none.rope_sections = {};
  std::vector<float> cs2(32), sn2(32);
  qwen35::RopeAngles(none, inv, {9, 30, 70}, cs.data(), sn.data());
  qwen35::RopeAngles(c, inv, qwen35::TextPosition(9), cs2.data(), sn2.data());
  Expect(cs == cs2 && sn == sn2, "no sections: t only");
}

void TestPromptPositions() {
  // HF get_rope_index (Qwen3.5-2B's config, spatial_merge_size 2) on
  // [10, 11, pad x 6, 12, 13, 14, pad x 3, 15] with image_grid_thw
  // [[1, 4, 6], [1, 6, 2]].
  constexpr int kPad = 248056;
  std::vector<int> ids = {10, 11};
  ids.insert(ids.end(), 6, kPad);
  ids.insert(ids.end(), {12, 13, 14});
  ids.insert(ids.end(), 3, kPad);
  ids.push_back(15);
  const int t[] = {0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 8, 8, 8, 11};
  const int h[] = {0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7, 8, 9, 10, 11};
  const int w[] = {0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 8, 8, 11};
  const auto got =
      qwen35::ImagePromptPositions(ids, kPad, {{1, 4, 6}, {1, 6, 2}});
  bool same = got.positions.size() == ids.size();
  for (std::size_t i = 0; same && i < ids.size(); ++i) {
    same = got.positions[i] == RopePosition{t[i], h[i], w[i]};
  }
  Expect(same, "image prompt positions match HF get_rope_index");
  Expect(got.rope_delta == -3, "image prompt rope_delta matches HF (-3)");

  const auto text = qwen35::ImagePromptPositions({1, 2, 3}, kPad, {});
  Expect(text.rope_delta == 0 && text.positions.size() == 3 &&
             text.positions[2] == qwen35::TextPosition(2),
         "text-only prompt: plain positions, delta 0");
  auto throws = [&](const std::vector<qwen35::ImageGrid>& grids) {
    try {
      (void)qwen35::ImagePromptPositions(ids, kPad, grids);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  Expect(throws({{1, 4, 6}}), "throws: fewer grids than image runs");
  Expect(throws({{1, 4, 6}, {1, 4, 2}}), "throws: run does not match grid");
  Expect(throws({{1, 4, 6}, {1, 6, 2}, {1, 2, 2}}),
         "throws: more grids than image runs");
}

// The prefill mixer's RoPE tables, through AppendKv's rotated keys: equal
// axes match the scalar SetChunk bit for bit, and each frequency pair of an
// MRoPE key is the text key's at that frequency's axis position.
void TestPrefillMixer() {
  qwen35::Config c;
  c.heads = 4;
  c.kv_heads = 1;
  c.head_dim = 32;
  c.rope_dim = 16;
  c.rope_theta = 1e7f;
  c.rope_sections = {3, 3, 2};
  c.rms_eps = 1e-6f;
  c.conv_kernel = 4;
  c.linear_key_heads = 1;
  c.linear_value_heads = 1;
  c.linear_head_dim = 32;
  constexpr int kTokens = 5;
  gufo_npu::BurstPool pool(2);
  const gufo_npu::BurstPool::Burst burst(pool);
  qwen35::PrefillMixer mixer(c, pool, kTokens);
  qwen35::PrefillMixerWeights w;
  w.attention = true;
  w.k_norm.assign(c.head_dim, 1.0f);
  w.q_norm.assign(c.head_dim, 1.0f);
  const std::size_t stride = 2 * c.heads * c.head_dim + 2 * c.head_dim;
  std::vector<float> in(kTokens * stride);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (auto& x : in) {
    x = u(rng);
  }
  auto keys = [&](auto set) {
    qwen35::DecodeState::Layer st;
    set();
    mixer.AppendKv(in.data(), stride, w, st);
    return st.keys;
  };
  constexpr int kStart = 11;
  const auto scalar = keys([&] { mixer.SetChunk(kTokens, kStart); });
  std::vector<RopePosition> text(kTokens);
  for (int i = 0; i < kTokens; ++i) {
    text[i] = qwen35::TextPosition(kStart + i);
  }
  const auto rows = keys([&] { mixer.SetChunk(kTokens, text.data()); });
  Expect(scalar == rows, "mixer: equal axes == SetChunk(n, position)");

  // Every token of the chunk at (t, h, w) = (2, 9, 30): pair i of each key
  // is the text key's at position (2, 9 or 30) (pairs i, i + 8).
  const std::vector<RopePosition> image(kTokens, {2, 9, 30});
  const auto mrope = keys([&] { mixer.SetChunk(kTokens, image.data()); });
  bool pairs = true;
  for (const int axis : {0, 1, 2}) {
    const int p = axis == 0 ? 2 : axis == 1 ? 9 : 30;
    const auto at = keys([&] {
      const std::vector<RopePosition> same(kTokens, qwen35::TextPosition(p));
      mixer.SetChunk(kTokens, same.data());
    });
    for (int t = 0; t < kTokens; ++t) {
      for (int e = 0; e < c.head_dim; ++e) {
        const int freq = e % (c.rope_dim / 2);
        const bool rotated = e < c.rope_dim;
        if (rotated && qwen35::MropeAxis(c.rope_sections, freq) != axis) {
          continue;
        }
        const std::size_t k = static_cast<std::size_t>(t) * c.head_dim + e;
        pairs = pairs && mrope[k] == at[k];
      }
    }
  }
  Expect(pairs, "mixer: MRoPE pairs follow their axes");
  Expect(mrope != scalar, "mixer: MRoPE keys differ from text keys");
}

// A tiny Qwen3.5 GGUF (F32 tensors, random values) in memory.
class TinyModel {
public:
  static constexpr int kHidden = 64;
  static constexpr int kVocab = 40;
  static constexpr int kFfn = 64;
  static constexpr int kHeads = 4;
  static constexpr int kKvHeads = 2;
  static constexpr int kHeadDim = 32;
  static constexpr int kLinearHeads = 2;
  static constexpr int kLinearDim = 16;
  static constexpr int kChannels = 3 * kLinearHeads * kLinearDim;

  TinyModel() {
    const std::string a = "qwen35.";
    Key("general.architecture", "qwen35");
    Key(a + "block_count", 4u);  // layers 1 and 3 attention
    Key(a + "embedding_length", static_cast<std::uint32_t>(kHidden));
    Key(a + "feed_forward_length", static_cast<std::uint32_t>(kFfn));
    Key(a + "attention.layer_norm_rms_epsilon", 1e-6f);
    Key(a + "full_attention_interval", 2u);
    Key(a + "attention.head_count", static_cast<std::uint32_t>(kHeads));
    Key(a + "attention.head_count_kv", static_cast<std::uint32_t>(kKvHeads));
    Key(a + "attention.key_length", static_cast<std::uint32_t>(kHeadDim));
    Key(a + "attention.value_length", static_cast<std::uint32_t>(kHeadDim));
    Key(a + "rope.dimension_count", 16u);
    Key(a + "rope.freq_base", 10000.0f);
    Sections(a + "rope.dimension_sections", {3, 3, 2, 0});
    Key(a + "ssm.conv_kernel", 4u);
    Key(a + "ssm.state_size", static_cast<std::uint32_t>(kLinearDim));
    Key(a + "ssm.group_count", static_cast<std::uint32_t>(kLinearHeads));
    Key(a + "ssm.time_step_rank", static_cast<std::uint32_t>(kLinearHeads));
    Key(a + "ssm.inner_size",
        static_cast<std::uint32_t>(kLinearHeads * kLinearDim));

    Matrix("token_embd.weight", kHidden, kVocab, 1.0f);
    Norm("output_norm.weight", kHidden);
    for (int il = 0; il < 4; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      Norm(p + "attn_norm.weight", kHidden);
      Norm(p + "post_attention_norm.weight", kHidden);
      Matrix(p + "ffn_gate.weight", kHidden, kFfn);
      Matrix(p + "ffn_up.weight", kHidden, kFfn);
      Matrix(p + "ffn_down.weight", kFfn, kHidden);
      if (il % 2 == 1) {
        Matrix(p + "attn_q.weight", kHidden, 2 * kHeads * kHeadDim);
        Matrix(p + "attn_k.weight", kHidden, kKvHeads * kHeadDim);
        Matrix(p + "attn_v.weight", kHidden, kKvHeads * kHeadDim);
        Matrix(p + "attn_output.weight", kHeads * kHeadDim, kHidden);
        Norm(p + "attn_q_norm.weight", kHeadDim);
        Norm(p + "attn_k_norm.weight", kHeadDim);
      } else {
        const int value = kLinearHeads * kLinearDim;
        Matrix(p + "attn_qkv.weight", kHidden, kChannels);
        Matrix(p + "attn_gate.weight", kHidden, value);
        Matrix(p + "ssm_alpha.weight", kHidden, kLinearHeads);
        Matrix(p + "ssm_beta.weight", kHidden, kLinearHeads);
        Matrix(p + "ssm_out.weight", value, kHidden);
        Matrix(p + "ssm_conv1d.weight", 4, kChannels, 0.5f);
        Values(p + "ssm_a", {kLinearHeads}, -1.0f, 0.5f);
        Values(p + "ssm_dt.bias", {kLinearHeads}, 0.0f, 0.2f);
        Values(p + "ssm_norm.weight", {kLinearDim}, 1.0f, 0.1f);
      }
    }
    Assemble();
    file_ = gufo_npu::gguf::File::FromBytes(bytes_.data(), bytes_.size());
  }

  const gufo_npu::gguf::File& file() const { return *file_; }

private:
  struct Tensor {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<float> data;
  };

  void PutU32(std::uint32_t v) { Put(&v, sizeof(v)); }
  void PutU64(std::uint64_t v) { Put(&v, sizeof(v)); }
  void Put(const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    header_.insert(header_.end(), b, b + n);
  }
  void PutString(const std::string& s) {
    PutU64(s.size());
    Put(s.data(), s.size());
  }
  void Key(const std::string& name, std::uint32_t v) {
    PutString(name);
    PutU32(4);
    PutU32(v);
    ++keys_;
  }
  void Key(const std::string& name, float v) {
    PutString(name);
    PutU32(6);
    Put(&v, sizeof(v));
    ++keys_;
  }
  void Key(const std::string& name, const char* v) {
    PutString(name);
    PutU32(8);
    PutString(v);
    ++keys_;
  }
  // An int32 array, as llama.cpp writes rope.dimension_sections.
  void Sections(const std::string& name, const std::vector<std::int32_t>& v) {
    PutString(name);
    PutU32(9);
    PutU32(5);
    PutU64(v.size());
    Put(v.data(), v.size() * sizeof(std::int32_t));
    ++keys_;
  }
  void Values(const std::string& name, std::vector<std::uint64_t> shape,
              float mean, float spread) {
    std::size_t n = 1;
    for (const auto d : shape) {
      n *= d;
    }
    Tensor t{name, std::move(shape), std::vector<float>(n)};
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& x : t.data) {
      x = mean + spread * u(rng_);
    }
    tensors_.push_back(std::move(t));
  }
  // ggml order: `cols` inputs per row, `rows` outputs.
  void Matrix(const std::string& name, int cols, int rows, float scale = 0) {
    const float spread =
        scale > 0 ? scale : 1.5f / std::sqrt(static_cast<float>(cols));
    Values(name,
           {static_cast<std::uint64_t>(cols), static_cast<std::uint64_t>(rows)},
           0.0f, spread);
  }
  void Norm(const std::string& name, int n) {
    Values(name, {static_cast<std::uint64_t>(n)}, 1.0f, 0.2f);
  }

  void Assemble() {
    std::vector<std::uint8_t> kv = std::move(header_);
    header_.clear();
    PutU32(0x46554747);  // "GGUF"
    PutU32(3);
    PutU64(tensors_.size());
    PutU64(keys_);
    header_.insert(header_.end(), kv.begin(), kv.end());
    std::uint64_t offset = 0;
    std::vector<std::uint64_t> offsets;
    for (const auto& t : tensors_) {
      PutString(t.name);
      PutU32(static_cast<std::uint32_t>(t.shape.size()));
      for (const auto d : t.shape) {
        PutU64(d);
      }
      PutU32(0);  // F32
      PutU64(offset);
      offsets.push_back(offset);
      offset += (t.data.size() * sizeof(float) + 31) / 32 * 32;
    }
    const std::size_t base = (header_.size() + 31) / 32 * 32;
    bytes_ = header_;
    bytes_.resize(base + offset, 0);
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
      std::memcpy(bytes_.data() + base + offsets[i], tensors_[i].data.data(),
                  tensors_[i].data.size() * sizeof(float));
    }
  }

  std::mt19937 rng_{1234};
  std::vector<std::uint8_t> header_;
  std::uint64_t keys_ = 0;
  std::vector<Tensor> tensors_;
  std::vector<std::uint8_t> bytes_;
  std::unique_ptr<gufo_npu::gguf::File> file_;
};

double RelativeL2(const std::vector<float>& a, const float* b) {
  double diff = 0;
  double norm = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff += (double{a[i]} - b[i]) * (double{a[i]} - b[i]);
    norm += double{b[i]} * b[i];
  }
  return std::sqrt(diff / std::max(norm, 1e-30));
}

// The reference's caches as a DecodeState (fp16 K/V, the same layouts).
qwen35::DecodeState ToDecodeState(const qwen35::Config& c,
                                  const qwen35::State& ref,
                                  qwen35::DecodeState state) {
  state.position = ref.position;
  state.rope_delta = ref.rope_delta;
  for (int il = 0; il < c.layers; ++il) {
    auto& to = state.layers[il];
    if (c.IsFullAttention(il)) {
      const auto& from = ref.attention[il];
      to.keys.resize(from.keys.size());
      to.values.resize(from.values.size());
      gufo_npu::vec::ToHalf(from.keys.data(), to.keys.data(),
                            static_cast<int>(from.keys.size()));
      gufo_npu::vec::ToHalf(from.values.data(), to.values.data(),
                            static_cast<int>(from.values.size()));
    } else {
      to.conv = ref.linear[il].conv;
      to.ssm = ref.linear[il].ssm;
    }
  }
  return state;
}

void TestDecodeOffset() {
  const TinyModel model;
  const qwen35::Reference reference(model.file());
  const auto& c = reference.config();
  Expect(c.rope_sections == std::array<int, 3>{3, 3, 2},
         "config: rope.dimension_sections");

  // Text: Forward and ForwardPrompt with explicit text positions and
  // embedding rows equal to token_embd's are the same pass, bit for bit.
  const std::vector<int> words = {3, 17, 8, 25, 1, 30};
  {
    auto a = reference.NewState();
    auto b = reference.NewState();
    const auto plain = reference.Forward(words, a);
    qwen35::PromptInput input;
    input.tokens = words;
    for (int t = 0; t < static_cast<int>(words.size()); ++t) {
      input.positions.push_back(qwen35::TextPosition(t));
    }
    // Rows 2 and 3 replaced by token_embd's own rows of their tokens.
    const auto* embed = model.file().FindTensor("token_embd.weight");
    const auto* rows = reinterpret_cast<const float*>(embed->data);
    qwen35::EmbeddingRows two{2, {}};
    for (const int id : {words[2], words[3]}) {
      two.values.insert(two.values.end(), rows + id * TinyModel::kHidden,
                        rows + (id + 1) * TinyModel::kHidden);
    }
    input.embeddings.push_back(std::move(two));
    const auto explicit_text = reference.ForwardPrompt(input, b);
    Expect(plain.logits == explicit_text.logits && b.rope_delta == 0,
           "reference: explicit text positions == Forward, bit for bit");
  }

  // An image-like prompt: text, a 2 x 3 grid of image rows (random
  // embeddings) at (s, s + r, s + c), text after it at s + 3.
  constexpr int kPad = 39;
  const std::vector<int> prompt = {5,    9,    kPad, kPad, kPad, kPad,
                                   kPad, kPad, 12,   7,    21};
  const auto pp = qwen35::ImagePromptPositions(prompt, kPad, {{1, 4, 6}});
  Expect(pp.rope_delta == -3, "tiny prompt rope_delta");
  qwen35::PromptInput input;
  input.tokens = prompt;
  input.positions = pp.positions;
  {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> image(6 * TinyModel::kHidden);
    for (auto& x : image) {
      x = u(rng);
    }
    input.embeddings.push_back({2, std::move(image)});
  }
  const std::vector<int> next = {14, 2, 33, 6};

  // Reference: the prompt with its positions, then the continuation as
  // plain tokens (at index + rope_delta) one at a time.
  auto ref_state = reference.NewState();
  (void)reference.ForwardPrompt(input, ref_state);
  Expect(ref_state.position == 11 && ref_state.rope_delta == -3,
         "reference: rope_delta after the image prompt");
  const auto prompt_state = ref_state;
  std::vector<std::vector<float>> want;
  for (const int id : next) {
    want.push_back(reference.Forward({id}, ref_state).logits);
  }
  // The same continuation in one ForwardPrompt with HF's positions for the
  // whole sequence (the continuation at 8, 9, 10, 11 on every axis).
  {
    auto whole = input;
    whole.tokens.insert(whole.tokens.end(), next.begin(), next.end());
    for (int i = 0; i < static_cast<int>(next.size()); ++i) {
      whole.positions.push_back(qwen35::TextPosition(8 + i));
    }
    auto state = reference.NewState();
    const auto out = reference.ForwardPrompt(whole, state);
    double worst = 0;
    for (std::size_t i = 0; i < next.size(); ++i) {
      worst = std::max(
          worst, RelativeL2(want[i],
                            out.logits.data() + (prompt.size() + i) * c.vocab));
    }
    Expect(worst < 1e-6,
           "reference: continuation at index + rope_delta == "
           "explicit positions (rel err " +
               std::to_string(worst) + ")");
  }

  // The CPU decoder from the reference's caches, with and without the
  // offset: only rope_delta = -3 matches.
  const qwen35::GgufProjections projections(
      model.file(), qwen35::GgufProjections::Precision::kDouble);
  qwen35::Decoder decoder(model.file(), projections);
  auto run = [&](int delta) {
    auto state = ToDecodeState(c, prompt_state, decoder.NewState());
    state.rope_delta = delta;
    double worst = 0;
    for (std::size_t i = 0; i < next.size(); ++i) {
      const auto& logits = decoder.Step(next[i], state);
      worst = std::max(worst, RelativeL2(logits, want[i].data()));
    }
    Expect(state.position == 15 && state.rope_delta == delta,
           "decoder: position advances, rope_delta stays");
    return worst;
  };
  const double with = run(-3);
  const double without = run(0);
  std::printf(
      "decoder vs reference after an image prompt: rel err %.2e with "
      "rope_delta -3, %.2e with 0\n",
      with, without);
  Expect(with < 2e-3, "decoder with rope_delta matches the reference");
  Expect(without > 10 * with, "decoder without rope_delta does not");
}

}  // namespace

int main() {
  TestAngles();
  TestPromptPositions();
  TestPrefillMixer();
  TestDecodeOffset();
  if (failures == 0) {
    std::puts("PASS: qwen35_rope_test");
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
