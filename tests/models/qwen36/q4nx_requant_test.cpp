#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/cpus.hpp"
#include "core/gguf.hpp"
#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "core/thread_pool.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/gguf_projections.hpp"
#include "models/qwen35/reference.hpp"

// What the NPU's q4nx weight format (src/core/q4nx.hpp, 4.625 bits per
// weight) costs on Qwen3.6-35B-A3B. The fp32 CPU decoder runs the oracle
// prompts teacher forced, with these weight sets:
//   (a) as stored in the GGUF;
//   (b) every non-expert matrix through q4nx (attention and DeltaNet
//       projections, shared expert, LM head: Q8_0 in the GGUF), routed
//       experts as stored;
//   (c) (b) plus the routed experts through q4nx (Q4_K codes carry over with
//       d and dmin rounded to bf16; Q5_K and Q6_K requantize);
//   (d) (b) with the LM head as stored.
// Weights through q4nx are packed with q4nx::PackRows and multiplied with
// q4nx::Gemv: the weights q4nx::Unpack returns, accumulated in double. Q4_K
// experts are unpacked on the fly by the same rule, checked bit for bit
// against q4nx::Unpack(q4nx::PackRows(...)) first. F32 matrices (router,
// DeltaNet alpha/beta) and the embedding lookup stay as stored. Each set is
// compared with the llama.cpp oracle, the reference (GGUF, double) and (a);
// then greedy decoding of the chat prompt under (a) and (c).
// Reports only; it fails if the Q4_K rule disagrees with q4nx or (a) strays
// from the reference.
//   GUFO_NPU_QWEN36_MODEL   the Qwen3.6-35B-A3B GGUF
//   GUFO_NPU_QWEN36_ORACLE  oracle output directory
// Exits 77 (skip) when either is unset.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;
namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
using gufo_npu::gguf::GgmlType;
using gufo_npu::gguf::TensorInfo;
using qwen35::Stage;

constexpr int kGenerate = 64;
// (a), the fp32 decoder on the stored weights, against the reference.
constexpr double kStoredAgreement = 0.95;

const char* const kPrompts[] = {"chat", "code", "prose", "long"};

std::string ReadText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

std::vector<int> ReadIds(const fs::path& path) {
  std::string text = ReadText(path);
  std::replace_if(
      text.begin(), text.end(),
      [](char ch) { return ch == '[' || ch == ']' || ch == ','; }, ' ');
  std::istringstream stream(text);
  std::vector<int> ids;
  for (int id; stream >> id;) {
    ids.push_back(id);
  }
  return ids;
}

std::vector<float> ReadFloats(const fs::path& path) {
  const std::string bytes = ReadText(path);
  std::vector<float> out(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
  return out;
}

// A Q4_K row as q4nx unpacks it: per 256 values d and dmin rounded to bf16,
// then min + step * code with step = d * sc and min = dmin * -m (q4nx.cpp's
// PackQ4KSuper and Unpack).
void UnpackQ4KAsQ4nx(const std::uint8_t* src, float* dst, std::size_t n) {
  for (std::size_t b = 0; b < n / 256; ++b, src += 144, dst += 256) {
    std::uint16_t bits;
    std::memcpy(&bits, src, 2);
    const float d =
        quant::Bf16ToFloat(quant::FloatToBf16(quant::HalfToFloat(bits)));
    std::memcpy(&bits, src + 2, 2);
    const float dmin =
        quant::Bf16ToFloat(quant::FloatToBf16(quant::HalfToFloat(bits)));
    for (std::size_t g = 0; g < 8; ++g) {
      std::uint8_t sc;
      std::uint8_t m;
      quant::ScaleMinK4(g, src + 4, &sc, &m);
      const float step = d * static_cast<float>(sc);
      const float min = dmin * static_cast<float>(static_cast<std::int8_t>(-m));
      const std::uint8_t* q = src + 16 + (g / 2) * 32;
      const int shift = static_cast<int>(g % 2) * 4;
      for (std::size_t l = 0; l < 32; ++l) {
        dst[g * 32 + l] =
            min + step * static_cast<float>((q[l] >> shift) & 0xF);
      }
    }
  }
}

q4nx::Segment SegmentOf(const TensorInfo& t, std::size_t first = 0,
                        std::size_t rows = 0) {
  return {t.type, t.data + first * quant::RowBytes(t.type, t.RowElements()),
          rows != 0 ? rows : static_cast<std::size_t>(t.Rows())};
}

// Packed q4nx matrices for the weight sets.
struct Store {
  struct Layer {
    q4nx::Matrix in;  // [qkv | z] or [q | k | v]
    q4nx::Matrix out;
    q4nx::Matrix ffn_in;  // shared expert [gate | up]
    q4nx::Matrix ffn_down;
    // Routed experts that requantize, per expert; empty when Q4_K.
    std::vector<q4nx::Matrix> expert_in;
    std::vector<q4nx::Matrix> expert_down;
    const TensorInfo* gate_exps = nullptr;
    const TensorInfo* up_exps = nullptr;
    const TensorInfo* down_exps = nullptr;
  };
  std::vector<Layer> layers;
  q4nx::Matrix lm_head;
};

void PackDense(const gufo_npu::gguf::File& file, const qwen35::Config& c,
               Store& store) {
  const auto hidden = static_cast<std::size_t>(c.hidden);
  store.layers.resize(c.layers);
  for (int il = 0; il < c.layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto t = [&](const char* name) -> const TensorInfo& {
      return file.Tensor(p + name);
    };
    auto& l = store.layers[il];
    if (c.IsFullAttention(il)) {
      l.in = q4nx::PackRows(
          {SegmentOf(t("attn_q.weight")), SegmentOf(t("attn_k.weight")),
           SegmentOf(t("attn_v.weight"))},
          hidden);
      const auto& o = t("attn_output.weight");
      l.out = q4nx::Pack(o.type, o.data, o.Rows(), o.RowElements());
    } else {
      l.in = q4nx::PackRows(
          {SegmentOf(t("attn_qkv.weight")), SegmentOf(t("attn_gate.weight"))},
          hidden);
      const auto& o = t("ssm_out.weight");
      l.out = q4nx::Pack(o.type, o.data, o.Rows(), o.RowElements());
    }
    l.ffn_in = q4nx::PackRows({SegmentOf(t("ffn_gate_shexp.weight")),
                               SegmentOf(t("ffn_up_shexp.weight"))},
                              hidden);
    const auto& down = t("ffn_down_shexp.weight");
    l.ffn_down =
        q4nx::Pack(down.type, down.data, down.Rows(), down.RowElements());
    l.gate_exps = &t("ffn_gate_exps.weight");
    l.up_exps = &t("ffn_up_exps.weight");
    l.down_exps = &t("ffn_down_exps.weight");
  }
  const auto& head = file.Tensor("output.weight");
  store.lm_head = q4nx::Pack(head.type, head.data, head.Rows(), hidden);
}

void PackExperts(const qwen35::Config& c, Store& store) {
  const auto ef = static_cast<std::size_t>(c.expert_ffn);
  const auto hidden = static_cast<std::size_t>(c.hidden);
  for (auto& l : store.layers) {
    for (int e = 0; e < c.experts; ++e) {
      if (l.gate_exps->type != GgmlType::kQ4_K ||
          l.up_exps->type != GgmlType::kQ4_K) {
        l.expert_in.push_back(
            q4nx::PackRows({SegmentOf(*l.gate_exps, e * ef, ef),
                            SegmentOf(*l.up_exps, e * ef, ef)},
                           hidden));
      }
      if (l.down_exps->type != GgmlType::kQ4_K) {
        l.expert_down.push_back(
            q4nx::PackRows({SegmentOf(*l.down_exps, e * hidden, hidden)}, ef));
      }
    }
  }
}

std::size_t PackedBytes(const Store& store, bool experts) {
  std::size_t bytes = store.lm_head.bytes.size();
  for (const auto& l : store.layers) {
    bytes += l.in.bytes.size() + l.out.bytes.size() + l.ffn_in.bytes.size() +
             l.ffn_down.bytes.size();
    if (experts) {
      for (const auto& m : l.expert_in) {
        bytes += m.bytes.size();
      }
      for (const auto& m : l.expert_down) {
        bytes += m.bytes.size();
      }
    }
  }
  return bytes;
}

// The stored projections with some matrices replaced by their q4nx form.
class Q4nxSet : public qwen35::Projections {
public:
  Q4nxSet(const qwen35::Config& c, const qwen35::Projections& stored,
          const Store& store, bool dense, bool lm_head, bool experts)
      : c_(c),
        stored_(stored),
        store_(store),
        dense_(dense),
        lm_head_(lm_head),
        experts_(experts) {}

  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override {
    stored_.Apply(stage, layer, x, tokens, y);
    const q4nx::Matrix* m = nullptr;
    if (stage == Stage::kLmHead) {
      m = lm_head_ ? &store_.lm_head : nullptr;
    } else if (dense_) {
      const auto& l = store_.layers[layer];
      switch (stage) {
        case Stage::kLinearIn:  // alpha and beta (F32) keep the stored rows
        case Stage::kAttentionIn:
          m = &l.in;
          break;
        case Stage::kMixerOut:
          m = &l.out;
          break;
        case Stage::kFfnIn:
          m = &l.ffn_in;
          break;
        case Stage::kFfnDown:
          m = &l.ffn_down;
          break;
        default:
          break;
      }
    }
    if (m != nullptr) {
      Gemv(*m, stage, layer, x, tokens, y);
    }
  }

  void ApplyExpert(Stage stage, int layer, int expert, const float* x,
                   int tokens, float* y) const override {
    if (!experts_) {
      return stored_.ApplyExpert(stage, layer, expert, x, tokens, y);
    }
    const auto& l = store_.layers[layer];
    const auto e = static_cast<std::size_t>(expert);
    const auto ef = static_cast<std::size_t>(c_.expert_ffn);
    const auto hidden = static_cast<std::size_t>(c_.hidden);
    if (stage == Stage::kExpertIn) {
      if (!l.expert_in.empty()) {
        return Gemv(l.expert_in[e], stage, layer, x, tokens, y);
      }
      return Q4K({{l.gate_exps, e * ef, ef}, {l.up_exps, e * ef, ef}}, x,
                 tokens, y);
    }
    if (!l.expert_down.empty()) {
      return Gemv(l.expert_down[e], stage, layer, x, tokens, y);
    }
    Q4K({{l.down_exps, e * hidden, hidden}}, x, tokens, y);
  }

private:
  // The matrix's rows are the stage's leading outputs.
  void Gemv(const q4nx::Matrix& m, Stage stage, int layer, const float* x,
            int tokens, float* y) const {
    const std::size_t in = qwen35::StageInputs(c_, stage, layer);
    const std::size_t out = qwen35::StageOutputs(c_, stage, layer);
    out_.resize(m.rows);
    for (int t = 0; t < tokens; ++t) {
      q4nx::Gemv(m, x + t * in, out_.data());
      std::copy_n(out_.data(), m.rows, y + t * out);
    }
  }

  // Q4_K rows unpacked by q4nx's rule, side by side.
  void Q4K(const std::vector<qwen35::GgufProjections::Rows>& parts,
           const float* x, int tokens, float* y) const {
    const auto in = static_cast<std::size_t>(parts.front().w->RowElements());
    std::size_t stride = 0;
    for (const auto& part : parts) {
      stride += part.rows;
    }
    std::size_t offset = 0;
    for (const auto& part : parts) {
      const auto row_bytes = quant::RowBytes(part.w->type, in);
      constexpr std::size_t kChunk = 32;
      pool_.Run((part.rows + kChunk - 1) / kChunk, [&](std::size_t job) {
        thread_local std::vector<float> row;
        row.resize(in);
        const std::size_t end = std::min(part.rows, (job + 1) * kChunk);
        for (std::size_t r = job * kChunk; r < end; ++r) {
          UnpackQ4KAsQ4nx(part.w->data + (part.first + r) * row_bytes,
                          row.data(), in);
          for (int t = 0; t < tokens; ++t) {
            const float* xt = x + t * in;
            double acc = 0;
            for (std::size_t k = 0; k < in; ++k) {
              acc += static_cast<double>(row[k]) * xt[k];
            }
            y[t * stride + offset + r] = static_cast<float>(acc);
          }
        }
      });
      offset += part.rows;
    }
  }

  const qwen35::Config& c_;
  const qwen35::Projections& stored_;
  const Store& store_;
  bool dense_;
  bool lm_head_;
  bool experts_;
  mutable std::vector<float> out_;
  mutable gufo_npu::SpinPool pool_{gufo_npu::HalfCpuWorkers()};
};

std::size_t Argmax(const float* x, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(x, x + n) - x);
}

double RelativeL2(const float* a, const float* b, std::size_t n) {
  double diff = 0;
  double norm = 0;
  for (std::size_t i = 0; i < n; ++i) {
    diff +=
        (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
    norm += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(diff / std::max(norm, 1e-30));
}

// KL(softmax(p) || softmax(q)) in nats.
double Kl(const float* p, const float* q, std::size_t n) {
  auto lse = [n](const float* x) {
    const double max = *std::max_element(x, x + n);
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sum += std::exp(x[i] - max);
    }
    return max + std::log(sum);
  };
  const double lp = lse(p);
  const double lq = lse(q);
  double kl = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double logp = p[i] - lp;
    kl += std::exp(logp) * (logp - (q[i] - lq));
  }
  return kl;
}

// Logits at every position of `ids`, one decoder step at a time.
std::vector<float> Decode(qwen35::Decoder& decoder,
                          const std::vector<int>& ids) {
  const auto vocab = static_cast<std::size_t>(decoder.config().vocab);
  std::vector<float> out(ids.size() * vocab);
  auto state = decoder.NewState();
  for (std::size_t t = 0; t < ids.size(); ++t) {
    const auto& logits = decoder.Step(ids[t], state);
    std::copy(logits.begin(), logits.end(), out.begin() + t * vocab);
  }
  return out;
}

std::vector<int> Greedy(qwen35::Decoder& decoder, const std::vector<int>& ids,
                        int count) {
  auto state = decoder.NewState();
  const std::vector<float>* logits = nullptr;
  for (const int id : ids) {
    logits = &decoder.Step(id, state);
  }
  std::vector<int> out;
  for (int i = 0; i < count; ++i) {
    out.push_back(static_cast<int>(Argmax(logits->data(), logits->size())));
    logits = &decoder.Step(out.back(), state);
  }
  return out;
}

// A /proc/self/status field in kB.
long StatusKb(const char* field) {
  std::ifstream in("/proc/self/status");
  std::string line;
  const std::size_t len = std::strlen(field);
  while (std::getline(in, line)) {
    if (line.compare(0, len, field) == 0 && line.size() > len &&
        line[len] == ':') {
      return std::strtol(line.c_str() + len + 1, nullptr, 10);
    }
  }
  return -1;
}

std::string Escape(const std::string& text) {
  std::string out;
  for (const char ch : text) {
    out += ch == '\n' ? std::string("\\n") : std::string(1, ch);
  }
  return out;
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_NPU_QWEN36_MODEL");
  const char* oracle = std::getenv("GUFO_NPU_QWEN36_ORACLE");
  if (model == nullptr || oracle == nullptr || *model == '\0' ||
      *oracle == '\0') {
    std::puts("SKIP: set GUFO_NPU_QWEN36_MODEL and GUFO_NPU_QWEN36_ORACLE");
    return 77;
  }
  const auto start = std::chrono::steady_clock::now();
  auto since = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
  };
  const auto file = gufo_npu::gguf::File::Open(model);
  const gufo_npu::Tokenizer tokenizer(*file);
  const qwen35::Reference reference(*file);
  const auto& c = reference.config();
  const auto vocab = static_cast<std::size_t>(c.vocab);
  const qwen35::GgufProjections stored(
      *file, qwen35::GgufProjections::Precision::kFloat);
  int failures = 0;

  // The Q4_K rule against q4nx itself, on one expert's gate rows.
  {
    const auto& gate = file->Tensor("blk.0.ffn_gate_exps.weight");
    const auto ef = static_cast<std::size_t>(c.expert_ffn);
    const auto dim = static_cast<std::size_t>(c.hidden);
    const auto packed =
        q4nx::Unpack(q4nx::PackRows({SegmentOf(gate, 7 * ef, ef)}, dim));
    std::vector<float> mine(ef * dim);
    for (std::size_t r = 0; r < ef; ++r) {
      UnpackQ4KAsQ4nx(
          gate.data + (7 * ef + r) * quant::RowBytes(gate.type, dim),
          &mine[r * dim], dim);
    }
    const bool same =
        std::memcmp(mine.data(), packed.data(), mine.size() * 4) == 0;
    std::printf("%s Q4_K expert rows unpack like q4nx::Unpack(PackRows)\n",
                same ? "ok  " : "FAIL");
    failures += same ? 0 : 1;
  }

  struct Prompt {
    std::string name;
    std::vector<int> ids;
    std::vector<float> llama;
    std::vector<float> ref;
    std::vector<float> stored;  // set (a)
  };
  std::vector<Prompt> prompts;
  for (const char* name : kPrompts) {
    Prompt p;
    p.name = name;
    p.ids = ReadIds(fs::path(oracle) / (p.name + ".ids.json"));
    p.llama = ReadFloats(fs::path(oracle) / (p.name + ".logits.f32"));
    if (p.ids.empty() || p.llama.size() != p.ids.size() * vocab) {
      std::fprintf(stderr, "FAIL: oracle files for %s\n", name);
      return 1;
    }
    auto state = reference.NewState();
    p.ref = reference.Forward(p.ids, state).logits;
    prompts.push_back(std::move(p));
  }
  std::printf("reference logits ready (%.0f s)\n", since());

  Store store;
  PackDense(*file, c, store);
  std::printf("non-expert matrices packed: %.2f GB (%.0f s)\n",
              PackedBytes(store, false) / 1e9, since());

  struct Set {
    const char* name;
    bool dense, lm_head, experts;
  };
  const Set sets[] = {{"(a) as stored", false, false, false},
                      {"(d) non-expert, LM head stored", true, false, false},
                      {"(b) non-expert", true, true, false},
                      {"(c) everything", true, true, true}};
  std::vector<int> greedy_stored;
  for (const auto& set : sets) {
    if (set.experts) {
      PackExperts(c, store);
      std::printf("requantized experts packed: %.2f GB in all (%.0f s)\n",
                  PackedBytes(store, true) / 1e9, since());
    }
    const Q4nxSet projections(c, stored, store, set.dense, set.lm_head,
                              set.experts);
    qwen35::Decoder decoder(*file, projections);
    std::size_t total = 0;
    std::size_t llama_top1 = 0;
    std::size_t ref_top1 = 0;
    std::size_t stored_top1 = 0;
    double kl_sum = 0;
    double kl_max = 0;
    double llama_kl_sum = 0;  // KL((a) || llama.cpp), set (a) only
    double llama_kl_max = 0;
    const bool as_stored = !set.dense && !set.experts;
    std::printf("%s\n", set.name);
    for (auto& p : prompts) {
      const std::size_t n = p.ids.size();
      auto logits = Decode(decoder, p.ids);
      if (!set.dense && !set.experts) {
        p.stored = logits;
      }
      std::size_t llama_match = 0;
      std::size_t ref_match = 0;
      std::size_t stored_match = 0;
      int first_diff = -1;
      double prompt_kl = 0;
      double prompt_kl_max = 0;
      for (std::size_t t = 0; t < n; ++t) {
        const float* got = &logits[t * vocab];
        const auto top = Argmax(got, vocab);
        llama_match += top == Argmax(&p.llama[t * vocab], vocab) ? 1 : 0;
        ref_match += top == Argmax(&p.ref[t * vocab], vocab) ? 1 : 0;
        const bool same = top == Argmax(&p.stored[t * vocab], vocab);
        stored_match += same ? 1 : 0;
        if (!same && first_diff < 0) {
          first_diff = static_cast<int>(t);
        }
        const double kl = Kl(&p.stored[t * vocab], got, vocab);
        prompt_kl += kl;
        prompt_kl_max = std::max(prompt_kl_max, kl);
        if (as_stored) {
          const double llama_kl = Kl(got, &p.llama[t * vocab], vocab);
          llama_kl_sum += llama_kl;
          llama_kl_max = std::max(llama_kl_max, llama_kl);
        }
      }
      total += n;
      llama_top1 += llama_match;
      ref_top1 += ref_match;
      stored_top1 += stored_match;
      kl_sum += prompt_kl;
      kl_max = std::max(kl_max, prompt_kl_max);
      std::printf(
          "  %-6s %3zu tok | llama.cpp: rel L2 %.2e, top-1 %3zu/%zu | "
          "reference: rel L2 %.2e, top-1 %3zu/%zu | vs (a): top-1 %3zu/%zu, "
          "first diff %s, KL mean %.2e max %.2e\n",
          p.name.c_str(), n,
          RelativeL2(logits.data(), p.llama.data(), n * vocab), llama_match, n,
          RelativeL2(logits.data(), p.ref.data(), n * vocab), ref_match, n,
          stored_match, n,
          first_diff < 0 ? "none" : std::to_string(first_diff).c_str(),
          prompt_kl / n, prompt_kl_max);
    }
    std::printf(
        "  all    %3zu tok | llama.cpp top-1 %zu/%zu (%.1f%%) | reference "
        "top-1 %zu/%zu (%.1f%%) | vs (a) top-1 %zu/%zu (%.1f%%), KL mean "
        "%.2e max %.2e (%.0f s)\n",
        total, llama_top1, total, 100.0 * llama_top1 / total, ref_top1, total,
        100.0 * ref_top1 / total, stored_top1, total,
        100.0 * stored_top1 / total, kl_sum / total, kl_max, since());
    if (as_stored) {
      std::printf("  llama.cpp against (a): KL mean %.2e max %.2e\n",
                  llama_kl_sum / total, llama_kl_max);
    }
    if (as_stored && static_cast<double>(ref_top1) / total < kStoredAgreement) {
      ++failures;
    }

    if (!set.dense || set.experts) {
      const auto& chat = prompts.front().ids;
      const auto tokens = Greedy(decoder, chat, kGenerate);
      std::printf("  greedy %d tokens from chat: \"%s\"\n", kGenerate,
                  Escape(tokenizer.Decode(tokens)).c_str());
      if (!set.dense) {
        greedy_stored = tokens;
      } else {
        int same = 0;
        int prefix = 0;
        for (int i = 0; i < kGenerate; ++i) {
          same += tokens[i] == greedy_stored[i] ? 1 : 0;
          if (prefix == i && tokens[i] == greedy_stored[i]) {
            ++prefix;
          }
        }
        std::printf("  greedy vs (a): %d/%d tokens equal, first %d identical\n",
                    same, kGenerate, prefix);
      }
    }
  }
  std::printf("peak RSS %.2f GB (now %.2f GB anonymous, %.2f GB file-backed)\n",
              StatusKb("VmHWM") / 1048576.0, StatusKb("RssAnon") / 1048576.0,
              StatusKb("RssFile") / 1048576.0);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
