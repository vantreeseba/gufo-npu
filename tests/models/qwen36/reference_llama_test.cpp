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
#include <stdexcept>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/quant.hpp"
#include "core/tokenizer.hpp"
#include "core/vec.hpp"
#include "models/qwen35/chat.hpp"
#include "models/qwen35/reference.hpp"

// The fp32 CPU reference on Qwen3.6-35B-A3B against llama.cpp on the CPU
// (tools/reference/qwen36_llama_oracle.py), same GGUF, same token ids,
// teacher forced, in two ways:
// - end to end: the whole prompt through the reference as it is;
// - layer by layer: each decoder layer (and the LM head) on llama.cpp's own
//   input to it, with llama.cpp's activation rounding applied to the
//   reference's matrix inputs (LlamaCpuRounding), against llama.cpp's
//   output. This leaves only accumulation order between the two, so most
//   outputs agree to float precision.
// Also checks that our tokenizer and chat rendering give llama.cpp's ids,
// and that one-token-at-a-time decode reproduces the whole-prompt forward.
//   GUFO_NPU_QWEN36_MODEL   the Qwen3.6-35B-A3B GGUF
//   GUFO_NPU_QWEN36_ORACLE  oracle output directory
// Exits 77 (skip) when either is unset.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;
using gufo_npu::gguf::GgmlType;
using qwen35::Stage;

// Thresholds and their measurements: docs/models/qwen3.6-35b-a3b/QUALITY.md.
// End to end: logits relative L2 per prompt, and top-1 agreement over all
// prompts. llama.cpp rounds every quantized matmul's input to 8 bits, and
// that noise compounds over 40 layers (llama.cpp one token per call differs
// from itself on the whole prompt by as much); the reference keeps fp32.
constexpr double kLogitTolerance = 1.5e-1;
constexpr double kArgmaxAgreement = 0.95;
// Layer by layer, per token and layer output (relative L2): at least
// kExactShare of a prompt's layer outputs within kExactTolerance. The rest
// follow an accumulation-order difference that flipped one 8-bit rounding,
// which DeltaNet state and the KV cache carry to later tokens; none may
// differ by more than kLayerTolerance. The LM head's logits must agree
// within kExactTolerance, and on every top-1.
constexpr double kExactTolerance = 1e-5;
constexpr double kExactShare = 0.7;
constexpr double kLayerTolerance = 2e-2;
// Token-by-token decode vs the whole-prompt forward.
constexpr double kIncrementalTolerance = 1e-6;

// llama.cpp's CPU matmul converts the activation to the weight's dot-product
// type first (ggml-cpu, `vec_dot_type`): Q8_0 per 32 values with an fp16
// scale for Q8_0 weights, Q8_K per 256 with an fp32 scale for K-quants, bf16
// for BF16 and none for F32. This applies the same rounding to the inputs of
// the reference's own matrix products.
std::vector<float> RoundLikeLlamaCpu(GgmlType type, const float* x,
                                     std::size_t n) {
  std::vector<float> out(x, x + n);
  switch (type) {
    case GgmlType::kF32:
      break;
    case GgmlType::kF16:
      for (auto& v : out) {
        v = gufo_npu::vec::FromHalf(gufo_npu::vec::ToHalf(v));
      }
      break;
    case GgmlType::kBF16:
      for (auto& v : out) {
        v = gufo_npu::quant::Bf16ToFloat(gufo_npu::quant::FloatToBf16(v));
      }
      break;
    case GgmlType::kQ8_0:  // quantize_row_q8_0 (x86)
      for (std::size_t b = 0; b < n; b += 32) {
        float amax = 0;
        for (std::size_t j = b; j < b + 32; ++j) {
          amax = std::max(amax, std::fabs(x[j]));
        }
        const float d =
            gufo_npu::vec::FromHalf(gufo_npu::vec::ToHalf(amax / 127.0f));
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        for (std::size_t j = b; j < b + 32; ++j) {
          out[j] = d * std::nearbyint(x[j] * id);
        }
      }
      break;
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ6_K:  // quantize_row_q8_K_ref
      for (std::size_t b = 0; b < n; b += 256) {
        float amax = 0;
        float max = 0;
        for (std::size_t j = b; j < b + 256; ++j) {
          if (std::fabs(x[j]) > amax) {
            amax = std::fabs(x[j]);
            max = x[j];
          }
        }
        if (amax == 0.0f) {
          std::fill_n(&out[b], 256, 0.0f);
          continue;
        }
        const float iscale = -127.0f / max;
        const float d = 1.0f / iscale;
        for (std::size_t j = b; j < b + 256; ++j) {
          out[j] = d * std::min(127.0f, std::nearbyint(iscale * x[j]));
        }
      }
      break;
  }
  return out;
}

class LlamaCpuRounding : public qwen35::Projections {
public:
  LlamaCpuRounding(const gufo_npu::gguf::File& file,
                   const qwen35::Reference& reference)
      : file_(file), c_(reference.config()), inner_(reference.projections()) {}

  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override {
    const std::size_t in = qwen35::StageInputs(c_, stage, layer);
    const auto rounded =
        RoundLikeLlamaCpu(TypeOf(stage, layer), x, tokens * in);
    inner_.Apply(stage, layer, rounded.data(), tokens, y);
    if (stage == Stage::kLinearIn) {
      // alpha and beta have F32 weights: their input is not rounded.
      const std::size_t out = qwen35::StageOutputs(c_, stage, layer);
      const std::size_t ab =
          2 * static_cast<std::size_t>(c_.linear_value_heads);
      std::vector<float> raw(tokens * out);
      inner_.Apply(stage, layer, x, tokens, raw.data());
      for (int t = 0; t < tokens; ++t) {
        std::copy_n(&raw[(t + 1) * out - ab], ab, y + (t + 1) * out - ab);
      }
    }
  }

  void ApplyExpert(Stage stage, int layer, int expert, const float* x,
                   int tokens, float* y) const override {
    const std::size_t in = qwen35::StageInputs(c_, stage, layer);
    const auto rounded =
        RoundLikeLlamaCpu(TypeOf(stage, layer), x, tokens * in);
    inner_.ApplyExpert(stage, layer, expert, rounded.data(), tokens, y);
  }

private:
  // The weight type of a stage; a fused stage's tensors share one type
  // (except DeltaNet alpha and beta, handled in Apply).
  GgmlType TypeOf(Stage stage, int layer) const {
    const std::string p = "blk." + std::to_string(layer) + ".";
    auto type = [&](const std::string& first, const std::string& second) {
      const auto t = file_.Tensor(first).type;
      if (!second.empty() && file_.Tensor(second).type != t) {
        throw std::runtime_error("fused weights of different types: " + first);
      }
      return t;
    };
    switch (stage) {
      case Stage::kLinearIn:
        return type(p + "attn_qkv.weight", p + "attn_gate.weight");
      case Stage::kAttentionIn:
        return type(p + "attn_q.weight", p + "attn_k.weight");
      case Stage::kMixerOut:
        return type(p + (c_.IsFullAttention(layer) ? "attn_output.weight"
                                                   : "ssm_out.weight"),
                    "");
      case Stage::kFfnIn:
        return type(p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight");
      case Stage::kFfnDown:
        return type(p + "ffn_down_shexp.weight", "");
      case Stage::kRouter:
        return type(p + "ffn_gate_inp.weight", p + "ffn_gate_inp_shexp.weight");
      case Stage::kExpertIn:
        return type(p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight");
      case Stage::kExpertDown:
        return type(p + "ffn_down_exps.weight", "");
      case Stage::kLmHead:
        return type(c_.tied_embeddings ? "token_embd.weight" : "output.weight",
                    "");
    }
    throw std::logic_error("unknown stage");
  }

  const gufo_npu::gguf::File& file_;
  const qwen35::Config& c_;
  const qwen35::Projections& inner_;
};

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

double Median(std::vector<double> x) {
  std::nth_element(x.begin(), x.begin() + x.size() / 2, x.end());
  return x[x.size() / 2];
}

std::size_t Argmax(const float* x, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(x, x + n) - x);
}

// Gap between the largest and second-largest value.
double TopGap(const float* x, std::size_t n) {
  float first = -INFINITY;
  float second = -INFINITY;
  for (std::size_t i = 0; i < n; ++i) {
    if (x[i] > first) {
      second = first;
      first = x[i];
    } else if (x[i] > second) {
      second = x[i];
    }
  }
  return static_cast<double>(first) - second;
}

struct Comparison {
  double logit_err = 0;       // whole prompt
  double worst_position = 0;  // largest per-position error
  std::size_t argmax = 0;     // positions with the same top-1
};

Comparison Compare(const char* name, const char* label,
                   const std::vector<float>& got,
                   const std::vector<float>& want, std::size_t n,
                   std::size_t vocab) {
  Comparison c;
  c.logit_err = RelativeL2(got.data(), want.data(), n * vocab);
  for (std::size_t t = 0; t < n; ++t) {
    const float* g = &got[t * vocab];
    const float* w = &want[t * vocab];
    c.worst_position = std::max(c.worst_position, RelativeL2(g, w, vocab));
    const auto a = Argmax(g, vocab);
    const auto b = Argmax(w, vocab);
    if (a == b) {
      ++c.argmax;
    } else {
      std::printf(
          "  %s, %s, position %zu: top-1 %zu vs llama.cpp %zu; top-2 gap "
          "%.3f vs llama.cpp %.3f\n",
          name, label, t, a, b, TopGap(g, vocab), TopGap(w, vocab));
    }
  }
  return c;
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

double Since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
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
  const auto file = gufo_npu::gguf::File::Open(model);
  const gufo_npu::Tokenizer tokenizer(*file);
  const qwen35::Reference reference(*file);
  const LlamaCpuRounding rounding(*file, reference);
  const auto vocab = static_cast<std::size_t>(reference.config().vocab);
  const fs::path dir = oracle;
  int failures = 0;
  std::size_t agree = 0;
  std::size_t head_agree = 0;
  std::size_t total = 0;

  for (const char* name : {"chat", "code", "prose"}) {
    const auto text = ReadText(dir / (std::string(name) + ".text"));
    const auto ids = ReadIds(dir / (std::string(name) + ".ids.json"));
    const auto want = ReadFloats(dir / (std::string(name) + ".logits.f32"));
    const std::size_t n = ids.size();
    if (n == 0 || want.size() != n * vocab) {
      std::fprintf(stderr, "FAIL: %s: oracle files have unexpected sizes\n",
                   name);
      return 1;
    }

    // Our tokenizer and chat rendering against llama.cpp's.
    bool tokens_ok = tokenizer.Encode(text) == ids;
    if (std::string(name) == "chat") {
      tokens_ok =
          tokens_ok &&
          qwen35::RenderChat({{"user", "Write one sentence about the moon."}},
                             true) == text;
    }

    const auto want_hidden =
        ReadFloats(dir / (std::string(name) + ".hidden.f32"));
    const auto hidden = static_cast<std::size_t>(reference.config().hidden);
    const int layers = reference.config().layers;
    const std::size_t layer_size = n * hidden;
    if (want_hidden.size() != (layers + 1) * layer_size) {
      std::fprintf(stderr, "FAIL: %s: oracle hidden states have %zu values\n",
                   name, want_hidden.size());
      return 1;
    }

    // End to end: the whole prompt, fp32 activations throughout.
    auto state = reference.NewState();
    auto start = std::chrono::steady_clock::now();
    const auto out = reference.Forward(ids, state, true, true);
    const double seconds = Since(start);
    const auto& plain = out.logits;
    const auto p = Compare(name, "end to end", plain, want, n, vocab);
    agree += p.argmax;
    total += n;
    const double last_hidden_err =
        RelativeL2(&out.hidden[layers * layer_size],
                   &want_hidden[layers * layer_size], layer_size);

    // Layer by layer: each layer on llama.cpp's own input to it, with
    // llama.cpp's activation rounding, against llama.cpp's output; then the
    // LM head on llama.cpp's last hidden state.
    std::vector<double> errs;  // per layer and token
    for (int il = 0; il < layers; ++il) {
      std::vector<float> h(&want_hidden[il * layer_size],
                           &want_hidden[(il + 1) * layer_size]);
      auto layer_state = reference.NewState();
      reference.ForwardLayer(il, h, layer_state, &rounding);
      for (std::size_t t = 0; t < n; ++t) {
        errs.push_back(RelativeL2(
            &h[t * hidden], &want_hidden[(il + 1) * layer_size + t * hidden],
            hidden));
      }
    }
    const auto exact = static_cast<std::size_t>(
        std::count_if(errs.begin(), errs.end(),
                      [](double e) { return e < kExactTolerance; }));
    const double exact_share = static_cast<double>(exact) / errs.size();
    const double worst_layer_err = *std::max_element(errs.begin(), errs.end());
    const auto head = reference.Logits(
        std::vector<float>(&want_hidden[layers * layer_size],
                           &want_hidden[(layers + 1) * layer_size]),
        &rounding);
    const auto head_cmp = Compare(name, "LM head", head, want, n, vocab);
    head_agree += head_cmp.argmax;

    // Token by token, for the shortest prompt.
    double step_err = 0;
    double step_seconds = 0;
    if (std::string(name) == "code") {
      auto step_state = reference.NewState();
      std::vector<float> last;
      start = std::chrono::steady_clock::now();
      for (const int id : ids) {
        last = reference.Forward({id}, step_state).logits;
      }
      step_seconds = Since(start);
      step_err = RelativeL2(last.data(), &plain[(n - 1) * vocab], vocab);
    }

    const bool ok = tokens_ok && p.logit_err < kLogitTolerance &&
                    exact_share >= kExactShare &&
                    worst_layer_err < kLayerTolerance &&
                    head_cmp.logit_err < kExactTolerance &&
                    step_err < kIncrementalTolerance;
    std::printf(
        "%s %s: %zu tokens, ids %s; prompt forward %.1f s (%.2f tok/s)\n"
        "     end to end:     logits rel err %.2e (worst position %.2e), "
        "argmax %zu/%zu; last hidden rel err %.2e\n"
        "     layer by layer: %zu/%zu layer outputs (%.1f%%) < %.0e, median "
        "%.1e, worst %.1e\n"
        "     LM head:        logits rel err %.2e, argmax %zu/%zu\n",
        ok ? "ok  " : "FAIL", name, n, tokens_ok ? "match" : "DIFFER", seconds,
        n / seconds, p.logit_err, p.worst_position, p.argmax, n,
        last_hidden_err, exact, errs.size(), 100 * exact_share, kExactTolerance,
        Median(errs), worst_layer_err, head_cmp.logit_err, head_cmp.argmax, n);
    if (step_seconds > 0) {
      std::printf("     token by token: %.2f tok/s, rel err %.2e\n",
                  n / step_seconds, step_err);
    }
    failures += ok ? 0 : 1;
  }

  const double agreement = static_cast<double>(agree) / total;
  const bool ok = agreement >= kArgmaxAgreement && head_agree == total;
  failures += ok ? 0 : 1;
  std::printf(
      "%s argmax end to end %zu/%zu, LM head %zu/%zu; peak RSS %.2f GB (now "
      "%.2f GB anonymous, %.2f GB file-backed)\n",
      ok ? "ok  " : "FAIL", agree, total, head_agree, total,
      StatusKb("VmHWM") / 1048576.0, StatusKb("RssAnon") / 1048576.0,
      StatusKb("RssFile") / 1048576.0);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
