#include "models/qwen35/npu_prefill.hpp"

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
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/npu_decoder.hpp"

// Qwen3.5-2B prompt processing on the NPU (NpuPrefill: bfp16 GEMMs, batched
// CPU mixers) against the decode path (NpuDecoder, one or two tokens per
// pass) over the same q4nx weights, on the oracle prompts and on a long
// prompt that spans several GEMM chunks:
//   - the final residual of every prompt position (prefill's hidden rows vs
//     the decoder's), relative L2 error;
//   - the prompt's last token run by the decoder after prefill vs after
//     token-by-token decode, then teacher-forced decode steps on the
//     decode path's greedy tokens: logits relative error and argmax;
//   - MTP speculative decoding after prefill gives exactly the plain greedy
//     output after prefill.
//   GUFO_NPU_QWEN35_2B_Q4_GGUF  UD-Q4_K_XL GGUF of Qwen3.5-2B
//   GUFO_NPU_QWEN35_2B_ORACLE   oracle output directory (the prompts)
// Exits 77 (skip) when either is unset or there is no NPU.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;

constexpr int kSteps = 16;  // teacher-forced decode steps after the prompt
constexpr double kMaxHiddenError = 5e-2;
constexpr double kMaxLogitError = 5e-2;
constexpr double kMinArgmaxAgreement = 0.95;

std::vector<int> ReadIds(const fs::path& path) {
  std::ifstream in(path);
  std::string text((std::istreambuf_iterator<char>(in)), {});
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

int Argmax(const float* x, int n) {
  return static_cast<int>(std::max_element(x, x + n) - x);
}

struct Error {
  double diff = 0;
  double norm = 0;
  void Add(const float* got, const float* want, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      const double e = static_cast<double>(got[i]) - want[i];
      diff += e * e;
      norm += static_cast<double>(want[i]) * want[i];
    }
  }
  double Relative() const { return std::sqrt(diff / norm); }
};

struct Totals {
  int failures = 0;
  std::size_t agree = 0;
  std::size_t positions = 0;
};

void Compare(qwen35::NpuDecoder& npu, qwen35::NpuPrefill& prefill,
             const std::string& name, const std::vector<int>& ids,
             Totals& totals) {
  const auto& c = npu.config();
  const int vocab = c.vocab;
  const auto d = static_cast<std::size_t>(c.hidden);
  const int n = static_cast<int>(ids.size());

  // Decode path: the prompt token by token, then greedy steps.
  auto ds = npu.NewState();
  std::vector<float> want_hidden(static_cast<std::size_t>(n) * d);
  std::vector<std::vector<float>> want_logits;
  std::vector<int> tokens;
  const float* logits = nullptr;
  for (int p = 0; p < n; ++p) {
    logits = npu.Step(&ids[p], 1, ds);
    std::copy_n(npu.hidden(), d, want_hidden.data() + p * d);
  }
  for (int s = 0; s <= kSteps; ++s) {
    want_logits.emplace_back(logits, logits + vocab);
    tokens.push_back(Argmax(logits, vocab));
    if (s < kSteps) {
      logits = npu.Step(&tokens.back(), 1, ds);
    }
  }

  // Prefill path: all but the last prompt token, then the decoder.
  auto ps = npu.NewState();
  std::vector<float> hidden(static_cast<std::size_t>(n - 1) * d);
  const auto start = std::chrono::steady_clock::now();
  prefill.Run(ids.data(), n - 1, ps, hidden.data());
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  Error hidden_err;
  hidden_err.Add(hidden.data(), want_hidden.data(), hidden.size());
  Error logit_err;
  std::size_t agree = 0;
  logits = npu.Step(&ids[n - 1], 1, ps);
  bool first_same = false;
  for (int s = 0; s <= kSteps; ++s) {
    logit_err.Add(logits, want_logits[s].data(), vocab);
    const bool same = Argmax(logits, vocab) == tokens[s];
    agree += same;
    if (s == 0) {
      first_same = same;
    }
    if (s < kSteps) {
      logits = npu.Step(&tokens[s], 1, ps);
    }
  }
  const bool ok = hidden_err.Relative() < kMaxHiddenError &&
                  logit_err.Relative() < kMaxLogitError && first_same;
  totals.failures += ok ? 0 : 1;
  totals.agree += agree;
  totals.positions += kSteps + 1;
  std::printf(
      "%s %s: %d prompt tokens (prefill %.3f s); hidden rel err %.2e; "
      "logits rel err %.2e, argmax %zu/%d (first token %s)\n",
      ok ? "ok  " : "FAIL", name.c_str(), n, seconds, hidden_err.Relative(),
      logit_err.Relative(), agree, kSteps + 1, first_same ? "same" : "DIFFERS");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen35_npu_prefill_test <kernel-dir>\n");
    return 2;
  }
  const char* model = std::getenv("GUFO_NPU_QWEN35_2B_Q4_GGUF");
  const char* oracle = std::getenv("GUFO_NPU_QWEN35_2B_ORACLE");
  if (model == nullptr || oracle == nullptr || *model == '\0' ||
      *oracle == '\0') {
    std::puts(
        "SKIP: set GUFO_NPU_QWEN35_2B_Q4_GGUF and GUFO_NPU_QWEN35_2B_ORACLE");
    return 77;
  }
  if (!fs::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(model);
  auto start = std::chrono::steady_clock::now();
  qwen35::NpuDecoder npu(*file, argv[1]);
  const double decoder_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  start = std::chrono::steady_clock::now();
  qwen35::NpuPrefill prefill(*file, argv[1]);
  std::printf(
      "loaded the NPU decoder in %.1f s and the prefill weights in %.1f s\n",
      decoder_s,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());

  Totals totals;
  std::vector<int> all;
  for (const char* name : {"chat", "code", "prose"}) {
    const auto ids =
        ReadIds(fs::path(oracle) / (std::string(name) + ".ids.json"));
    if (ids.size() < 2) {
      std::fprintf(stderr, "FAIL: %s: no prompt\n", name);
      return 1;
    }
    all.insert(all.end(), ids.begin(), ids.end());
    Compare(npu, prefill, name, ids, totals);
  }
  // A long prompt over several GEMM chunks: the prompts repeated.
  std::vector<int> long_ids;
  while (long_ids.size() < 600) {
    long_ids.insert(long_ids.end(), all.begin(), all.end());
  }
  long_ids.resize(600);
  Compare(npu, prefill, "long", long_ids, totals);

  const double rate = static_cast<double>(totals.agree) / totals.positions;
  const bool agree_ok = rate >= kMinArgmaxAgreement;
  totals.failures += agree_ok ? 0 : 1;
  std::printf(
      "%s argmax after prefill vs decode: %zu/%zu (%.1f%%, need %.0f%%)\n",
      agree_ok ? "ok  " : "FAIL", totals.agree, totals.positions, 100 * rate,
      100 * kMinArgmaxAgreement);

  // Speculative decoding after prefill must give the plain greedy output
  // after prefill, token for token.
  {
    const auto prompt = ReadIds(fs::path(oracle) / "chat.ids.json");
    const std::vector<int> stop;  // none: always max_tokens
    constexpr int kTokens = 48;
    const auto hook = qwen35::PrefillPrompt(prefill);
    qwen35::SpeculativeStats spec_stats;
    const auto plain = qwen35::SpeculativeGenerate(npu, prompt, kTokens, stop,
                                                   false, {}, nullptr, hook);
    const auto spec = qwen35::SpeculativeGenerate(npu, prompt, kTokens, stop,
                                                  true, {}, &spec_stats, hook);
    const auto unprefilled =
        qwen35::SpeculativeGenerate(npu, prompt, kTokens, stop, false, {});
    const bool same = plain == spec;
    std::size_t common = 0;
    while (common < plain.size() && common < unprefilled.size() &&
           plain[common] == unprefilled[common]) {
      ++common;
    }
    totals.failures += same ? 0 : 1;
    std::printf(
        "%s speculative decode after prefill: %zu tokens %s plain greedy; %d "
        "steps, %d drafts accepted (%.0f%%); greedy with and without prefill "
        "agree on the first %zu of %zu tokens\n",
        same ? "ok  " : "FAIL", spec.size(),
        same ? "identical to" : "DIFFER from", spec_stats.steps,
        spec_stats.accepted,
        spec_stats.steps ? 100.0 * spec_stats.accepted / spec_stats.steps : 0.0,
        common, plain.size());
  }
  return totals.failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
