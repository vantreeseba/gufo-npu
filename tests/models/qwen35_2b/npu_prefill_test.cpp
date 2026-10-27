#include "models/qwen35/npu_prefill.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
#include "core/vec.hpp"
#include "models/qwen35/npu_decoder.hpp"

// Qwen3.5-2B prompt processing on the NPU (NpuPrefill: bfp16 GEMMs, batched
// CPU mixers) against the decode path (NpuDecoder, two tokens per pass) over
// the same q4nx weights, on the oracle prompts and on prompts of 1, 2, 256,
// 257, 600, 1,100 and 2,049 tokens (one and two chunks of 256, 512- and
// 1,024-row chunks). For every prompt but its last token:
//   - the final residual of every position (prefill's hidden rows vs the
//     decoder's);
//   - the state the decoder continues from, layer by layer: each DeltaNet
//     layer's conv window and recurrent state, each attention layer's K/V
//     cache;
//   - the MTP layer's K/V cache filled by NpuPrefill::FillMtp against
//     NpuDecoder::FillMtp, from the same hidden rows;
//   - then the decoder runs the last token and 16 greedy steps from both
//     states, teacher-forced on the decode path's tokens: logits relative
//     error and argmax.
// Then MTP speculative decoding after prefill must give exactly the plain
// greedy output after prefill.
//   GUFO_NPU_QWEN35_2B_Q4_GGUF  UD-Q4_K_XL GGUF of Qwen3.5-2B
//   GUFO_NPU_QWEN35_2B_ORACLE   oracle output directory (the prompts)
// Exits 77 (skip) when either is unset or there is no NPU.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;
using gufo_npu::qwen35::DecodeState;

constexpr int kSteps = 16;  // teacher-forced decode steps after the prompt
// The bfp16 activations of the prefill GEMMs (~7e-3 each, 96 per pass) and
// their bf16 outputs, against the decode GEMV's bf16 inputs. Measured up to
// 2,049 tokens (2026-10-27): hidden <= 4.1e-2, conv windows and K/V caches
// <= 4.4e-2, DeltaNet states <= 5.8e-2 (they integrate every token's
// error), MTP K/V <= 1.3e-2, logits <= 3.3e-2. A wrong offset or a stale
// window gives errors of order 1.
constexpr double kMaxHiddenError = 6e-2;
constexpr double kMaxStateError = 8e-2;
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
  void Add(double got, double want) {
    diff += (got - want) * (got - want);
    norm += want * want;
  }
  void Add(const float* got, const float* want, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      Add(got[i], want[i]);
    }
  }
  void Add(const std::vector<std::uint16_t>& got,
           const std::vector<std::uint16_t>& want) {
    for (std::size_t i = 0; i < want.size(); ++i) {
      Add(gufo_npu::vec::FromHalf(got[i]), gufo_npu::vec::FromHalf(want[i]));
    }
  }
  double Relative() const { return norm > 0 ? std::sqrt(diff / norm) : 0; }
};

// Worst relative error over the layers of each kind of state; false (and a
// note) when the two states do not have the same shape.
struct StateError {
  double conv = 0;
  double ssm = 0;
  double kv = 0;
};

bool CompareStates(const DecodeState& got, const DecodeState& want,
                   std::size_t layers, StateError* out) {
  if (got.position != want.position) {
    std::printf("  position %d, want %d\n", got.position, want.position);
    return false;
  }
  for (std::size_t il = 0; il < layers; ++il) {
    const auto& g = got.layers[il];
    const auto& w = want.layers[il];
    if (g.conv.size() != w.conv.size() || g.ssm.size() != w.ssm.size() ||
        g.keys.size() != w.keys.size() || g.values.size() != w.values.size()) {
      std::printf("  layer %zu: state sizes differ\n", il);
      return false;
    }
    Error conv, ssm, kv;
    conv.Add(g.conv.data(), w.conv.data(), w.conv.size());
    ssm.Add(g.ssm.data(), w.ssm.data(), w.ssm.size());
    kv.Add(g.keys, w.keys);
    kv.Add(g.values, w.values);
    out->conv = std::max(out->conv, conv.Relative());
    out->ssm = std::max(out->ssm, ssm.Relative());
    out->kv = std::max(out->kv, kv.Relative());
  }
  return true;
}

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
  const int k = n - 1;  // prompt tokens before the last

  // Decode path: tokens [0, k) two per pass, then the last token and
  // greedy steps.
  auto ds = npu.NewState();
  std::vector<float> want_hidden(static_cast<std::size_t>(k) * d);
  for (int p = 0; p < k; p += 2) {
    const int count = std::min(2, k - p);
    npu.Step(&ids[p], count, ds);
    if (count == 2) {
      npu.Resolve(ds, true);
    }
    for (int t = 0; t < count; ++t) {
      std::copy_n(npu.hidden(t), d, want_hidden.data() + (p + t) * d);
    }
  }
  const DecodeState want_state = ds;
  std::vector<std::vector<float>> want_logits;
  std::vector<int> tokens;
  const float* logits = npu.Step(&ids[k], 1, ds);
  for (int s = 0; s <= kSteps; ++s) {
    want_logits.emplace_back(logits, logits + vocab);
    tokens.push_back(Argmax(logits, vocab));
    if (s < kSteps) {
      logits = npu.Step(&tokens.back(), 1, ds);
    }
  }

  // Prefill: the same k tokens, then the decoder from that state.
  auto ps = npu.NewState();
  std::vector<float> hidden(static_cast<std::size_t>(k) * d);
  const auto start = std::chrono::steady_clock::now();
  prefill.Run(ids.data(), k, ps, hidden.data());
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  Error hidden_err;
  hidden_err.Add(hidden.data(), want_hidden.data(), hidden.size());
  StateError state_err;
  bool ok = CompareStates(ps, want_state, static_cast<std::size_t>(c.layers),
                          &state_err);

  // The MTP cache over positions 1 .. k: pairs (decode path's hidden row
  // p - 1, token p), through both fills.
  Error mtp_err;
  if (c.mtp_layers > 0 && k >= 1) {
    auto mtp_want = npu.NewMtpState();
    for (int p = 1; p <= k; p += 2) {
      const int count = std::min(2, k + 1 - p);
      const float* hs[2] = {want_hidden.data() + (p - 1) * d,
                            want_hidden.data() + std::min(p, k - 1) * d};
      npu.FillMtp(hs, &ids[p], count, p, mtp_want);
    }
    auto mtp_got = npu.NewMtpState();
    prefill.FillMtp(want_hidden.data(), &ids[1], k, 1, mtp_got);
    const auto& g = mtp_got.layers[c.layers];
    const auto& w = mtp_want.layers[c.layers];
    if (g.keys.size() != w.keys.size()) {
      std::printf("  MTP cache sizes differ\n");
      ok = false;
    } else {
      mtp_err.Add(g.keys, w.keys);
      mtp_err.Add(g.values, w.values);
    }
  }

  Error logit_err;
  std::size_t agree = 0;
  logits = npu.Step(&ids[k], 1, ps);
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
  ok = ok && hidden_err.Relative() < kMaxHiddenError &&
       state_err.conv < kMaxStateError && state_err.ssm < kMaxStateError &&
       state_err.kv < kMaxStateError && mtp_err.Relative() < kMaxStateError &&
       logit_err.Relative() < kMaxLogitError && first_same;
  totals.failures += ok ? 0 : 1;
  totals.agree += agree;
  totals.positions += kSteps + 1;
  std::printf(
      "%s %s: %d prompt tokens (prefill %.3f s); rel err: hidden %.2e, conv "
      "%.2e, ssm %.2e, kv %.2e, MTP kv %.2e, logits %.2e; argmax %zu/%d "
      "(first token %s)\n",
      ok ? "ok  " : "FAIL", name.c_str(), n, seconds, hidden_err.Relative(),
      state_err.conv, state_err.ssm, state_err.kv, mtp_err.Relative(),
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
  // Lengths at the chunk boundaries: nothing to prefill, one token, a full
  // and a just-over 256-token chunk, two lanes of 256, of 512 and of 1,024.
  for (const int length : {1, 2, 256, 257, 600, 1100, 2049}) {
    std::vector<int> ids;
    while (static_cast<int>(ids.size()) < length) {
      ids.insert(ids.end(), all.begin(), all.end());
    }
    ids.resize(length);
    Compare(npu, prefill, "len" + std::to_string(length), ids, totals);
  }

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
