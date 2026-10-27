#include "models/qwen36_35b_a3b/npu_prefill.hpp"

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

// Qwen3.6-35B-A3B prompt processing on the NPU (qwen36_35b_a3b::NpuPrefill:
// q4nx converted to bfp16 on the device, bfp16 GEMMs, CPU mixers and
// routing) against the decode path (NpuDecoder, one token per pass) over the
// same q4nx weights, on the oracle prompts and on prompts of 1, 2, 1,024,
// 1,025, 2,049 and 4,000 tokens (dense blocks of 1,024; experts split into
// several 1,024-row calls; several expert groups per layer).
//
// Top-k routing is discontinuous: wherever a layer's 8th and 9th experts
// are nearly tied, the prefill's bfp16 rounding can pick the other one, and
// from there the two paths drift apart. So the prefill first follows the
// decode path's routing (NpuPrefill::FollowRouting), and every token's state
// is compared closely:
//   - the final residual of every position;
//   - the state the decoder continues from, layer by layer: each DeltaNet
//     layer's conv window and recurrent state, each attention layer's K/V
//     cache;
//   - the MTP layer's K/V cache filled by NpuPrefill::FillMtp against
//     NpuDecoder::FillMtp, from the same hidden rows;
//   - then the decoder runs the last token and 16 greedy steps from both
//     states, teacher-forced on the decode path's tokens: logits relative
//     error and argmax.
// Wherever the decode path chose other experts than the prefill would have,
// it must have been a near tie in the prefill's own router logits.
// Then, routing on its own: the first generated token must match the decode
// path's, and a second run of the same prompt must give bit-identical
// states. Last, MTP speculative decoding after prefill must give exactly the
// plain greedy output after prefill.
//   GUFO_NPU_QWEN36_MODEL   the Qwen3.6-35B-A3B GGUF
//   GUFO_NPU_QWEN36_ORACLE  oracle output directory (the prompts)
// Exits 77 (skip) when either is unset or there is no NPU.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;
namespace moe = gufo_npu::qwen36_35b_a3b;
using gufo_npu::qwen35::DecodeState;

constexpr int kSteps = 16;  // teacher-forced decode steps after the prompt
// docs/models/qwen3.6-35b-a3b/QUALITY.md, "NPU prefill".
constexpr double kMaxHiddenError = 6e-2;
constexpr double kMaxStateError = 8e-2;
constexpr double kMaxLogitError = 5e-2;
constexpr double kMinArgmaxAgreement = 0.95;
// Relative probability gap between an expert only the decode path chose and
// one only the prefill would have (as the decode test's kNearTie).
constexpr double kNearTie = 5e-2;

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

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
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

bool SameStates(const DecodeState& a, const DecodeState& b) {
  if (a.position != b.position || a.layers.size() != b.layers.size()) {
    return false;
  }
  for (std::size_t il = 0; il < a.layers.size(); ++il) {
    const auto& x = a.layers[il];
    const auto& y = b.layers[il];
    if (x.conv != y.conv || x.ssm != y.ssm || x.keys != y.keys ||
        x.values != y.values) {
      return false;
    }
  }
  return true;
}

struct Totals {
  int failures = 0;
  std::size_t agree = 0;
  std::size_t positions = 0;
  int differences = 0;
  long routings = 0;
  double gap = 0;
  int first_same = 0;
  int prompts = 0;
};

void Compare(qwen35::NpuDecoder& npu, moe::NpuPrefill& prefill,
             const std::string& name, const std::vector<int>& ids,
             Totals& totals) {
  const auto& c = npu.config();
  const int vocab = c.vocab;
  const auto d = static_cast<std::size_t>(c.hidden);
  const int n = static_cast<int>(ids.size());
  const int k = n - 1;  // prompt tokens before the last
  const int used = c.experts_used;

  // Decode path: tokens [0, k) one per pass (recording each layer's
  // experts), then the last token and greedy steps.
  auto start = std::chrono::steady_clock::now();
  auto ds = npu.NewState();
  std::vector<float> want_hidden(static_cast<std::size_t>(k) * d);
  std::vector<int> routing(static_cast<std::size_t>(k) * c.layers * used);
  for (int p = 0; p < k; ++p) {
    npu.Step(&ids[p], 1, ds, false);
    std::copy_n(npu.hidden(0), d, want_hidden.data() + p * d);
    for (int il = 0; il < c.layers; ++il) {
      std::copy_n(npu.experts(il), used,
                  routing.data() +
                      (static_cast<std::size_t>(p) * c.layers + il) * used);
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
  const double decode_seconds = Seconds(start);

  // Prefill following the decode path's routing, then the decoder from it.
  auto ps = npu.NewState();
  std::vector<float> hidden(static_cast<std::size_t>(k) * d);
  prefill.FollowRouting(&routing);
  start = std::chrono::steady_clock::now();
  prefill.Run(ids.data(), k, ps, hidden.data());
  const double seconds = Seconds(start);
  prefill.FollowRouting(nullptr);
  const int differences = prefill.routing_differences();
  const double gap = prefill.routing_gap();
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
  for (int s = 0; s <= kSteps; ++s) {
    logit_err.Add(logits, want_logits[s].data(), vocab);
    agree += Argmax(logits, vocab) == tokens[s];
    if (s < kSteps) {
      logits = npu.Step(&tokens[s], 1, ps);
    }
  }

  // Routing on its own: the first token, and a second run bit for bit.
  auto own = npu.NewState();
  prefill.Run(ids.data(), k, own, hidden.data());
  auto again = npu.NewState();
  std::vector<float> hidden2(hidden.size());
  prefill.Run(ids.data(), k, again, hidden2.data());
  const bool deterministic = SameStates(own, again) && hidden == hidden2;
  const int own_differences = prefill.routing_differences();
  logits = npu.Step(&ids[k], 1, own);
  const bool first_same = Argmax(logits, vocab) == tokens[0];

  ok = ok && hidden_err.Relative() < kMaxHiddenError &&
       state_err.conv < kMaxStateError && state_err.ssm < kMaxStateError &&
       state_err.kv < kMaxStateError && mtp_err.Relative() < kMaxStateError &&
       logit_err.Relative() < kMaxLogitError && gap < kNearTie &&
       deterministic && first_same;
  totals.failures += ok ? 0 : 1;
  totals.agree += agree;
  totals.positions += kSteps + 1;
  totals.differences += differences;
  totals.routings += static_cast<long>(k) * c.layers;
  totals.gap = std::max(totals.gap, gap);
  totals.first_same += first_same;
  ++totals.prompts;
  std::printf(
      "%s %s: %d prompt tokens (prefill %.2f s, decode path %.1f s); rel "
      "err: hidden %.2e, conv %.2e, ssm %.2e, kv %.2e, MTP kv %.2e, logits "
      "%.2e; argmax %zu/%d; %d of %d routings followed unlike its own "
      "(largest gap %.1e); own routing: first token %s, %s\n",
      ok ? "ok  " : "FAIL", name.c_str(), n, seconds, decode_seconds,
      hidden_err.Relative(), state_err.conv, state_err.ssm, state_err.kv,
      mtp_err.Relative(), logit_err.Relative(), agree, kSteps + 1, differences,
      k * c.layers, gap, first_same ? "same" : "DIFFERS",
      deterministic ? "repeat bit-identical" : "repeat DIFFERS");
  (void)own_differences;
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen36_npu_prefill_test <kernel-dir>\n");
    return 2;
  }
  const char* model = std::getenv("GUFO_NPU_QWEN36_MODEL");
  const char* oracle = std::getenv("GUFO_NPU_QWEN36_ORACLE");
  if (model == nullptr || oracle == nullptr || *model == '\0' ||
      *oracle == '\0') {
    std::puts("SKIP: set GUFO_NPU_QWEN36_MODEL and GUFO_NPU_QWEN36_ORACLE");
    return 77;
  }
  if (!fs::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(model);
  auto start = std::chrono::steady_clock::now();
  qwen35::NpuDecoder npu(*file, argv[1]);
  const double decoder_s = Seconds(start);
  start = std::chrono::steady_clock::now();
  moe::NpuPrefill prefill(*file, npu, argv[1]);
  std::printf("loaded the NPU decoder in %.1f s and the prefill in %.1f s\n",
              decoder_s, Seconds(start));

  Totals totals;
  std::vector<int> all;
  for (const char* name : {"chat", "code", "prose", "long"}) {
    const auto ids =
        ReadIds(fs::path(oracle) / (std::string(name) + ".ids.json"));
    if (ids.size() < 2) {
      std::fprintf(stderr, "FAIL: %s: no prompt\n", name);
      return 1;
    }
    all.insert(all.end(), ids.begin(), ids.end());
    Compare(npu, prefill, name, ids, totals);
  }
  // Lengths at the boundaries: nothing to prefill, one token, a full and a
  // just-over 1,024-token dense block, three blocks, and a prompt whose hot
  // experts and shared expert take several 1,024-row calls and whose experts
  // take several groups.
  for (const int length : {1, 2, 1024, 1025, 2049, 4000}) {
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
      "%s argmax after prefill vs decode (routing followed): %zu/%zu (%.1f%%, "
      "need %.0f%%); %d of %ld routings followed unlike the prefill's own "
      "(largest gap %.1e, need < %.0e); own routing: first token same on %d "
      "of %d prompts\n",
      agree_ok ? "ok  " : "FAIL", totals.agree, totals.positions, 100 * rate,
      100 * kMinArgmaxAgreement, totals.differences, totals.routings,
      totals.gap, kNearTie, totals.first_same, totals.prompts);

  // Speculative decoding after prefill must give the plain greedy output
  // after prefill, token for token (on a prompt long enough for the hook to
  // prefill it).
  {
    const auto prompt = ReadIds(fs::path(oracle) / "long.ids.json");
    const std::vector<int> stop;  // none: always max_tokens
    constexpr int kTokens = 48;
    const auto hook = moe::PrefillPrompt(prefill);
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
        same ? "ok  " : "FAIL", spec.size(), same ? "equal" : "DIFFER from",
        spec_stats.steps, spec_stats.accepted,
        spec_stats.steps ? 100.0 * spec_stats.accepted / spec_stats.steps : 0.0,
        common, plain.size());
  }
  std::printf("%s\n", totals.failures == 0 ? "PASS" : "FAIL");
  return totals.failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
