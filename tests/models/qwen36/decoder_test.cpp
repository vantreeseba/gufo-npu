#include "models/qwen35/decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/moe.hpp"
#include "models/qwen35/reference.hpp"

// The decode step (Decoder, with its MoE routing) against the reference on
// Qwen3.6-35B-A3B, token by token, both over the reference's own GGUF
// projections, so only the non-matrix math is compared: mixers, routing,
// expert sums, norms. The decoder keeps the attention K/V cache in fp16 and
// computes in fp32 (the reference in fp32 and double).
//
// Top-k routing is discontinuous: where a layer's 8th and 9th experts are
// nearly tied, those small differences can pick a different 8th expert, and
// the change carries on through the later layers and, via the recurrent
// state, to later tokens. So the logits must agree closely up to the first
// token routed differently, the first difference must be at a near tie, and
// top-1 must agree throughout.
//   GUFO_NPU_QWEN36_MODEL   the Qwen3.6-35B-A3B GGUF
//   GUFO_NPU_QWEN36_ORACLE  oracle output directory (for the prompt ids)
// Exits 77 (skip) when either is unset.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;
using qwen35::Stage;

// docs/models/qwen3.6-35b-a3b/QUALITY.md.
// Logits relative L2 before the first routing difference (fp16 K/V cache).
constexpr double kLogitTolerance = 2e-3;
// Largest relative gap between the 8th and 9th expert probabilities (in the
// reference) at which the two may first choose different experts.
constexpr double kNearTie = 1e-3;

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

// Passes everything to `inner` and records each routing decision: the
// chosen experts (sorted) and the relative gap between the last chosen and
// the first unchosen expert's probability.
class RoutingLog : public qwen35::Projections {
public:
  struct Entry {
    std::vector<int> experts;
    double gap = 0;
  };

  RoutingLog(const qwen35::Projections& inner, const qwen35::Config& c)
      : inner_(inner), c_(c) {}

  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override {
    inner_.Apply(stage, layer, x, tokens, y);
    if (stage != Stage::kRouter) {
      return;
    }
    for (int t = 0; t < tokens; ++t) {
      const float* logits = y + static_cast<std::size_t>(t) * (c_.experts + 1);
      Entry entry;
      entry.experts.resize(c_.experts_used);
      std::vector<float> weights(c_.experts_used);
      qwen35::Route(logits, c_.experts, c_.experts_used, entry.experts.data(),
                    weights.data());
      std::vector<float> sorted(logits, logits + c_.experts);
      std::nth_element(sorted.begin(), sorted.begin() + c_.experts_used,
                       sorted.end(), std::greater<>());
      const float next = sorted[c_.experts_used];
      const float last = logits[entry.experts.back()];
      entry.gap = 1.0 - std::exp(static_cast<double>(next) - last);
      std::sort(entry.experts.begin(), entry.experts.end());
      entries.push_back(std::move(entry));
    }
  }

  void ApplyExpert(Stage stage, int layer, int expert, const float* x,
                   int tokens, float* y) const override {
    inner_.ApplyExpert(stage, layer, expert, x, tokens, y);
  }

  mutable std::vector<Entry> entries;  // per token, then layer

private:
  const qwen35::Projections& inner_;
  const qwen35::Config& c_;
};

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
  const qwen35::Reference reference(*file);
  const auto& c = reference.config();
  const RoutingLog ref_routing(reference.projections(), c);
  const RoutingLog dec_routing(reference.projections(), c);
  qwen35::Decoder decoder(*file, dec_routing);
  const auto vocab = static_cast<std::size_t>(c.vocab);

  const auto ids = ReadIds(fs::path(oracle) / "chat.ids.json");
  auto ref_state = reference.NewState();
  auto state = decoder.NewState();
  double worst_before = 0;  // before the first routing difference
  double worst = 0;
  int first_difference = -1;  // position
  double first_gap = 0;       // the reference's gap there
  int differences = 0;
  std::size_t argmax_match = 0;
  double decoder_seconds = 0;
  for (std::size_t t = 0; t < ids.size(); ++t) {
    const auto want =
        reference.Forward({ids[t]}, ref_state, true, false, &ref_routing)
            .logits;
    const auto start = std::chrono::steady_clock::now();
    const auto& got = decoder.Step(ids[t], state);
    decoder_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    for (int il = 0; il < c.layers; ++il) {
      const auto& r = ref_routing.entries[t * c.layers + il];
      if (r.experts != dec_routing.entries[t * c.layers + il].experts) {
        std::printf("  position %zu, layer %d: other experts (gap %.1e)\n", t,
                    il, r.gap);
        ++differences;
        if (first_difference < 0) {
          first_difference = static_cast<int>(t);
          first_gap = r.gap;
        }
      }
    }
    double diff = 0;
    double norm = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
      diff += (static_cast<double>(got[i]) - want[i]) *
              (static_cast<double>(got[i]) - want[i]);
      norm += static_cast<double>(want[i]) * want[i];
    }
    const double err = std::sqrt(diff / norm);
    worst = std::max(worst, err);
    if (first_difference < 0) {
      worst_before = std::max(worst_before, err);
    }
    argmax_match += (std::max_element(got.begin(), got.end()) - got.begin()) ==
                    (std::max_element(want.begin(), want.end()) - want.begin());
  }
  const bool ok = worst_before < kLogitTolerance && first_gap < kNearTie &&
                  argmax_match == ids.size();
  std::printf(
      "%s decoder vs reference: %zu tokens; logits rel err %.2e before the "
      "first routing difference (%s, gap %.1e), %.2e overall; %d routing "
      "differences; argmax %zu/%zu; %.2f s per decoder step over the "
      "reference's projections\n",
      ok ? "ok  " : "FAIL", ids.size(), worst_before,
      first_difference < 0
          ? "none"
          : ("position " + std::to_string(first_difference)).c_str(),
      first_gap, worst, differences, argmax_match, ids.size(),
      decoder_seconds / ids.size());
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
