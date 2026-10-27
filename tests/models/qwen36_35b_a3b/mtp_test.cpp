#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/reference.hpp"

// The reference's MTP head on Qwen3.6-35B-A3B. There is no independent MTP
// oracle, so this checks it against the main model: with teacher forcing,
// the head's draft for position p + 1 (from the main model's hidden state at
// p - 1 and the token at p) should usually be the main model's own top-1 for
// p + 1. A miswired head (inputs swapped, wrong norm, wrong head) drafts
// nearly nothing right. Also checks that drafting one token at a time gives
// the same logits as all at once. (Where RoPE starts does not matter: the
// block attends only to its own cache, and RoPE scores depend only on
// position differences.)
//   GUFO_NPU_QWEN36_MODEL   the Qwen3.6-35B-A3B GGUF
//   GUFO_NPU_QWEN36_ORACLE  oracle output directory (for the prompt ids)
// Exits 77 (skip) when either is unset.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;

// docs/models/qwen3.6-35b-a3b/QUALITY.md: drafts that match the main
// model's top-1, over all prompts.
constexpr double kAgreement = 0.5;
constexpr double kIncrementalTolerance = 1e-6;

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

std::size_t Argmax(const float* x, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(x, x + n) - x);
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
  const qwen35::Reference reference(*file);
  const auto& c = reference.config();
  const auto vocab = static_cast<std::size_t>(c.vocab);
  const auto d = static_cast<std::size_t>(c.hidden);
  std::size_t agree = 0;
  std::size_t text_match = 0;
  std::size_t drafts = 0;
  double step_err = 0;

  for (const char* name : {"chat", "code", "prose"}) {
    const auto ids =
        ReadIds(fs::path(oracle) / (std::string(name) + ".ids.json"));
    const std::size_t n = ids.size();
    if (n < 3) {
      std::fprintf(stderr, "FAIL: %s: need at least 3 token ids, have %zu\n",
                   name, n);
      return 1;
    }
    auto state = reference.NewState();
    const auto main = reference.Forward(ids, state, true, true);
    // Last layer outputs at positions 0 .. n - 2, tokens at 1 .. n - 1.
    const std::vector<float> h(&main.hidden[c.layers * n * d],
                               &main.hidden[c.layers * n * d + (n - 1) * d]);
    const std::vector<int> next(ids.begin() + 1, ids.end());

    auto mtp_state = reference.NewState();
    mtp_state.position = 1;  // RoPE at the token's position
    const auto draft = reference.Mtp(h, next, mtp_state);
    std::size_t prompt_agree = 0;
    for (std::size_t j = 0; j + 2 < n; ++j) {
      // Draft j is for position j + 2: the main model's row j + 1, and the
      // text's token j + 2.
      const auto top = Argmax(&draft[j * vocab], vocab);
      prompt_agree +=
          top == Argmax(&main.logits[(j + 1) * vocab], vocab) ? 1 : 0;
      text_match += top == static_cast<std::size_t>(ids[j + 2]) ? 1 : 0;
      ++drafts;
    }
    agree += prompt_agree;

    // One draft at a time: the same logits.
    auto step_state = reference.NewState();
    step_state.position = 1;
    std::vector<float> last;
    for (std::size_t j = 0; j + 1 < n; ++j) {
      last = reference.Mtp(std::vector<float>(&h[j * d], &h[(j + 1) * d]),
                           {next[j]}, step_state);
    }
    double diff = 0;
    double norm = 0;
    const float* want = &draft[(n - 2) * vocab];
    for (std::size_t i = 0; i < vocab; ++i) {
      diff += (static_cast<double>(last[i]) - want[i]) *
              (static_cast<double>(last[i]) - want[i]);
      norm += static_cast<double>(want[i]) * want[i];
    }
    step_err = std::max(step_err, std::sqrt(diff / norm));
    std::printf("  %s: %zu/%zu drafts match the main model's top-1\n", name,
                prompt_agree, n - 2);
  }

  const double agreement = static_cast<double>(agree) / drafts;
  const bool ok = agreement >= kAgreement && step_err < kIncrementalTolerance;
  std::printf(
      "%s MTP drafts: %zu/%zu (%.0f%%) match the main model's top-1, %zu/%zu "
      "the text; one at a time vs all at once rel err %.2e\n",
      ok ? "ok  " : "FAIL", agree, drafts, 100 * agreement, text_match, drafts,
      step_err);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
