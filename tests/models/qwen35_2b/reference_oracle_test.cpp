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
#include "models/qwen35/reference.hpp"

// Compares the fp32 CPU reference with the HF transformers oracle
// (tools/reference/qwen35_oracle.py) on the same token ids.
//   GUFO_NPU_QWEN35_2B_GGUF    BF16 GGUF of Qwen3.5-2B (same weights as HF)
//   GUFO_NPU_QWEN35_2B_ORACLE  oracle output directory
// Exits 77 (skip) when either is unset.
namespace {

namespace fs = std::filesystem;

// Relative L2 error between this run and the oracle must stay below these.
constexpr double kHiddenTolerance = 1e-5;
constexpr double kLogitTolerance = 1e-5;
// One-token-at-a-time decode vs whole-prompt forward.
constexpr double kIncrementalTolerance = 1e-4;

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

std::vector<float> ReadFloats(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
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

std::size_t Argmax(const float* x, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(x, x + n) - x);
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_NPU_QWEN35_2B_GGUF");
  const char* oracle = std::getenv("GUFO_NPU_QWEN35_2B_ORACLE");
  if (model == nullptr || oracle == nullptr || *model == '\0' ||
      *oracle == '\0') {
    std::puts(
        "SKIP: set GUFO_NPU_QWEN35_2B_GGUF and GUFO_NPU_QWEN35_2B_ORACLE");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(model);
  const gufo_npu::qwen35::Reference reference(*file);
  const auto& cfg = reference.config();
  const auto vocab = static_cast<std::size_t>(cfg.vocab);
  const auto hidden = static_cast<std::size_t>(cfg.hidden);
  int failures = 0;

  for (const char* name : {"chat", "code", "prose"}) {
    const fs::path dir = oracle;
    const auto ids = ReadIds(dir / (std::string(name) + ".ids.json"));
    const auto want_logits =
        ReadFloats(dir / (std::string(name) + ".logits.f32"));
    const auto want_hidden =
        ReadFloats(dir / (std::string(name) + ".hidden.f32"));
    const std::size_t n = ids.size();
    if (n == 0 || want_logits.size() != n * vocab ||
        want_hidden.size() != (cfg.layers + 1) * n * hidden) {
      std::fprintf(stderr, "FAIL: %s: oracle files have unexpected sizes\n",
                   name);
      return 1;
    }

    auto state = reference.NewState();
    const auto start = std::chrono::steady_clock::now();
    const auto out = reference.Forward(ids, state, true, true);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();

    // Hidden states: index 0 is the embeddings, i is layer i-1's output.
    // HF's last entry may be after the final norm, so it is reported only.
    double worst_hidden = 0;
    int worst_layer = 0;
    for (int layer = 0; layer <= cfg.layers; ++layer) {
      const std::size_t offset = static_cast<std::size_t>(layer) * n * hidden;
      const double err =
          RelativeL2(&out.hidden[offset], &want_hidden[offset], n * hidden);
      if (layer < cfg.layers && err > worst_hidden) {
        worst_hidden = err;
        worst_layer = layer;
      }
      if (layer == cfg.layers) {
        std::printf("  %s: last hidden entry rel err %.2e (informational)\n",
                    name, err);
      }
    }

    const double logit_err =
        RelativeL2(out.logits.data(), want_logits.data(), n * vocab);
    std::size_t argmax_match = 0;
    for (std::size_t t = 0; t < n; ++t) {
      argmax_match += Argmax(&out.logits[t * vocab], vocab) ==
                      Argmax(&want_logits[t * vocab], vocab);
    }

    // Incremental decode: feed the prompt one token at a time.
    auto step_state = reference.NewState();
    std::vector<float> last;
    for (const int id : ids) {
      last = reference.Forward({id}, step_state).logits;
    }
    const double step_err =
        RelativeL2(last.data(), &out.logits[(n - 1) * vocab], vocab);

    const bool ok = worst_hidden < kHiddenTolerance &&
                    logit_err < kLogitTolerance && argmax_match == n &&
                    step_err < kIncrementalTolerance;
    std::printf(
        "%s %s: %zu tokens in %.1f s; worst hidden rel err %.2e (layer %d); "
        "logits rel err %.2e; argmax %zu/%zu; incremental rel err %.2e\n",
        ok ? "ok  " : "FAIL", name, n, seconds, worst_hidden, worst_layer,
        logit_err, argmax_match, n, step_err);
    failures += ok ? 0 : 1;
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
