#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/reference.hpp"
#include "models/qwen35/rope.hpp"

// The fp32 CPU reference on image prompts against the HF transformers oracle
// (Qwen3_5ForConditionalGeneration in fp32): the oracle's vision features
// replace the image pads' embedding rows, the prompt runs at the oracle's
// (t, h, w) MRoPE positions, and the continuation runs as plain tokens at
// KV index + rope_delta. Teacher-forced: every logits row from the prompt's
// last token on must have the oracle's argmax.
//   GUFO_NPU_QWEN35_2B_GGUF       BF16 GGUF of Qwen3.5-2B (same weights as HF)
//   GUFO_NPU_QWEN35_2B_VL_ORACLE  image-prompt oracle directory
// Exits 77 (skip) when either is unset.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;
using nlohmann::json;

// Relative L2 error of the logits rows against the oracle.
constexpr double kLogitTolerance = 1e-4;

json ReadJson(const fs::path& path) {
  std::ifstream in(path);
  return json::parse(in);
}

template<typename T>
std::vector<T> ReadBinary(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
  std::vector<T> out(bytes.size() / sizeof(T));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
  return out;
}

double RelativeL2(const float* a, const float* b, std::size_t n) {
  double diff = 0;
  double norm = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    diff += d * d;
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
  const char* oracle = std::getenv("GUFO_NPU_QWEN35_2B_VL_ORACLE");
  if (model == nullptr || oracle == nullptr || *model == '\0' ||
      *oracle == '\0') {
    std::puts(
        "SKIP: set GUFO_NPU_QWEN35_2B_GGUF and GUFO_NPU_QWEN35_2B_VL_ORACLE");
    return 77;
  }
  const fs::path dir = oracle;
  const json manifest = ReadJson(dir / "manifest.json");
  const int image_pad = manifest.at("token_ids").at("image_pad").get<int>();
  const auto file = gufo_npu::gguf::File::Open(model);
  const qwen35::Reference reference(*file);
  const auto& cfg = reference.config();
  const auto vocab = static_cast<std::size_t>(cfg.vocab);
  const auto hidden = static_cast<std::size_t>(cfg.hidden);
  std::printf("rope.dimension_sections (t, h, w): %d %d %d\n",
              cfg.rope_sections[0], cfg.rope_sections[1], cfg.rope_sections[2]);
  int failures = 0;
  if (cfg.rope_sections != std::array<int, 3>{11, 11, 10}) {
    std::fprintf(stderr, "FAIL: expected MRoPE sections 11 11 10\n");
    ++failures;
  }

  for (const auto& [name, entry] : manifest.at("cases").items()) {
    auto path = [&](const char* suffix) { return dir / (name + suffix); };
    const json ids_json = ReadJson(path(".ids.json"));
    const auto ids = ids_json.at("ids").get<std::vector<int>>();
    const int prompt_len = ids_json.at("prompt_len").get<int>();
    const int n = static_cast<int>(ids.size());
    const auto positions = ReadBinary<std::int32_t>(path(".positions.i32"));
    const auto features = ReadBinary<float>(path(".features.f32"));
    const auto want_logits = ReadBinary<float>(path(".logits.f32"));
    const auto want_hidden = ReadBinary<float>(path(".hidden.f32"));
    const int lo = entry.at("logits_rows")[0].get<int>();
    const int hi = entry.at("logits_rows")[1].get<int>();
    const int want_delta = entry.at("rope_delta").get<int>();
    std::vector<qwen35::ImageGrid> grids;
    for (const auto& g : ReadJson(path(".grid.json"))) {
      grids.push_back({g[0].get<int>(), g[1].get<int>(), g[2].get<int>()});
    }
    if (positions.size() != 3 * static_cast<std::size_t>(n) ||
        lo != prompt_len - 1 || hi != n ||
        want_logits.size() != static_cast<std::size_t>(hi - lo) * vocab ||
        features.size() % hidden != 0 ||
        want_hidden.size() !=
            static_cast<std::size_t>(cfg.layers + 1) * n * hidden) {
      std::fprintf(stderr, "FAIL: %s: oracle files have unexpected sizes\n",
                   name.c_str());
      return 1;
    }

    // The oracle's positions, and ours from the ids and grids (HF's
    // get_rope_index).
    std::vector<qwen35::RopePosition> want_positions(n);
    for (int t = 0; t < n; ++t) {
      want_positions[t] = {positions[t], positions[n + t],
                           positions[2 * n + t]};
    }
    const auto ours = qwen35::ImagePromptPositions(ids, image_pad, grids);
    const bool positions_ok =
        ours.positions == want_positions && ours.rope_delta == want_delta;

    // The prompt with the oracle's features and positions; the image rows,
    // one EmbeddingRows per run of image pads.
    qwen35::PromptInput input;
    input.tokens.assign(ids.begin(), ids.begin() + prompt_len);
    input.positions.assign(want_positions.begin(),
                           want_positions.begin() + prompt_len);
    std::size_t used = 0;
    for (int t = 0; t < prompt_len;) {
      if (ids[t] != image_pad) {
        ++t;
        continue;
      }
      int run = 0;
      while (t + run < prompt_len && ids[t + run] == image_pad) {
        ++run;
      }
      const auto begin = features.begin() + static_cast<std::ptrdiff_t>(used);
      used += static_cast<std::size_t>(run) * hidden;
      if (used > features.size()) {
        break;
      }
      input.embeddings.push_back(
          {t, std::vector<float>(begin, begin + run * hidden)});
      t += run;
    }
    if (used != features.size()) {
      std::fprintf(stderr, "FAIL: %s: features do not cover the image pads\n",
                   name.c_str());
      return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    auto state = reference.NewState();
    const auto prompt_out = reference.ForwardPrompt(input, state, true, true);
    const int delta = state.rope_delta;
    // The continuation, teacher-forced, at KV index + rope_delta.
    const std::vector<int> continuation(ids.begin() + prompt_len, ids.end());
    const auto cont_out = reference.Forward(continuation, state);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();

    // Logits rows lo .. hi - 1: the prompt's last, then the continuation's.
    std::vector<float> got(want_logits.size());
    std::copy_n(prompt_out.logits.end() - static_cast<std::ptrdiff_t>(vocab),
                vocab, got.begin());
    std::copy(cont_out.logits.begin(), cont_out.logits.end(),
              got.begin() + static_cast<std::ptrdiff_t>(vocab));
    const std::size_t rows = static_cast<std::size_t>(hi - lo);
    std::size_t match = 0;
    double max_diff = 0;
    for (std::size_t r = 0; r < rows; ++r) {
      match += Argmax(&got[r * vocab], vocab) ==
               Argmax(&want_logits[r * vocab], vocab);
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
      max_diff = std::max(
          max_diff, std::fabs(static_cast<double>(got[i]) - want_logits[i]));
    }
    const double logit_err =
        RelativeL2(got.data(), want_logits.data(), got.size());

    // Hidden states over the prompt rows (informational): entry 0 is the
    // embeddings with the image rows, entry i layer i-1's output.
    double worst_hidden = 0;
    int worst_layer = 0;
    for (int layer = 0; layer < cfg.layers; ++layer) {
      const std::size_t ours_at =
          static_cast<std::size_t>(layer) * prompt_len * hidden;
      const std::size_t want_at = static_cast<std::size_t>(layer) * n * hidden;
      const double err =
          RelativeL2(&prompt_out.hidden[ours_at], &want_hidden[want_at],
                     static_cast<std::size_t>(prompt_len) * hidden);
      if (err > worst_hidden) {
        worst_hidden = err;
        worst_layer = layer;
      }
    }

    const bool ok = positions_ok && delta == want_delta && match == rows &&
                    logit_err < kLogitTolerance;
    std::printf(
        "%s %s: %d tokens (%d prompt, %zu image rows, rope_delta %d) in "
        "%.1f s; positions %s; argmax %zu/%zu; logits rel err %.2e, max abs "
        "diff %.2e; prompt hidden worst rel err %.2e (entry %d)\n",
        ok ? "ok  " : "FAIL", name.c_str(), n, prompt_len,
        features.size() / hidden, delta, seconds,
        positions_ok ? "match" : "DIFFER", match, rows, logit_err, max_diff,
        worst_hidden, worst_layer);
    failures += ok ? 0 : 1;
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
