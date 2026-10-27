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
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/moe.hpp"
#include "models/qwen35/npu_decoder.hpp"
#include "models/qwen35/q4nx_projections.hpp"

// Qwen3.6-35B-A3B decoded on the NPU (NpuDecoder: q4nx GEMVs, expert gathers,
// the router on the CPU), teacher forced over the oracle prompts, against the
// CPU decoder over the same q4nx weights (Q4nxProjections): the NPU's own
// error, mostly the bf16 rounding of each call's input.
//
// Top-k routing is discontinuous: where a layer's 8th and 9th experts are
// nearly tied, the NPU's rounding can pick the other one (it does within the
// first few tokens of most prompts), and the two decodes then drift apart.
// So the CPU follows the NPU's routing (FollowRouting) and the logits are
// compared at every position; separately, wherever the NPU chose experts the
// CPU would not have, that must have been a near tie in the CPU's own router
// logits. Top-1 against llama.cpp is reported (it includes requantizing to
// q4nx; QUALITY.md).
//   GUFO_NPU_QWEN36_MODEL   the Qwen3.6-35B-A3B GGUF
//   GUFO_NPU_QWEN36_ORACLE  oracle output directory
// Exits 77 (skip) when either is unset or there is no NPU.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;
using qwen35::Stage;

// docs/models/qwen3.6-35b-a3b/QUALITY.md.
// Logits relative L2 per prompt, every position (routing followed).
constexpr double kLogitTolerance = 1e-2;
// Where the NPU chose other experts than the CPU's own top-k: the largest
// relative probability gap between an expert only the CPU chose and one only
// the NPU chose.
constexpr double kNearTie = 5e-2;
// Top-1 agreement with the CPU over all prompts.
constexpr double kMinArgmaxAgreement = 0.98;

const char* const kPrompts[] = {"chat", "code", "prose", "long"};

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

std::size_t Argmax(const float* x, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(x, x + n) - x);
}

// Passes everything to `inner`, but routes each layer to the experts in
// `routes`: the other experts' router logits drop far below theirs, which
// leaves the chosen experts' logits, and so their renormalized weights,
// untouched. Where the CPU's own top-k differs, records how near a tie it
// was: 1 - exp(l_x - l_y) for the lowest-scored expert x only `routes` has
// and the highest-scored y only the CPU's top-k has.
class FollowRouting : public qwen35::Projections {
public:
  FollowRouting(const qwen35::Projections& inner, const qwen35::Config& c)
      : routes(c.layers), inner_(inner), c_(c) {}

  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override {
    inner_.Apply(stage, layer, x, tokens, y);
    if (stage != Stage::kRouter) {
      return;
    }
    if (tokens != 1) {
      throw std::logic_error("FollowRouting: one token at a time");
    }
    const auto& want = routes.at(static_cast<std::size_t>(layer));
    std::vector<int> own(c_.experts_used);
    std::vector<float> weights(c_.experts_used);
    qwen35::Route(y, c_.experts, c_.experts_used, own.data(), weights.data());
    std::sort(own.begin(), own.end());
    if (own != want) {
      ++differences;
      float lowest_want = 1e30f;
      float highest_own = -1e30f;
      for (const int e : want) {
        if (!std::binary_search(own.begin(), own.end(), e)) {
          lowest_want = std::min(lowest_want, y[e]);
        }
      }
      for (const int e : own) {
        if (!std::binary_search(want.begin(), want.end(), e)) {
          highest_own = std::max(highest_own, y[e]);
        }
      }
      worst_gap = std::max(
          worst_gap,
          1.0 - std::exp(static_cast<double>(lowest_want) - highest_own));
    }
    float floor = 1e30f;
    for (const int e : want) {
      floor = std::min(floor, y[e]);
    }
    for (int e = 0; e < c_.experts; ++e) {
      if (!std::binary_search(want.begin(), want.end(), e)) {
        y[e] = std::min(y[e], floor - 64.0f);  // the shared gate stays
      }
    }
  }

  void ApplyExpert(Stage stage, int layer, int expert, const float* x,
                   int tokens, float* y) const override {
    inner_.ApplyExpert(stage, layer, expert, x, tokens, y);
  }

  // Per layer: the experts to route to (sorted), set before each Step.
  std::vector<std::vector<int>> routes;
  mutable int differences = 0;  // (token, layer) routed unlike the CPU
  mutable double worst_gap = 0;

private:
  const qwen35::Projections& inner_;
  const qwen35::Config& c_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen36_npu_decode_test <kernel-dir>\n");
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
  std::printf(
      "packed and uploaded the NPU decoder in %.1f s\n",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
  const auto& c = npu.config();
  start = std::chrono::steady_clock::now();
  const qwen35::Q4nxProjections projections(*file);
  FollowRouting follow(projections, c);
  qwen35::Decoder cpu(*file, follow);
  std::printf(
      "packed the CPU q4nx weights in %.1f s\n",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
  const auto vocab = static_cast<std::size_t>(c.vocab);

  int failures = 0;
  std::size_t matched = 0;
  std::size_t matched_llama = 0;
  std::size_t positions = 0;
  int differences = 0;
  double worst_gap = 0;
  for (const char* name : kPrompts) {
    const fs::path dir = oracle;
    const auto ids = ReadIds(dir / (std::string(name) + ".ids.json"));
    const auto llama = ReadFloats(dir / (std::string(name) + ".logits.f32"));
    if (ids.empty() || llama.size() != ids.size() * vocab) {
      std::fprintf(stderr, "FAIL: %s: oracle files have unexpected sizes\n",
                   name);
      return 1;
    }
    auto npu_state = npu.NewState();
    auto cpu_state = cpu.NewState();
    follow.differences = 0;
    follow.worst_gap = 0;
    double diff = 0;
    double norm = 0;
    double worst = 0;  // one position
    std::size_t same = 0;
    std::size_t same_llama = 0;
    for (std::size_t t = 0; t < ids.size(); ++t) {
      const auto& a = npu.Step(ids[t], npu_state);
      for (int il = 0; il < c.layers; ++il) {
        auto& route = follow.routes[il];
        route.assign(npu.experts(il), npu.experts(il) + c.experts_used);
        std::sort(route.begin(), route.end());
      }
      const auto& b = cpu.Step(ids[t], cpu_state);
      double d = 0;
      double n = 0;
      for (std::size_t i = 0; i < vocab; ++i) {
        const double e = static_cast<double>(a[i]) - b[i];
        d += e * e;
        n += static_cast<double>(b[i]) * b[i];
      }
      diff += d;
      norm += n;
      worst = std::max(worst, std::sqrt(d / n));
      const std::size_t top = Argmax(a.data(), vocab);
      same += top == Argmax(b.data(), vocab);
      same_llama += top == Argmax(&llama[t * vocab], vocab);
    }
    const double err = std::sqrt(diff / norm);
    const bool ok = err < kLogitTolerance && follow.worst_gap < kNearTie;
    failures += ok ? 0 : 1;
    matched += same;
    matched_llama += same_llama;
    positions += ids.size();
    differences += follow.differences;
    worst_gap = std::max(worst_gap, follow.worst_gap);
    std::printf(
        "%s %s: %zu tokens; vs q4nx on CPU (NPU routing): logits rel err "
        "%.2e (worst position %.2e), argmax %zu/%zu; %d of %zu routings "
        "unlike the CPU's, largest gap %.1e | NPU top-1 vs llama.cpp "
        "%zu/%zu\n",
        ok ? "ok  " : "FAIL", name, ids.size(), err, worst, same, ids.size(),
        follow.differences, ids.size() * c.layers, follow.worst_gap, same_llama,
        ids.size());
  }
  const double total = static_cast<double>(matched) / positions;
  const bool agree_ok = total >= kMinArgmaxAgreement;
  failures += agree_ok ? 0 : 1;
  std::printf(
      "%s argmax NPU vs q4nx on CPU: %zu/%zu (%.1f%%, need %.0f%%); %d "
      "routings unlike the CPU's (largest gap %.1e, need < %.0e); NPU top-1 "
      "vs llama.cpp %zu/%zu (%.1f%%)\n",
      agree_ok ? "ok  " : "FAIL", matched, positions, 100 * total,
      100 * kMinArgmaxAgreement, differences, worst_gap, kNearTie,
      matched_llama, positions, 100.0 * matched_llama / positions);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
