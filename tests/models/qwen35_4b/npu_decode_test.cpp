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
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/q4nx_projections.hpp"
#include "models/qwen35_4b/npu_decoder.hpp"

// Qwen3.5-4B decoded on the NPU (qwen35_4b::NpuDecoder on its own decode
// program), teacher-forced on the oracle prompts, against the CPU decoder
// over the same q4nx weights (kernel error only). Agreement with llama.cpp
// is reported (it includes requantization).
//   GUFO_NPU_QWEN35_4B_GGUF    UD-Q4_K_XL GGUF of Qwen3.5-4B
//   GUFO_NPU_QWEN35_4B_ORACLE  llama.cpp oracle output directory
// Exits 77 (skip) when either is unset or there is no NPU.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;
namespace qwen35_4b = gufo_npu::qwen35_4b;

// NPU vs the same q4nx weights on the CPU: top-1 agreement over all prompts,
// and logits per prompt (the gap is mostly the bf16 rounding of each GEMV's
// input).
constexpr double kMinArgmaxAgreement = 0.95;
constexpr double kMaxLogitError = 1e-2;

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

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen35_4b_npu_decode_test <kernel-dir>\n");
    return 2;
  }
  const char* model = std::getenv("GUFO_NPU_QWEN35_4B_GGUF");
  const char* oracle = std::getenv("GUFO_NPU_QWEN35_4B_ORACLE");
  if (model == nullptr || oracle == nullptr || *model == '\0' ||
      *oracle == '\0') {
    std::puts(
        "SKIP: set GUFO_NPU_QWEN35_4B_GGUF and GUFO_NPU_QWEN35_4B_ORACLE");
    return 77;
  }
  if (!fs::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(model);
  const auto start = std::chrono::steady_clock::now();
  qwen35_4b::NpuDecoder npu(*file, argv[1]);
  std::printf(
      "packed and uploaded the NPU decoder in %.1f s\n",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
  const qwen35::Q4nxProjections projections(*file,
                                            gufo_npu::qwen35_4b::kQ4nxFit);
  qwen35::Decoder cpu(*file, projections);
  const auto vocab = static_cast<std::size_t>(npu.config().vocab);

  int failures = 0;
  std::size_t matched = 0;
  std::size_t positions = 0;
  for (const char* name : {"chat", "code", "prose", "long"}) {
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
    double diff = 0;
    double norm = 0;
    std::size_t same = 0;
    std::size_t same_llama = 0;
    for (std::size_t t = 0; t < ids.size(); ++t) {
      const auto& a = npu.Step(ids[t], npu_state);
      const auto& b = cpu.Step(ids[t], cpu_state);
      for (std::size_t i = 0; i < vocab; ++i) {
        const double e = static_cast<double>(a[i]) - b[i];
        diff += e * e;
        norm += static_cast<double>(b[i]) * b[i];
      }
      const std::size_t top = Argmax(a.data(), vocab);
      same += top == Argmax(b.data(), vocab);
      same_llama += top == Argmax(&llama[t * vocab], vocab);
    }
    const double err = std::sqrt(diff / norm);
    matched += same;
    positions += ids.size();
    const bool ok = err < kMaxLogitError;
    failures += ok ? 0 : 1;
    std::printf(
        "%s %s: %zu tokens; vs q4nx on CPU: argmax %zu/%zu, logits rel err "
        "%.2e | NPU argmax vs llama.cpp: %zu/%zu\n",
        ok ? "ok  " : "FAIL", name, ids.size(), same, ids.size(), err,
        same_llama, ids.size());
  }
  // Speculative decoding (MTP draft, 2-token verify) must give exactly the
  // plain greedy output.
  {
    const auto prompt = ReadIds(fs::path(oracle) / "chat.ids.json");
    const std::vector<int> stop;  // none: always max_tokens
    constexpr int kTokens = 48;
    qwen35::SpeculativeStats plain_stats;
    qwen35::SpeculativeStats spec_stats;
    const auto plain = qwen35_4b::SpeculativeGenerate(
        npu, prompt, kTokens, stop, false, {}, &plain_stats);
    const auto spec = qwen35_4b::SpeculativeGenerate(npu, prompt, kTokens, stop,
                                                     true, {}, &spec_stats);
    const bool same = plain == spec;
    failures += same ? 0 : 1;
    std::printf(
        "%s speculative decode: %zu tokens %s plain greedy; %d steps, %d "
        "drafts accepted (%.0f%%)\n",
        same ? "ok  " : "FAIL", spec.size(),
        same ? "identical to" : "DIFFER from", spec_stats.steps,
        spec_stats.accepted,
        spec_stats.steps ? 100.0 * spec_stats.accepted / spec_stats.steps
                         : 0.0);
  }
  const double total = static_cast<double>(matched) / positions;
  const bool agree_ok = total >= kMinArgmaxAgreement;
  std::printf("%s argmax NPU vs q4nx on CPU: %zu/%zu (%.1f%%, need %.0f%%)\n",
              agree_ok ? "ok  " : "FAIL", matched, positions, 100 * total,
              100 * kMinArgmaxAgreement);
  failures += agree_ok ? 0 : 1;
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
