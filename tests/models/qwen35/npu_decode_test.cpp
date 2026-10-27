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
#include "models/qwen35/q4nx_projections.hpp"
#include "models/qwen35/reference.hpp"

// Qwen3.5-2B with its matrix products on the NPU, teacher-forced on the
// oracle prompts. The NPU run must match the same q4nx weights run on the
// CPU (kernel error only); agreement with HF and with the GGUF reference is
// reported (it includes requantization).
//   GUFO_NPU_QWEN35_2B_Q4_GGUF  UD-Q4_K_XL GGUF of Qwen3.5-2B
//   GUFO_NPU_QWEN35_2B_ORACLE   oracle output directory
// Exits 77 (skip) when either is unset or there is no NPU.
namespace {

namespace fs = std::filesystem;
namespace qwen35 = gufo_npu::qwen35;

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

double RelativeL2(const std::vector<float>& a, const std::vector<float>& b) {
  double diff = 0;
  double norm = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    diff += d * d;
    norm += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(diff / std::max(norm, 1e-30));
}

std::size_t Matches(const std::vector<float>& a, const std::vector<float>& b,
                    std::size_t vocab) {
  const std::size_t n = a.size() / vocab;
  std::size_t same = 0;
  for (std::size_t t = 0; t < n; ++t) {
    const auto* pa = &a[t * vocab];
    const auto* pb = &b[t * vocab];
    same += (std::max_element(pa, pa + vocab) - pa) ==
            (std::max_element(pb, pb + vocab) - pb);
  }
  return same;
}

double Agreement(const std::vector<float>& a, const std::vector<float>& b,
                 std::size_t vocab) {
  return static_cast<double>(Matches(a, b, vocab)) /
         static_cast<double>(a.size() / vocab);
}

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen35_npu_decode_test <kernel-dir>\n");
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
  const qwen35::Reference reference(*file);
  const auto vocab = static_cast<std::size_t>(reference.config().vocab);

  auto start = std::chrono::steady_clock::now();
  const qwen35::Q4nxProjections npu(
      *file, qwen35::Q4nxProjections::Backend::kNpu, argv[1]);
  const qwen35::Q4nxProjections cpu(*file,
                                    qwen35::Q4nxProjections::Backend::kCpu);
  std::printf("packed and uploaded q4nx weights in %.1f s\n",
              SecondsSince(start));

  int failures = 0;
  std::size_t matched = 0;
  std::size_t positions = 0;
  for (const char* name : {"chat", "code", "prose"}) {
    const fs::path dir = oracle;
    const auto ids = ReadIds(dir / (std::string(name) + ".ids.json"));
    const auto hf = ReadFloats(dir / (std::string(name) + ".logits.f32"));
    if (ids.empty() || hf.size() != ids.size() * vocab) {
      std::fprintf(stderr, "FAIL: %s: oracle files have unexpected sizes\n",
                   name);
      return 1;
    }
    auto run = [&](const qwen35::Projections* p) {
      auto state = reference.NewState();
      return reference.Forward(ids, state, true, false, p).logits;
    };
    start = std::chrono::steady_clock::now();
    const auto on_npu = run(&npu);
    const double npu_seconds = SecondsSince(start);
    const auto on_cpu = run(&cpu);
    const auto gguf = run(nullptr);

    const double agree = Agreement(on_npu, on_cpu, vocab);
    matched += Matches(on_npu, on_cpu, vocab);
    positions += ids.size();
    const double err = RelativeL2(on_npu, on_cpu);
    const bool ok = err < kMaxLogitError;
    failures += ok ? 0 : 1;
    std::printf(
        "%s %s: %zu tokens (%.1f s on NPU); vs q4nx on CPU: argmax %.1f%%, "
        "logits rel err %.2e | argmax vs HF: NPU %.1f%%, q4nx CPU %.1f%%, "
        "GGUF CPU %.1f%%\n",
        ok ? "ok  " : "FAIL", name, ids.size(), npu_seconds, 100 * agree, err,
        100 * Agreement(on_npu, hf, vocab), 100 * Agreement(on_cpu, hf, vocab),
        100 * Agreement(gguf, hf, vocab));
  }
  const double total = static_cast<double>(matched) / positions;
  const bool agree_ok = total >= kMinArgmaxAgreement;
  std::printf("%s argmax NPU vs q4nx on CPU: %zu/%zu (%.1f%%, need %.0f%%)\n",
              agree_ok ? "ok  " : "FAIL", matched, positions, 100 * total,
              100 * kMinArgmaxAgreement);
  failures += agree_ok ? 0 : 1;
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
