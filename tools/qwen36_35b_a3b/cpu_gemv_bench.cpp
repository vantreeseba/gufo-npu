// Times the CPU q4nx GEMV (src/models/qwen36_35b_a3b/cpu_gemv.hpp) at the
// 35B's mixer output projection shape (2048 rows x 4096 inputs, 4.85 MB),
// for 1 and 2 tokens, on a pool sized like the NPU decoder's routing pool.
// It rotates over 40 copies of the matrix (one per layer, 194 MB in all),
// so each run reads its weights from DRAM as the decoder would. No NPU.
//
// usage: qwen36_cpu_gemv_bench [runs per token count] [--portable]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/cpus.hpp"
#include "core/q4nx.hpp"
#include "core/thread_pool.hpp"
#include "models/qwen36_35b_a3b/cpu_gemv.hpp"

int main(int argc, char** argv) {
  long runs = 2000;
  bool portable = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--portable") {
      portable = true;
    } else {
      runs = std::atol(argv[i]);
    }
  }
  constexpr std::size_t kRows = 2048;
  constexpr std::size_t kDim = 4096;
  constexpr std::size_t kLayers = 40;
  const std::size_t workers =
      std::min<std::size_t>(7, gufo_npu::HalfCpuWorkers());
  gufo_npu::SpinPool pool(workers);

  std::mt19937 rng(3);
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> act(0.0f, 1.0f);
  std::vector<float> w(kRows * kDim);
  for (auto& v : w) {
    v = weight(rng);
  }
  const auto m = gufo_npu::q4nx::PackFloat(w.data(), kRows, kDim,
                                           gufo_npu::q4nx::Fit::kMinMax);
  std::vector<std::vector<std::uint8_t>> copies(kLayers, m.bytes);
  std::vector<gufo_npu::qwen36_35b_a3b::CpuGemv> gemvs;
  gemvs.reserve(kLayers);
  for (auto& c : copies) {
    gemvs.emplace_back(pool, c.data(), kRows, kDim, portable);
  }
  std::vector<float> x(2 * kDim);
  for (auto& v : x) {
    v = act(rng);
  }
  std::vector<float> y(2 * kRows);
  std::printf("%s path, %zu workers + the caller, %zu matrices of %.2f MB\n",
              gemvs[0].fast() ? "AVX-512 BF16" : "portable", workers, kLayers,
              m.bytes.size() / 1e6);
  for (int tokens = 1; tokens <= 2; ++tokens) {
    for (std::size_t i = 0; i < 2 * kLayers; ++i) {  // warm up
      gemvs[i % kLayers].Run(x.data(), kDim, y.data(), kRows, tokens);
    }
    std::vector<double> us(static_cast<std::size_t>(runs));
    for (long i = 0; i < runs; ++i) {
      const auto start = std::chrono::steady_clock::now();
      gemvs[static_cast<std::size_t>(i) % kLayers].Run(x.data(), kDim,
                                                       y.data(), kRows, tokens);
      us[static_cast<std::size_t>(i)] =
          std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - start)
              .count();
    }
    double mean = 0;
    for (const double v : us) {
      mean += v;
    }
    mean /= static_cast<double>(runs);
    std::sort(us.begin(), us.end());
    const auto at = [&](double q) {
      return us[static_cast<std::size_t>(q * static_cast<double>(runs - 1))];
    };
    std::printf(
        "%d token(s): mean %.1f us, median %.1f, p10 %.1f, p90 %.1f, max "
        "%.1f (%.1f GB/s at the mean)\n",
        tokens, mean, at(0.5), at(0.1), at(0.9), us.back(),
        m.bytes.size() / mean / 1e3);
  }
  return 0;
}
