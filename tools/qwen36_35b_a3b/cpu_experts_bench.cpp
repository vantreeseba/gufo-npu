// One Qwen3.6-35B-A3B MoE layer of the prefill's CPU expert step
// (src/models/qwen36_35b_a3b/cpu_experts.hpp) on random q4nx experts in the
// decoder's arena layout (256 routed + 1 shared), routing skewed like real
// prompts (docs/research/NPU-MOE-PREFILL.md: a layer's busiest expert takes
// ~11x the mean): per token count, the mean time of a layer's expert step.
//
//   qwen36_cpu_experts_bench [tokens,...] [workers]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/burst_pool.hpp"
#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "models/qwen36_35b_a3b/cpu_experts.hpp"
#include "npu/gemv.hpp"

namespace {

namespace moe = gufo_npu::qwen36_35b_a3b;
using gufo_npu::BurstPool;
using gufo_npu::npu::ExpertArena;

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kRouted = 256;
constexpr std::size_t kUsed = 8;

std::vector<int> ParseInts(const char* s) {
  std::vector<int> out;
  std::string item;
  for (const char* p = s;; ++p) {
    if (*p == ',' || *p == '\0') {
      if (!item.empty()) {
        out.push_back(std::atoi(item.c_str()));
      }
      item.clear();
      if (*p == '\0') {
        break;
      }
    } else {
      item += *p;
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<int> counts =
      argc > 1 ? ParseInts(argv[1]) : std::vector<int>{125, 1024, 4096, 8192};
  const std::size_t workers =
      argc > 2 ? static_cast<std::size_t>(std::atoi(argv[2]))
               : std::max<unsigned>(std::thread::hardware_concurrency() / 2, 1) -
                     1;
  const ExpertArena layout{kRouted + 1};
  // Random bytes are valid q4nx (any sc, mn and code); keep d and dmin
  // small and finite so the sums stay finite.
  std::vector<std::uint8_t> arena(layout.bytes());
  std::mt19937 rng(3);
  for (std::size_t i = 0; i < arena.size(); i += 4) {
    const std::uint32_t v = rng();
    std::memcpy(arena.data() + i, &v, std::min<std::size_t>(4, arena.size() - i));
  }
  const std::uint16_t small = gufo_npu::quant::FloatToBf16(1e-4f);
  for (std::size_t b = 0; b < arena.size() / gufo_npu::q4nx::kBlockBytes; ++b) {
    std::uint8_t* block = arena.data() + b * gufo_npu::q4nx::kBlockBytes;
    for (std::size_t m = 0; m < 64; ++m) {  // d[32], dmin[32]
      std::memcpy(block + m * 2, &small, 2);
    }
  }
  BurstPool pool(workers);
  const BurstPool::Burst burst(pool);
  auto step = moe::MakeCpuExperts(pool, {arena.data()}, kRouted + 1, 16384);
  std::printf("fast path %s, %zu threads\n", moe::CpuExpertsFast() ? "on" : "off",
              pool.size());
  std::printf("%7s %10s %8s %10s %9s\n", "tokens", "ms/layer", "active",
              "busiest", "TFLOPS");
  for (const int count : counts) {
    // Zipf-like expert popularity, shuffled per token's draw.
    std::vector<double> popularity(kRouted);
    for (std::size_t e = 0; e < kRouted; ++e) {
      popularity[e] = 1.0 / std::pow(static_cast<double>(e + 1), 0.75);
    }
    std::shuffle(popularity.begin(), popularity.end(), rng);
    std::discrete_distribution<int> pick(popularity.begin(), popularity.end());
    std::vector<std::vector<int>> tokens(kRouted + 1);
    std::vector<std::vector<float>> weights(kRouted + 1);
    for (int t = 0; t < count; ++t) {
      std::vector<int> chosen;
      while (chosen.size() < kUsed) {
        const int e = pick(rng);
        if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
          chosen.push_back(e);
        }
      }
      for (const int e : chosen) {
        tokens[e].push_back(t);
        weights[e].push_back(0.125f);
      }
      tokens[kRouted].push_back(t);
      weights[kRouted].push_back(0.5f);
    }
    std::size_t active = 0;
    std::size_t busiest = 0;
    for (std::size_t e = 0; e < kRouted; ++e) {
      active += tokens[e].empty() ? 0 : 1;
      busiest = std::max(busiest, tokens[e].size());
    }
    const moe::ExpertRouting routing{tokens, weights};
    std::vector<std::uint16_t> x(static_cast<std::size_t>(count) * kHidden,
                                 gufo_npu::quant::FloatToBf16(0.5f));
    std::vector<float> h(static_cast<std::size_t>(count) * kHidden, 0.0f);
    step->Run(0, routing, x.data(), h.data());  // warm-up
    const int reps = count <= 1024 ? 5 : 2;
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
      step->Run(0, routing, x.data(), h.data());
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count() /
                      reps;
    const double flops = 2.0 * count * (kUsed + 1) * 3.0 * 512 * kHidden;
    std::printf("%7d %10.2f %8zu %10zu %9.2f\n", count, ms, active, busiest,
                flops / (ms * 1e-3) / 1e12);
    std::fflush(stdout);
  }
  return 0;
}
