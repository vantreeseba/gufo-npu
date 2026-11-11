// Qwen3-VL-Embedding-2B's ViT attention program on the NPU (npu::
// VitAttention) against a double-precision softmax, at patch counts from an
// oracle card's to the embedder's cap, most of them padded (so the key masks
// are exercised). Inputs are random q, k, v in FullAttention's layout (q
// scaled by 1/8), the scores' spread about that of the ViT's. "double" is
// the program's error against float64, "cpu" against FullAttention (fp32,
// ops.hpp), the path it replaces. Pass: relative L2 error against float64
// <= 3e-3 (rounding the output to bf16 alone gives ~1.6e-3). Every length
// is timed warm. Exits 77 without an NPU.
#include "models/qwen3vl_embed_2b/npu/vit_attention.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <random>
#include <vector>

#include "core/parallel.hpp"
#include "core/simd.hpp"
#include "models/qwen3vl_embed_2b/ops.hpp"

namespace {

namespace e = gufo_npu::qwen3vl_embed_2b;
namespace npu = e::npu;
namespace simd = gufo_npu::simd;
using Clock = std::chrono::steady_clock;

constexpr double kTolerance = 3e-3;
constexpr int kHeads = npu::kVitAttnHeads;
constexpr int kHd = npu::kVitAttnHeadDim;
constexpr int kDim = kHeads * kHd;

const e::ParallelRun kRun = [](std::size_t n,
                               const std::function<void(std::size_t)>& fn) {
  gufo_npu::ParallelFor(n, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      fn(i);
    }
  });
};

struct Errors {
  double vs_double = 0;
  double vs_cpu = 0;
  double bf16 = 0;  // rounding the double output to bf16
};

// The program's output for `rows` queries of every head (sampled when there
// are many) against float64 and FullAttention's.
Errors Compare(const npu::VitAttention& attn, const std::vector<float>& qkv,
               const std::vector<float>& cpu, int n) {
  std::vector<int> queries;
  const int samples = std::min(n, 96);
  for (int i = 0; i < samples; ++i) {
    queries.push_back(samples == n ? i
                                   : static_cast<int>(static_cast<long>(n - 1) *
                                                      i / (samples - 1)));
  }
  std::vector<double> den(queries.size(), 0.0);
  std::vector<double> err(queries.size()), err_cpu(queries.size()),
      err_bf(queries.size());
  gufo_npu::ParallelFor(queries.size(), [&](std::size_t begin,
                                            std::size_t end) {
    std::vector<std::uint16_t> got(kDim);
    std::vector<double> w(n);
    for (std::size_t s = begin; s < end; ++s) {
      const int query = queries[s];
      attn.Output(query, got.data());
      double e2 = 0, c2 = 0, b2 = 0, r2 = 0;
      for (int h = 0; h < kHeads; ++h) {
        const float* q =
            qkv.data() + static_cast<std::size_t>(query) * 3 * kDim + h * kHd;
        double mx = -1e300;
        for (int j = 0; j < n; ++j) {
          const float* k = qkv.data() + static_cast<std::size_t>(j) * 3 * kDim +
                           kDim + h * kHd;
          double dot = 0;
          for (int c = 0; c < kHd; ++c) {
            dot += static_cast<double>(q[c]) * k[c];
          }
          mx = std::max(mx, w[j] = dot);
        }
        double sum = 0;
        for (int j = 0; j < n; ++j) {
          sum += (w[j] = std::exp(w[j] - mx));
        }
        for (int c = 0; c < kHd; ++c) {
          double want = 0;
          for (int j = 0; j < n; ++j) {
            want += w[j] * qkv[static_cast<std::size_t>(j) * 3 * kDim +
                               2 * kDim + h * kHd + c];
          }
          want /= sum;
          const double g = simd::FromBf16(got[h * kHd + c]);
          const double cp =
              cpu[static_cast<std::size_t>(query) * kDim + h * kHd + c];
          const double bf =
              simd::FromBf16(simd::ToBf16(static_cast<float>(want)));
          e2 += (g - want) * (g - want);
          c2 += (g - cp) * (g - cp);
          b2 += (bf - want) * (bf - want);
          r2 += want * want;
        }
      }
      err[s] = e2;
      err_cpu[s] = c2;
      err_bf[s] = b2;
      den[s] = r2;
    }
  });
  double e2 = 0, c2 = 0, b2 = 0, r2 = 0;
  for (std::size_t s = 0; s < queries.size(); ++s) {
    e2 += err[s];
    c2 += err_cpu[s];
    b2 += err_bf[s];
    r2 += den[s];
  }
  return {std::sqrt(e2 / r2), std::sqrt(c2 / r2), std::sqrt(b2 / r2)};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr,
                 "usage: qwen3vl_embed_2b_vit_attention_test <kernel-dir>\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  npu::VitAttention attn(argv[1]);
  std::mt19937 rng(17);
  int failures = 0;
  // Patch counts: oracle cards (24, 48), partial groups and blocks, the
  // 640 x 480 card (1,200), a 3,000-patch image and the cap; spread is the
  // scores' standard deviation (natural log units).
  const struct {
    int patches;
    double spread;
  } cases[] = {{24, 1.0},   {48, 3.0},   {300, 2.0},  {1200, 1.0},
               {1200, 4.0}, {3000, 2.0}, {7200, 1.0}, {7200, 3.0}};
  for (const auto& c : cases) {
    const int n = c.patches;
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> qkv(static_cast<std::size_t>(n) * 3 * kDim);
    // q . k / 8 has standard deviation `spread` for q, k ~ N(0, spread).
    const float s = static_cast<float>(std::sqrt(c.spread));
    for (int r = 0; r < n; ++r) {
      float* row = qkv.data() + static_cast<std::size_t>(r) * 3 * kDim;
      for (int i = 0; i < kDim; ++i) {
        row[i] = g(rng) * s / 8.0f;  // q, scaled as FullAttention takes it
        row[kDim + i] = g(rng) * s;
        row[2 * kDim + i] = g(rng);
      }
    }
    std::vector<float> cpu(static_cast<std::size_t>(n) * kDim);
    e::FullAttention(kHeads, kHd, n, qkv.data(), cpu.data(), kRun);

    attn.Reset(n);
    attn.Run(qkv.data(), kRun);
    const Errors err = Compare(attn, qkv, cpu, n);
    for (int i = 0; i < 3; ++i) {  // warm-up
      attn.Run(qkv.data(), kRun);
    }
    const auto before = attn.profile();
    constexpr int kRuns = 5;
    const auto start = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
      attn.Run(qkv.data(), kRun);
    }
    const double ms =
        std::chrono::duration<double>(Clock::now() - start).count() * 1e3 /
        kRuns;
    const auto& after = attn.profile();
    const bool ok = err.vs_double <= kTolerance;
    failures += ok ? 0 : 1;
    std::printf(
        "%s %s (%d patches, spread %.0f): double %.2e (bf16 rounding %.2e), "
        "cpu %.2e; %.2f ms per call (pack %.2f, npu %.2f)\n",
        ok ? "ok  " : "FAIL",
        npu::VitAttnSequence(npu::VitAttnLength(n)).c_str(), n, c.spread,
        err.vs_double, err.bf16, err.vs_cpu, ms,
        (after.pack_seconds - before.pack_seconds) * 1e3 / kRuns,
        (after.npu_seconds - before.npu_seconds) * 1e3 / kRuns);
  }
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS: the ViT attention program matches float64");
  return EXIT_SUCCESS;
}
