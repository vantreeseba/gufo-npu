#include "npu/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <vector>

#include "core/bfp16.hpp"
#include "core/parallel.hpp"
#include "core/quant.hpp"
#include "models/qwen3vl_embed_2b/npu/embed_gemm.hpp"

// Qwen3-VL-Embedding-2B's GEMM program on the NPU at every compiled shape,
// against a double-precision CPU reference: bf16 weights (as the model's
// GGUF holds them) and bf16 activations. "kernel" compares against the
// bfp16 weights the NPU holds, so it is the kernel's own error (A's bfp16
// conversion inside the mmul, bf16 output); "total" against the bf16
// weights, so it adds the bfp16 rounding of the weights. Pass: total
// relative L2 error < 1e-2. Every shape runs at every chunk size and is
// timed. Exits 77 without an NPU.
namespace {

namespace bfp16 = gufo_npu::bfp16;
namespace npu = gufo_npu::npu;
namespace model = gufo_npu::qwen3vl_embed_2b::npu;
namespace quant = gufo_npu::quant;

constexpr double kTolerance = 1e-2;

struct Shape {
  std::size_t dim;
  std::size_t rows;
};

// Must match GEMM_SHAPES in
// src/models/qwen3vl_embed_2b/kernels/embed_gemm_variants.py: attention
// input (q | k | v), attention output, gate + up, down.
constexpr Shape kShapes[] = {
    {model::kHidden, model::kInRows},
    {model::kQDim, model::kHidden},
    {model::kHidden, 2 * model::kFfn},
    {model::kFfn, model::kHidden},
};

// y = A W^T in double, m x rows.
std::vector<double> Reference(const std::vector<float>& a,
                              const std::vector<float>& w, std::size_t m,
                              std::size_t dim, std::size_t rows) {
  std::vector<double> y(m * rows);
  gufo_npu::ParallelFor(rows, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = begin; r < end; ++r) {
      const float* wr = w.data() + r * dim;
      for (std::size_t t = 0; t < m; ++t) {
        const float* at = a.data() + t * dim;
        double acc = 0;
        for (std::size_t k = 0; k < dim; ++k) {
          acc += static_cast<double>(at[k]) * wr[k];
        }
        y[t * rows + r] = acc;
      }
    }
  });
  return y;
}

double RelativeError(const std::vector<float>& got,
                     const std::vector<double>& want) {
  double err2 = 0;
  double ref2 = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double e = got[i] - want[i];
    err2 += e * e;
    ref2 += want[i] * want[i];
  }
  return std::sqrt(err2 / ref2);
}

int Check(npu::GemmRuntime& rt, const Shape& s, std::mt19937& rng) {
  const std::size_t max_m = npu::kGemmMaxChunk;
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> act(0.0f, 1.0f);
  std::vector<float> w(s.rows * s.dim);
  for (auto& v : w) {
    v = weight(rng);
  }
  // The GGUF's bf16 weights.
  std::vector<float> q(w.size());
  for (std::size_t i = 0; i < w.size(); ++i) {
    q[i] = quant::Bf16ToFloat(quant::FloatToBf16(w[i]));
  }
  npu::Bfp16Gemm gemm(rt, q.data(), s.rows, s.dim);
  const auto held = bfp16::UnpackGemmB(
      bfp16::PackGemmB(q.data(), s.rows, s.dim), s.rows, s.dim);

  // One operand for every chunk size: a chunk of m rows is its first m.
  npu::GemmBuffer a(rt, gemm.a_bytes(max_m));
  npu::GemmBuffer c(rt, gemm.c_bytes(max_m));
  std::vector<float> operand(max_m * s.dim);
  auto* a_bits = a.as<std::uint16_t>();
  for (std::size_t i = 0; i < operand.size(); ++i) {
    a_bits[i] = quant::FloatToBf16(act(rng));
    operand[i] = quant::Bf16ToFloat(a_bits[i]);
  }
  a.ToDevice();
  const auto want_kernel = Reference(operand, held, max_m, s.dim, s.rows);
  const auto want_total = Reference(operand, q, max_m, s.dim, s.rows);
  int failures = 0;
  for (const std::size_t m : npu::kGemmChunks) {
    gemm.Run(a, c, m);
    c.FromDevice();
    std::vector<float> got(m * s.rows);
    const auto* c_bits = c.as<std::uint16_t>();
    for (std::size_t i = 0; i < got.size(); ++i) {
      got[i] = quant::Bf16ToFloat(c_bits[i]);
    }
    const std::vector<double> kernel_ref(want_kernel.begin(),
                                         want_kernel.begin() + got.size());
    const std::vector<double> total_ref(want_total.begin(),
                                        want_total.begin() + got.size());
    const double kernel_err = RelativeError(got, kernel_ref);
    const double total_err = RelativeError(got, total_ref);

    gemm.Time(10);  // warm up
    const auto t = gemm.Time(50);
    const double flop = 2.0 * m * s.dim * s.rows;
    const bool ok = total_err < kTolerance;
    failures += ok ? 0 : 1;
    std::printf(
        "%s %s: kernel error %.2e, total %.2e; mean %.0f us, best %.0f us "
        "(%.1f TFLOPS at best)\n",
        ok ? "ok  " : "FAIL",
        npu::Bfp16GemmSequence(m, s.dim, s.rows, model::kGemmName).c_str(),
        kernel_err, total_err, t.mean_us, t.best_us, flop / t.best_us / 1e6);
  }
  return failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen3vl_embed_2b_gemm_test <kernel-dir>\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("skip: no NPU at /dev/accel/accel0");
    return 77;
  }
  std::mt19937 rng(4321);
  npu::GemmRuntime runtime(argv[1], model::kGemmName);
  int failures = 0;
  for (const auto& s : kShapes) {
    failures += Check(runtime, s, rng);
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
