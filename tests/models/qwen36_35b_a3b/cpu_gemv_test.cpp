#include "models/qwen36_35b_a3b/cpu_gemv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "core/cpus.hpp"
#include "core/q4nx.hpp"
#include "core/simd.hpp"
#include "core/thread_pool.hpp"

// The CPU q4nx GEMV the 35B decoder runs its mixer output projection with
// (src/models/qwen36_35b_a3b/cpu_gemv.hpp), on random q4nx matrices for 1
// and 2 tokens, with strided x and y:
//   - against a double-precision product over the same q4nx weights and the
//     same bf16-rounded x: relative L2 error < 1e-4 per token;
//   - against the unrounded fp32 x: < 1e-2;
//   - the AVX-512 BF16 path (when the CPU has it) equal to the portable one
//     bit for bit;
//   - repeated runs bit-identical.
// Needs no NPU.
namespace {

namespace q4nx = gufo_npu::q4nx;
namespace simd = gufo_npu::simd;
using gufo_npu::qwen36_35b_a3b::CpuGemv;

double RelativeError(const float* got, const std::vector<double>& want,
                     std::size_t n) {
  double num = 0;
  double den = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double e = got[i] - want[i];
    num += e * e;
    den += want[i] * want[i];
  }
  return std::sqrt(num / den);
}

struct Shape {
  std::size_t rows;
  std::size_t dim;
};

}  // namespace

int main() {
  gufo_npu::SpinPool pool(std::min<std::size_t>(7, gufo_npu::HalfCpuWorkers()));
  std::mt19937 rng(7);
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> act(0.0f, 1.0f);
  bool ok = true;
  const bool has_fast = CpuGemv(pool, nullptr, 32, 256).fast();
  std::printf("AVX-512 BF16 path: %s\n", has_fast ? "yes" : "no");
  // The mixer output projection's shape, and a short one with rows that
  // are not a multiple of the tile.
  for (const Shape shape : {Shape{2048, 4096}, Shape{1000, 512}}) {
    std::vector<float> w(shape.rows * shape.dim);
    for (auto& v : w) {
      v = weight(rng);
    }
    const q4nx::Matrix m =
        q4nx::PackFloat(w.data(), shape.rows, shape.dim, q4nx::Fit::kMinMax);
    const std::vector<float> unpacked = q4nx::Unpack(m);
    // x: 2 tokens at a stride past dim; y: 2 tokens at a stride past rows.
    const std::size_t x_stride = shape.dim + 64;
    const std::size_t y_stride = shape.rows + 32;
    std::vector<float> x(2 * x_stride);
    for (auto& v : x) {
      v = act(rng);
    }
    for (int tokens = 1; tokens <= 2; ++tokens) {
      std::vector<float> portable(2 * y_stride, 0.0f);
      CpuGemv(pool, m.bytes.data(), shape.rows, shape.dim, true)
          .Run(x.data(), x_stride, portable.data(), y_stride, tokens);
      for (int t = 0; t < tokens; ++t) {
        const float* xt = x.data() + t * x_stride;
        std::vector<float> rounded(xt, xt + shape.dim);
        for (auto& v : rounded) {
          v = simd::FromBf16(simd::ToBf16(v));
        }
        const auto want_bf16 = q4nx::Reference(unpacked, m, rounded.data());
        const auto want_fp32 = q4nx::Reference(unpacked, m, xt);
        const float* got = portable.data() + t * y_stride;
        const double e_bf16 = RelativeError(got, want_bf16, shape.rows);
        const double e_fp32 = RelativeError(got, want_fp32, shape.rows);
        const bool pass = e_bf16 < 1e-4 && e_fp32 < 1e-2;
        ok = ok && pass;
        std::printf(
            "%s %zux%zu, %d token(s), token %d: error %.2e against bf16 x, "
            "%.2e against fp32 x\n",
            pass ? "ok  " : "FAIL", shape.rows, shape.dim, tokens, t, e_bf16,
            e_fp32);
      }
      if (has_fast) {
        CpuGemv fast(pool, m.bytes.data(), shape.rows, shape.dim);
        std::vector<float> got(2 * y_stride, 0.0f);
        bool same = true;
        for (int run = 0; run < 3; ++run) {
          fast.Run(x.data(), x_stride, got.data(), y_stride, tokens);
          for (int t = 0; t < tokens; ++t) {
            same = same && std::memcmp(got.data() + t * y_stride,
                                       portable.data() + t * y_stride,
                                       shape.rows * sizeof(float)) == 0;
          }
        }
        std::size_t differing = 0;
        double worst = 0;
        for (int t = 0; t < tokens; ++t) {
          for (std::size_t r = 0; r < shape.rows; ++r) {
            const float a = got[t * y_stride + r];
            const float b = portable[t * y_stride + r];
            if (a != b) {
              ++differing;
              worst = std::max(
                  worst, std::abs(static_cast<double>(a) - b) /
                             std::max(1e-30, std::abs(static_cast<double>(b))));
            }
          }
        }
        ok = ok && same;
        std::printf(
            "%s %zux%zu, %d token(s): AVX-512 BF16 path %s the portable one "
            "(%zu outputs differ, worst %.2e relative)\n",
            same ? "ok  " : "FAIL", shape.rows, shape.dim, tokens,
            same ? "equals" : "differs from", differing, worst);
      }
    }
  }
  std::puts(ok ? "PASS" : "FAIL");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
