#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <vector>

#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/gemv.hpp"

// The q4nx GEMV on the NPU against the CPU reference (double precision over
// the same packed weights), for every compiled decode shape. Exits 77 without
// an NPU.
namespace {

namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;

struct Shape {
  std::size_t dim;
  std::size_t rows;
};

// Must match SHAPES in kernels/designs/q4nx_gemv.py (rows before padding).
constexpr Shape kShapes[] = {
    {2048, 2048},  {2048, 5120}, {2048, 8192},
    {2048, 12288}, {6144, 2048}, {2048, 248320},
};

int CheckShape(gufo_npu::npu::GemvRuntime& runtime, const Shape& s,
               std::mt19937& rng) {
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> activation(0.0f, 1.0f);
  std::vector<float> w(s.rows * s.dim);
  for (auto& v : w) {
    v = weight(rng);
  }
  const auto packed = q4nx::PackFloat(w.data(), s.rows, s.dim);
  w.clear();
  w.shrink_to_fit();

  // The token in bf16.
  std::vector<std::uint16_t> x_bits(s.dim);
  std::vector<float> x(s.dim);
  for (std::size_t i = 0; i < s.dim; ++i) {
    x_bits[i] = quant::FloatToBf16(activation(rng) * 4.0f);
    x[i] = quant::Bf16ToFloat(x_bits[i]);
  }
  const auto ref = q4nx::Reference(q4nx::Unpack(packed), packed, x.data());

  gufo_npu::npu::Q4nxGemv gemv(runtime, packed);
  std::vector<float> y(packed.padded_rows);
  gemv.Run(x_bits.data(), y.data());

  double err2 = 0;
  double ref2 = 0;
  double max_err = 0;
  for (std::size_t r = 0; r < packed.rows; ++r) {
    const double e = y[r] - ref[r];
    err2 += e * e;
    ref2 += ref[r] * ref[r];
    max_err = std::fmax(max_err, std::fabs(e));
  }
  const auto name = gufo_npu::npu::Q4nxGemvSequence(s.dim, packed.padded_rows);
  for (std::size_t r = packed.rows; r < packed.padded_rows; ++r) {
    if (y[r] != 0.0f) {
      std::fprintf(stderr, "FAIL: %s: padding row %zu is not zero\n",
                   name.c_str(), r);
      return 1;
    }
  }
  const double rel = std::sqrt(err2 / ref2);
  const double rms = std::sqrt(ref2 / packed.rows);
  if (!(rel < 1e-2)) {
    std::fprintf(stderr, "FAIL: %s: relative error %.3e (max %.3e, rms %.3e)\n",
                 name.c_str(), rel, max_err, rms);
    return 1;
  }
  std::printf("ok: %s relative error %.3e (max %.3e of rms %.3e)\n",
              name.c_str(), rel, max_err, rms);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: q4nx_gemv_test <kernel-dir>\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("skip: no NPU at /dev/accel/accel0");
    return 77;
  }
  std::mt19937 rng(1234);
  // Every shape runs in the one hardware context.
  gufo_npu::npu::GemvRuntime runtime(argv[1]);
  int failures = 0;
  for (const auto& shape : kShapes) {
    failures += CheckShape(runtime, shape, rng);
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
