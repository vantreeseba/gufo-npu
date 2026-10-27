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
  bool norm;
};

// Must match SHAPES in kernels/designs/q4nx_gemv.py (rows before padding).
constexpr Shape kShapes[] = {
    {2048, 2048, false},
    {2048, 6144, false},
    {6144, 2048, false},
    {2048, 248320, true},
};

int CheckShape(const std::filesystem::path& kernels, const Shape& s,
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

  // The token (plus its norm weight) in bf16, and the vector the kernel
  // multiplies: the token itself, or its RMS norm rounded to bf16.
  std::vector<std::uint16_t> x_bits((s.norm ? 2 : 1) * s.dim);
  std::vector<float> x(x_bits.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float v =
        i < s.dim ? activation(rng) * 4.0f : 1.0f + 0.1f * activation(rng);
    x_bits[i] = quant::FloatToBf16(v);
    x[i] = quant::Bf16ToFloat(x_bits[i]);
  }
  std::vector<float> operand(x.begin(), x.begin() + s.dim);
  if (s.norm) {
    double sq = 0;
    for (std::size_t i = 0; i < s.dim; ++i) {
      sq += static_cast<double>(x[i]) * x[i];
    }
    const double inv = 1.0 / std::sqrt(sq / s.dim + 1e-6);
    for (std::size_t i = 0; i < s.dim; ++i) {
      operand[i] = quant::Bf16ToFloat(
          quant::FloatToBf16(static_cast<float>(x[i] * inv * x[s.dim + i])));
    }
  }
  const auto ref =
      q4nx::Reference(q4nx::Unpack(packed), packed, operand.data());

  gufo_npu::npu::Q4nxGemv gemv(kernels, packed, s.norm);
  std::vector<std::uint16_t> y(packed.padded_rows);
  gemv.Run(x_bits.data(), y.data());

  double err2 = 0;
  double ref2 = 0;
  double max_err = 0;
  for (std::size_t r = 0; r < packed.rows; ++r) {
    const double e = quant::Bf16ToFloat(y[r]) - ref[r];
    err2 += e * e;
    ref2 += ref[r] * ref[r];
    max_err = std::fmax(max_err, std::fabs(e));
  }
  const auto name =
      gufo_npu::npu::Q4nxGemvDesign(s.dim, packed.padded_rows, s.norm);
  for (std::size_t r = packed.rows; r < packed.padded_rows; ++r) {
    if (y[r] != 0 && y[r] != 0x8000) {
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
  int failures = 0;
  for (const auto& shape : kShapes) {
    failures += CheckShape(argv[1], shape, rng);
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
