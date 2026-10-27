#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/gemv.hpp"

// Every q4nx GEMV variant on the NPU against the CPU reference (double
// precision over the same packed weights and the same bf16 operand), each in
// the input/output stages it is built for, then the same calls chained in one
// runlist. Exits 77 without an NPU.
namespace {

namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
namespace npu = gufo_npu::npu;
using npu::Epilogue;
using npu::Prologue;

constexpr double kTolerance = 1e-2;

struct Variant {
  std::size_t dim;
  std::size_t rows;
  Prologue prologue;
  Epilogue epilogue;
};

// Must match VARIANTS in kernels/designs/q4nx_gemv.py (rows before padding).
constexpr Variant kVariants[] = {
    {2048, 2048, Prologue::kPlain, Epilogue::kStore},
    {2048, 5120, Prologue::kResNorm, Epilogue::kStore},
    {2048, 8192, Prologue::kResNorm, Epilogue::kStore},
    {2048, 12288, Prologue::kResNorm, Epilogue::kSwiGlu},
    {6144, 2048, Prologue::kAct, Epilogue::kStore},
    {2048, 248320, Prologue::kResNorm, Epilogue::kStore},
};

float Bf16(float v) {
  return quant::Bf16ToFloat(quant::FloatToBf16(v));
}

double Silu(double x) {
  return x / (1.0 + std::exp(-x));
}

struct Case {
  std::unique_ptr<npu::Q4nxGemv> gemv;
  std::unique_ptr<npu::Buffer> x;
  std::unique_ptr<npu::Buffer> y;
  std::unique_ptr<npu::Buffer> r;
  std::vector<double> want_y;  // padded_rows (swiglu: per its layout)
  std::vector<float> want_r;   // resnorm only
  std::string name;
};

Case Build(npu::GemvRuntime& rt, const Variant& v, std::mt19937& rng) {
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> act(0.0f, 1.0f);
  std::vector<float> w(v.rows * v.dim);
  for (auto& value : w) {
    value = weight(rng);
  }
  q4nx::Matrix packed;
  if (v.epilogue == Epilogue::kSwiGlu) {
    std::vector<std::uint16_t> bits(w.size());
    for (std::size_t i = 0; i < w.size(); ++i) {
      bits[i] = quant::FloatToBf16(w[i]);
    }
    packed = q4nx::PackRows(
        {{gufo_npu::gguf::GgmlType::kBF16,
          reinterpret_cast<const std::uint8_t*>(bits.data()), v.rows}},
        v.dim, q4nx::SwiGluOrder(v.rows / 2));
  } else {
    packed = q4nx::PackFloat(w.data(), v.rows, v.dim);
  }
  w.clear();
  w.shrink_to_fit();

  Case c;
  c.name =
      npu::Q4nxGemvSequence(v.dim, packed.padded_rows, v.prologue, v.epilogue);
  c.gemv = std::make_unique<npu::Q4nxGemv>(rt, packed, v.prologue, v.epilogue);
  const std::size_t x_bytes =
      v.prologue == Prologue::kAct ? 8 * v.dim : npu::kGemvInputBytes;
  c.x = std::make_unique<npu::Buffer>(rt, x_bytes);
  c.y = std::make_unique<npu::Buffer>(rt, packed.padded_rows * sizeof(float));
  c.r = std::make_unique<npu::Buffer>(rt, npu::kGemvHidden * sizeof(float));

  // The bf16 operand the kernel multiplies.
  std::vector<float> operand(v.dim);
  float* x = c.x->as<float>();
  if (v.prologue == Prologue::kResNorm) {
    float* h = x;
    float* d = x + v.dim;
    auto* nw = reinterpret_cast<std::uint16_t*>(x + 2 * v.dim);
    c.want_r.resize(v.dim);
    double sq = 0;
    for (std::size_t i = 0; i < v.dim; ++i) {
      h[i] = act(rng) * 3.0f;
      d[i] = act(rng);
      nw[i] = quant::FloatToBf16(1.0f + 0.2f * act(rng));
      c.want_r[i] = h[i] + d[i];
      sq += static_cast<double>(c.want_r[i]) * c.want_r[i];
    }
    const double inv = 1.0 / std::sqrt(sq / v.dim + 1e-6);
    for (std::size_t i = 0; i < v.dim; ++i) {
      operand[i] = Bf16(
          static_cast<float>(c.want_r[i] * inv * quant::Bf16ToFloat(nw[i])));
    }
  } else if (v.prologue == Prologue::kAct) {
    // The first 16 of every 32 values are the input; the rest must be
    // skipped.
    for (std::size_t i = 0; i < 2 * v.dim; ++i) {
      x[i] = (i % 32 < 16) ? act(rng) : 1e6f;
    }
    for (std::size_t i = 0; i < v.dim; ++i) {
      operand[i] = Bf16(x[(i / 16) * 32 + i % 16]);
    }
  } else {
    for (std::size_t i = 0; i < v.dim; ++i) {
      x[i] = act(rng) * 4.0f;
      operand[i] = Bf16(x[i]);
    }
  }
  c.x->ToDevice();

  const auto ref =
      q4nx::Reference(q4nx::Unpack(packed), packed, operand.data());
  c.want_y = ref;
  if (v.epilogue == Epilogue::kSwiGlu) {
    for (std::size_t t = 0; t < packed.padded_rows / 32; ++t) {
      for (std::size_t m = 0; m < 16; ++m) {
        c.want_y[t * 32 + m] = Silu(ref[t * 32 + m]) * ref[t * 32 + 16 + m];
        c.want_y[t * 32 + 16 + m] = 0;
      }
    }
  }
  return c;
}

int Check(Case& c, const char* how) {
  c.y->FromDevice();
  c.r->FromDevice();
  const float* y = c.y->as<float>();
  double err2 = 0;
  double ref2 = 0;
  for (std::size_t r = 0; r < c.want_y.size(); ++r) {
    const double e = y[r] - c.want_y[r];
    err2 += e * e;
    ref2 += c.want_y[r] * c.want_y[r];
  }
  const double rel = std::sqrt(err2 / ref2);
  // The residual is one fp32 add per value: allow its rounding, against the
  // residual's scale (H and D can nearly cancel).
  double r_err = 0;
  double r_scale = 0;
  for (std::size_t i = 0; i < c.want_r.size(); ++i) {
    r_err = std::fmax(r_err, std::fabs(c.r->as<float>()[i] - c.want_r[i]));
    r_scale = std::fmax(r_scale, std::fabs(c.want_r[i]));
  }
  r_err = r_scale > 0 ? r_err / r_scale : 0;
  const bool ok = rel < kTolerance && r_err < 1e-6;
  std::printf("%s %s (%s): relative error %.3e", ok ? "ok  " : "FAIL",
              c.name.c_str(), how, rel);
  if (!c.want_r.empty()) {
    std::printf(", residual max error %.3g of its scale", r_err);
  }
  std::printf("\n");
  // Clear the outputs for the next run.
  std::memset(c.y->data(), 0, c.y->size());
  std::memset(c.r->data(), 0, c.r->size());
  c.y->ToDevice();
  c.r->ToDevice();
  return ok ? 0 : 1;
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
  // Every variant runs in the one hardware context.
  npu::GemvRuntime runtime(argv[1]);
  std::vector<Case> cases;
  for (const auto& v : kVariants) {
    cases.push_back(Build(runtime, v, rng));
  }
  int failures = 0;
  for (auto& c : cases) {
    c.gemv->Run(*c.x, *c.y, *c.r);
    failures += Check(c, "alone");
  }
  // The same calls, chained in one runlist (the kernel edits only its
  // on-chip copy of an input, so the inputs are still in place).
  npu::Chain chain(runtime);
  for (auto& c : cases) {
    chain.Add(*c.gemv, *c.x, *c.y, *c.r);
  }
  chain.Run();
  for (auto& c : cases) {
    failures += Check(c, "chained");
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
