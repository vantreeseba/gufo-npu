#include <algorithm>
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
#include "models/qwen35_4b/npu/gemv.hpp"
#include "npu/gemv.hpp"

// Every variant of Qwen3.5-4B's decode GEMV (its own NPU program,
// src/models/qwen35_4b/kernels) on the NPU against the CPU reference (double
// precision over the same packed weights and the same bf16 operand), each in
// the input/output stages it is built for, for one and two tokens, then the
// same calls chained in one runlist. Hidden-sized outputs (2560 rows) run a
// partial last round: only their rows are stored, and the rest of the output
// buffer must stay untouched. Adapted from tests/npu/q4nx_gemv_test.cpp.
// Exits 77 without an NPU.
namespace {

namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
namespace npu = gufo_npu::npu;
namespace geo = gufo_npu::qwen35_4b::npu;
using npu::Epilogue;
using npu::Prologue;

constexpr double kTolerance = 1e-2;

struct Variant {
  std::size_t dim;
  std::size_t rows;
  Prologue prologue;
  Epilogue epilogue;
};

// Must match VARIANTS in src/models/qwen35_4b/kernels/decode_gemv_variants.py
// (rows before padding).
constexpr Variant kVariants[] = {
    {4096, 2560, Prologue::kPlain, Epilogue::kStore},
    {5120, 2560, Prologue::kPlain, Epilogue::kStore},
    {2560, 10240, Prologue::kResNorm, Epilogue::kStore},
    {2560, 12288, Prologue::kResNorm, Epilogue::kStore},
    {2560, 18432, Prologue::kResNorm, Epilogue::kSwiGlu},
    {9216, 2560, Prologue::kAct, Epilogue::kStore},
    {2560, 248320, Prologue::kResNorm, Epilogue::kStore},
    {2560, 32768, Prologue::kResNorm, Epilogue::kStore},
};
// Written into every output buffer before a call; stored rows overwrite it.
constexpr float kSentinel = 12345.0f;

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
  Variant v;
  std::size_t tokens = 1;
  std::vector<std::vector<double>> want_y;  // per token, padded_rows (swiglu:
                                            // padded_rows / 2 act values)
  std::vector<std::vector<float>> want_r;   // per token, resnorm only
  std::string name;
};

// One matrix, then its 1- and 2-token calls (both share the weights).
std::vector<Case> Build(npu::GemvRuntime& rt, const Variant& v,
                        std::mt19937& rng) {
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
  const auto unpacked = q4nx::Unpack(packed);

  std::vector<Case> cases;
  for (std::size_t tokens = 1; tokens <= geo::kTokens; ++tokens) {
    Case c;
    c.v = v;
    c.tokens = tokens;
    c.name = npu::Q4nxGemvSequence(v.dim, packed.padded_rows, v.prologue,
                                   v.epilogue, tokens, geo::kDecodeName);
    if (tokens == 1) {
      c.gemv =
          std::make_unique<npu::Q4nxGemv>(rt, packed, v.prologue, v.epilogue);
    }
    // act reads a SwiGLU output buffer: make it as large as one.
    const std::size_t x_bytes =
        std::max<std::size_t>(geo::kInputBytes, 2 * v.dim * 2 * sizeof(float));
    c.x = std::make_unique<npu::Buffer>(rt, x_bytes);
    c.r = std::make_unique<npu::Buffer>(rt, geo::kResidualBytes);
    float* x = c.x->as<float>();
    for (std::size_t t = 0; t < tokens; ++t) {
      // The bf16 operand the kernel multiplies for token t.
      std::vector<float> operand(v.dim);
      if (v.prologue == Prologue::kResNorm) {
        float* h = x + t * geo::kSlot;
        float* d = h + v.dim;
        auto* nw = reinterpret_cast<std::uint16_t*>(c.x->as<std::uint8_t>() +
                                                    geo::kNormOffset);
        std::vector<float> res(v.dim);
        double sq = 0;
        for (std::size_t i = 0; i < v.dim; ++i) {
          h[i] = act(rng) * 3.0f;
          d[i] = act(rng);
          if (t == 0) {
            nw[i] = quant::FloatToBf16(1.0f + 0.2f * act(rng));
          }
          res[i] = h[i] + d[i];
          sq += static_cast<double>(res[i]) * res[i];
        }
        const double inv = 1.0 / std::sqrt(sq / v.dim + 1e-6);
        for (std::size_t i = 0; i < v.dim; ++i) {
          operand[i] = Bf16(
              static_cast<float>(res[i] * inv * quant::Bf16ToFloat(nw[i])));
        }
        c.want_r.push_back(res);
      } else if (v.prologue == Prologue::kAct) {
        auto* a = reinterpret_cast<std::uint16_t*>(x) + t * geo::kMaxDim;
        for (std::size_t i = 0; i < v.dim; ++i) {
          a[i] = quant::FloatToBf16(act(rng));
          operand[i] = quant::Bf16ToFloat(a[i]);
        }
      } else {
        float* slot = x + t * geo::kSlot;
        for (std::size_t i = 0; i < v.dim; ++i) {
          slot[i] = act(rng) * 4.0f;
          operand[i] = Bf16(slot[i]);
        }
      }
      const auto ref = q4nx::Reference(unpacked, packed, operand.data());
      if (v.epilogue == Epilogue::kSwiGlu) {
        std::vector<double> a(packed.padded_rows / 2);
        for (std::size_t tile = 0; tile < packed.padded_rows / 32; ++tile) {
          for (std::size_t m = 0; m < 16; ++m) {
            a[tile * 16 + m] =
                Silu(ref[tile * 32 + m]) * ref[tile * 32 + 16 + m];
          }
        }
        c.want_y.push_back(a);
      } else {
        c.want_y.emplace_back(ref.begin(), ref.begin() + v.rows);
      }
    }
    c.x->ToDevice();
    cases.push_back(std::move(c));
  }
  // The 2-token case shares the 1-token case's GEMV.
  auto* shared = cases[0].gemv.get();
  for (auto& c : cases) {
    c.y = std::make_unique<npu::Buffer>(rt, shared->y_bytes());
  }
  return cases;
}

void FillSentinel(npu::Buffer& y) {
  std::fill_n(y.as<float>(), y.size() / sizeof(float), kSentinel);
  y.ToDevice();
}

// Values in token t's hidden-sized output past `rows` (up to the next token's
// slot, or the region's end) must keep the sentinel.
std::size_t Overwritten(Case& c, npu::Q4nxGemv& gemv) {
  if (c.v.epilogue != Epilogue::kStore || c.v.rows != geo::kHidden) {
    return 0;
  }
  std::size_t bad = 0;
  const float* y = c.y->as<float>();
  const std::size_t n = c.y->size() / sizeof(float);
  for (std::size_t t = 0; t < geo::kTokens; ++t) {
    const std::size_t begin = t * gemv.y_token_stride() + c.v.rows;
    const std::size_t end = std::min(n, (t + 1) * gemv.y_token_stride());
    for (std::size_t i = begin; i < end; ++i) {
      bad += y[i] != kSentinel;
    }
  }
  return bad;
}

int Check(Case& c, npu::Q4nxGemv& gemv, const char* how) {
  c.y->FromDevice();
  c.r->FromDevice();
  double err2 = 0;
  double ref2 = 0;
  for (std::size_t t = 0; t < c.tokens; ++t) {
    const auto& want = c.want_y[t];
    for (std::size_t i = 0; i < want.size(); ++i) {
      double got;
      if (c.v.epilogue == Epilogue::kSwiGlu) {
        const auto* bits = reinterpret_cast<const std::uint16_t*>(
            c.y->as<float>() + t * gemv.y_token_stride());
        got = quant::Bf16ToFloat(bits[i]);
      } else {
        got = c.y->as<float>()[t * gemv.y_token_stride() + i];
      }
      const double e = got - want[i];
      err2 += e * e;
      ref2 += want[i] * want[i];
    }
  }
  const double rel = std::sqrt(err2 / ref2);
  // SwiGLU results are bf16 (the next GEMV's input precision anyway).
  const double tolerance = kTolerance;
  // The residual is one fp32 add per value: allow its rounding, against the
  // residual's scale (H and D can nearly cancel).
  double r_err = 0;
  for (std::size_t t = 0; t < c.want_r.size(); ++t) {
    double worst = 0;
    double scale = 0;
    for (std::size_t i = 0; i < c.want_r[t].size(); ++i) {
      worst = std::fmax(worst, std::fabs(c.r->as<float>()[t * geo::kSlot + i] -
                                         c.want_r[t][i]));
      scale = std::fmax(scale, std::fabs(c.want_r[t][i]));
    }
    r_err = std::fmax(r_err, scale > 0 ? worst / scale : 0);
  }
  const std::size_t overwritten = Overwritten(c, gemv);
  const bool ok = rel < tolerance && r_err < 1e-6 && overwritten == 0;
  std::printf("%s %s (%s): relative error %.3e", ok ? "ok  " : "FAIL",
              c.name.c_str(), how, rel);
  if (overwritten != 0) {
    std::printf(", %zu values written past the rows", overwritten);
  }
  if (!c.want_r.empty()) {
    std::printf(", residual max error %.3g of its scale", r_err);
  }
  std::printf("\n");
  // Reset the outputs for the next run.
  FillSentinel(*c.y);
  std::memset(c.r->data(), 0, c.r->size());
  c.r->ToDevice();
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen35_4b_decode_gemv_test <kernel-dir>\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("skip: no NPU at /dev/accel/accel0");
    return 77;
  }
  std::mt19937 rng(1234);
  // Every variant runs in the one hardware context.
  npu::GemvRuntime runtime(argv[1], npu::Flow::kElf, geo::DecodeDesign());
  std::vector<std::vector<Case>> groups;
  for (const auto& v : kVariants) {
    groups.push_back(Build(runtime, v, rng));
    for (auto& c : groups.back()) {
      FillSentinel(*c.y);
    }
  }
  int failures = 0;
  for (auto& group : groups) {
    for (auto& c : group) {
      group[0].gemv->Run(*c.x, *c.y, *c.r, c.tokens);
      failures += Check(c, *group[0].gemv, "alone");
    }
  }
  // The same calls, chained in one runlist (the kernel edits only its
  // on-chip copy of an input, so the inputs are still in place).
  npu::Chain chain(runtime);
  for (auto& group : groups) {
    for (auto& c : group) {
      chain.Add(*group[0].gemv, *c.x, *c.y, *c.r, c.tokens);
    }
  }
  chain.Run();
  for (auto& group : groups) {
    for (auto& c : group) {
      failures += Check(c, *group[0].gemv, "chained");
    }
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
