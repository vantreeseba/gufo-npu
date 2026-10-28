#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <random>
#include <vector>

#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/gemv.hpp"

// Determinism of chained q4nx GEMVs: one decoder layer's chain (in, out,
// gate + up with SwiGLU, down, the next layer's in), wired as NpuDecoder
// wires it, runs many times with the same inputs, alternating 1 and 2
// tokens. Every output must be bit-identical to the first run with the same
// token count. This caught a DMA wait-order bug that put the wrong round's
// values into the SwiGLU output about once in 30,000 calls, so the default
// run is long (about a minute); argv[2] overrides the iteration count.
// Exits 77 (skip) without an NPU.
namespace {

namespace npu = gufo_npu::npu;
namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
using npu::Buffer;
using npu::Epilogue;
using npu::Prologue;

constexpr long kDefaultIterations = 60000;

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: q4nx_gemv_chain_test <kernel-dir> [iterations]\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const long iterations = argc > 2 ? std::atol(argv[2]) : kDefaultIterations;
  npu::GemvRuntime rt(argv[1]);
  std::mt19937 rng(1);
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> act(0.0f, 1.0f);
  auto matrix = [&](std::size_t rows, std::size_t dim, bool swiglu) {
    std::vector<float> w(rows * dim);
    for (auto& v : w) {
      v = weight(rng);
    }
    if (!swiglu) {
      return q4nx::PackFloat(w.data(), rows, dim, q4nx::Fit::kMinMax);
    }
    std::vector<std::uint16_t> bits(w.size());
    for (std::size_t i = 0; i < w.size(); ++i) {
      bits[i] = quant::FloatToBf16(w[i]);
    }
    return q4nx::PackRows(
        {{gufo_npu::gguf::GgmlType::kBF16,
          reinterpret_cast<const std::uint8_t*>(bits.data()), rows}},
        dim, q4nx::Fit::kMinMax, q4nx::SwiGluOrder(rows / 2));
  };
  npu::Q4nxGemv in(rt, matrix(8192, 2048, false), Prologue::kResNorm,
                   Epilogue::kStore);
  npu::Q4nxGemv out(rt, matrix(2048, 2048, false), Prologue::kPlain,
                    Epilogue::kStore);
  npu::Q4nxGemv ffn(rt, matrix(12288, 2048, true), Prologue::kResNorm,
                    Epilogue::kSwiGlu);
  npu::Q4nxGemv down(rt, matrix(2048, 6144, false), Prologue::kAct,
                     Epilogue::kStore);
  npu::Q4nxGemv next(rt, matrix(8192, 2048, false), Prologue::kResNorm,
                     Epilogue::kStore);

  const std::size_t hidden = npu::kGemvHidden * sizeof(float);
  const std::size_t slotted = npu::kGemvSlot * sizeof(float) + hidden;
  Buffer in_x(rt, npu::kGemvInputBytes);
  Buffer in_y(rt, in.y_bytes());
  Buffer out_x(rt, npu::kGemvInputBytes);
  Buffer ffn_x(rt, npu::kGemvInputBytes);
  Buffer ffn_y(rt, ffn.y_bytes());
  Buffer next_x(rt, npu::kGemvInputBytes);
  Buffer next_y(rt, next.y_bytes());
  Buffer next_r(rt, npu::kGemvResidualBytes);
  Buffer scratch(rt, npu::kGemvResidualBytes);
  for (Buffer* b : {&in_x, &ffn_x, &next_x}) {
    auto* norm = reinterpret_cast<std::uint16_t*>(b->as<std::uint8_t>() +
                                                  npu::kGemvNormOffset);
    for (std::size_t i = 0; i < npu::kGemvHidden; ++i) {
      norm[i] = quant::FloatToBf16(1.0f + 0.2f * act(rng));
    }
  }
  for (std::size_t t = 0; t < npu::kGemvTokens; ++t) {
    float* h = in_x.as<float>() + t * npu::kGemvSlot;  // H and D
    for (std::size_t i = 0; i < 2 * npu::kGemvHidden; ++i) {
      h[i] = act(rng);
    }
    float* o = out_x.as<float>() + t * npu::kGemvSlot;
    for (std::size_t i = 0; i < npu::kGemvHidden; ++i) {
      o[i] = act(rng);
    }
  }
  for (Buffer* b : {&in_x, &out_x, &ffn_x, &next_x}) {
    b->ToDevice();
  }
  Buffer ffn_h = ffn_x.Sub(0, slotted);
  Buffer ffn_d = ffn_x.Sub(hidden, slotted);
  Buffer next_h = next_x.Sub(0, slotted);
  Buffer next_d = next_x.Sub(hidden, slotted);
  npu::Chain chains[npu::kGemvTokens] = {npu::Chain(rt), npu::Chain(rt)};
  for (std::size_t t = 1; t <= npu::kGemvTokens; ++t) {
    auto& c = chains[t - 1];
    c.Add(in, in_x, in_y, ffn_h, t);
    c.Add(out, out_x, ffn_d, scratch, t);
    c.Add(ffn, ffn_x, ffn_y, next_h, t);
    c.Add(down, ffn_y, next_d, scratch, t);
    c.Add(next, next_x, next_y, next_r, t);
  }

  struct Watched {
    const char* name;
    Buffer* buffer;
  };
  const Watched watched[] = {{"in_y", &in_y},     {"ffn_x", &ffn_x},
                             {"ffn_y", &ffn_y},   {"next_x", &next_x},
                             {"next_y", &next_y}, {"next_r", &next_r}};
  std::vector<std::vector<std::uint8_t>> want[npu::kGemvTokens];
  long bad_runs = 0;
  for (long it = 0; it < iterations; ++it) {
    const auto t = static_cast<std::size_t>(it) % npu::kGemvTokens;
    chains[t].Run();
    bool bad = false;
    for (std::size_t w = 0; w < std::size(watched); ++w) {
      Buffer& b = *watched[w].buffer;
      b.FromDevice();
      const auto* p = b.as<std::uint8_t>();
      if (want[t].size() <= w) {
        want[t].emplace_back(p, p + b.size());
        continue;
      }
      if (std::memcmp(p, want[t][w].data(), b.size()) != 0 && !bad) {
        std::size_t i = 0;
        while (p[i] == want[t][w][i]) {
          ++i;
        }
        std::fprintf(stderr,
                     "run %ld (%zu tokens): %s first differs at byte %zu\n", it,
                     t + 1, watched[w].name, i);
        bad = true;
      }
    }
    bad_runs += bad ? 1 : 0;
  }
  std::printf("%s chained GEMVs: %ld runs, %ld with any output differing\n",
              bad_runs == 0 ? "ok  " : "FAIL", iterations, bad_runs);
  return bad_runs == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
