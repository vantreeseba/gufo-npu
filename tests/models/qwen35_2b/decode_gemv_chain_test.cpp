#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <vector>

#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "models/qwen35_2b/npu/gemv.hpp"
#include "npu/gemv.hpp"

// Determinism of chained q4nx GEMVs on Qwen3.5-2B's decode program
// (qwen35_2b_decode): one decoder layer (in, out, gate + up with SwiGLU,
// down, the next layer's in), wired as NpuDecoder wires it, two ways on two
// copies of the same activation buffer (npu/gemv.hpp's block layout):
//   per call  every GEMV its own run in one runlist;
//   chain     the input projection, then out .. next in as the chain
//             sequence (one run), as the decoder submits them.
// Runs alternate 1 and 2 tokens. After every run, both copies must be bit
// for bit equal to each other and to the first run with the same token
// count. The per-call check caught a DMA wait-order bug that put the wrong
// round's values into the SwiGLU output about once in 30,000 calls, so the
// default run is long (about two minutes); argv[2] overrides the iteration
// count. Exits 77 (skip) without an NPU.
namespace {

namespace npu = gufo_npu::npu;
namespace geo = gufo_npu::qwen35_2b::npu;
namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
using npu::Buffer;
using npu::Epilogue;
using npu::Prologue;

constexpr long kDefaultIterations = 60000;
// Two blocks: the layer's and the next layer's.
constexpr std::size_t kActsBytes = 2 * geo::kBlockBytes;

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(
        stderr,
        "usage: qwen35_decode_gemv_chain_test <kernel-dir> [iterations]\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const long iterations = argc > 2 ? std::atol(argv[2]) : kDefaultIterations;
  npu::GemvRuntime rt(argv[1], npu::Flow::kElf, geo::DecodeDesign());
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
  // out, gate + up and down back to back, as the chain reads them.
  const auto out_w = matrix(2048, 2048, false);
  const auto ffn_w = matrix(12288, 2048, true);
  const auto down_w = matrix(2048, 6144, false);
  const std::size_t out_bytes = out_w.bytes.size();
  const std::size_t ffn_bytes = ffn_w.bytes.size();
  Buffer layer_w(rt, out_bytes + ffn_bytes + down_w.bytes.size());
  npu::Q4nxGemv out(rt, out_w, Prologue::kPlain, Epilogue::kStore,
                    layer_w.Sub(0, out_bytes));
  npu::Q4nxGemv ffn(rt, ffn_w, Prologue::kResNorm, Epilogue::kSwiGlu,
                    layer_w.Sub(out_bytes, ffn_bytes));
  npu::Q4nxGemv down(rt, down_w, Prologue::kAct, Epilogue::kStore,
                     layer_w.Sub(out_bytes + ffn_bytes, down_w.bytes.size()));
  npu::Q4nxGemv next(rt, matrix(8192, 2048, false), Prologue::kResNorm,
                     Epilogue::kStore);
  Buffer next_w = next.weights();
  Buffer scratch(rt, geo::kResidualBytes);

  // The two copies' inputs: the layer's input region (H and D per token,
  // the norm weight), its mixer output and both blocks' norm weights.
  Buffer per_call(rt, kActsBytes);
  Buffer chained(rt, kActsBytes);
  {
    std::vector<std::uint8_t> init(kActsBytes, 0);
    auto norm = [&](std::size_t region) {
      auto* w =
          reinterpret_cast<std::uint16_t*>(&init[region + geo::kNormOffset]);
      for (std::size_t i = 0; i < geo::kHidden; ++i) {
        w[i] = quant::FloatToBf16(1.0f + 0.2f * act(rng));
      }
    };
    for (std::size_t block : {std::size_t{0}, geo::kBlockBytes}) {
      norm(block + geo::kBlockInX);
      norm(block + geo::kBlockFfnX);
    }
    for (std::size_t t = 0; t < geo::kTokens; ++t) {
      auto* h =
          reinterpret_cast<float*>(&init[geo::kBlockInX]) + t * geo::kSlot;
      for (std::size_t i = 0; i < 2 * geo::kHidden; ++i) {
        h[i] = act(rng);
      }
      auto* o =
          reinterpret_cast<float*>(&init[geo::kBlockOutX]) + t * geo::kSlot;
      for (std::size_t i = 0; i < geo::kHidden; ++i) {
        o[i] = act(rng);
      }
    }
    for (Buffer* b : {&per_call, &chained}) {
      std::memcpy(b->data(), init.data(), kActsBytes);
      b->ToDevice();
    }
  }

  const std::size_t hidden = geo::kHidden * sizeof(float);
  const std::size_t slotted = geo::kSlot * sizeof(float) + hidden;
  // Region `offset` of `acts`, `bytes` long, kept alive for the chains.
  std::vector<std::unique_ptr<Buffer>> regions;
  auto region = [&](Buffer& acts, std::size_t offset,
                    std::size_t bytes) -> Buffer& {
    regions.push_back(std::make_unique<Buffer>(acts.Sub(offset, bytes)));
    return *regions.back();
  };
  const std::size_t b1 = geo::kBlockBytes;
  npu::Chain calls[geo::kTokens] = {npu::Chain(rt), npu::Chain(rt)};
  npu::Chain chains[geo::kTokens] = {npu::Chain(rt), npu::Chain(rt)};
  for (std::size_t t = 1; t <= geo::kTokens; ++t) {
    for (Buffer* acts : {&per_call, &chained}) {
      auto& c = acts == &per_call ? calls[t - 1] : chains[t - 1];
      c.Add(in, region(*acts, geo::kBlockInX, geo::kInputBytes),
            region(*acts, geo::kBlockInY, in.y_bytes()),
            region(*acts, geo::kBlockFfnX, slotted), t);
      if (acts == &chained) {
        c.Add(geo::ChainSequence(geo::ChainNext::kDeltaNet, t),
              region(*acts, geo::kBlockFfnX,
                     geo::ChainWindowBytes(geo::ChainNext::kDeltaNet)),
              layer_w, next_w, scratch);
        continue;
      }
      c.Add(out, region(*acts, geo::kBlockOutX, geo::kInputBytes),
            region(*acts, geo::kBlockFfnX + hidden, slotted), scratch, t);
      c.Add(ffn, region(*acts, geo::kBlockFfnX, geo::kInputBytes),
            region(*acts, geo::kBlockFfnY, ffn.y_bytes()),
            region(*acts, b1 + geo::kBlockInX, slotted), t);
      c.Add(down, region(*acts, geo::kBlockFfnY, ffn.y_bytes()),
            region(*acts, b1 + geo::kBlockInX + hidden, slotted), scratch, t);
      c.Add(next, region(*acts, b1 + geo::kBlockInX, geo::kInputBytes),
            region(*acts, b1 + geo::kBlockInY, next.y_bytes()),
            region(*acts, b1 + geo::kBlockFfnX, slotted), t);
    }
  }

  std::vector<std::uint8_t> want[geo::kTokens];
  long bad_runs = 0;
  long mismatched_runs = 0;
  auto first_difference = [](const std::uint8_t* a, const std::uint8_t* b) {
    std::size_t i = 0;
    while (a[i] == b[i]) {
      ++i;
    }
    return i;
  };
  for (long it = 0; it < iterations; ++it) {
    const auto t = static_cast<std::size_t>(it) % geo::kTokens;
    calls[t].Run();
    chains[t].Run();
    per_call.FromDevice();
    chained.FromDevice();
    const auto* p = per_call.as<std::uint8_t>();
    const auto* c = chained.as<std::uint8_t>();
    if (want[t].empty()) {
      want[t].assign(p, p + kActsBytes);
    }
    if (std::memcmp(p, want[t].data(), kActsBytes) != 0) {
      ++bad_runs;
      std::fprintf(stderr,
                   "run %ld (%zu tokens): per-call output first differs from "
                   "run %zu at byte %zu\n",
                   it, t + 1, t, first_difference(p, want[t].data()));
    }
    if (std::memcmp(c, p, kActsBytes) != 0) {
      ++mismatched_runs;
      std::fprintf(stderr,
                   "run %ld (%zu tokens): chain sequence differs from the "
                   "per-call chain at byte %zu\n",
                   it, t + 1, first_difference(c, p));
    }
  }
  const bool ok = bad_runs == 0 && mismatched_runs == 0;
  std::printf(
      "%s chained GEMVs: %ld runs, %ld per-call runs differing from the "
      "first, %ld chain-sequence runs differing from the per-call chain\n",
      ok ? "ok  " : "FAIL", iterations, bad_runs, mismatched_runs);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
