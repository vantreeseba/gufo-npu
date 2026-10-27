#include <algorithm>
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

// Determinism of an MoE layer's calls, wired as NpuDecoder wires a
// Qwen3.6-35B-A3B layer (instruction-buffer flow): the input projection and
// the mixer's output projection in one submission, then, with the gathers
// pointed at a new selection, gate + up and down gathers and the next
// layer's input projection in another. The selections cycle through a few
// fixed ones; every output must be bit-identical to the first run with the
// same selection. Guards the DMA wait-order rule (AGENTS.md) for the gather
// sequences, whose many segment fills reuse BDs. argv[2] overrides the
// iteration count. Exits 77 (skip) without an NPU.
namespace {

namespace npu = gufo_npu::npu;
namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
using npu::Buffer;
using npu::Epilogue;
using npu::Prologue;

constexpr long kDefaultIterations = 20000;
constexpr int kArena = 24;
constexpr int kSelections = 4;

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: q4nx_moe_chain_test <kernel-dir> [iterations]\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const long iterations = argc > 2 ? std::atol(argv[2]) : kDefaultIterations;
  npu::GemvRuntime rt(argv[1], npu::Flow::kInstructionBuffer);
  std::mt19937 rng(3);
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> act(0.0f, 1.0f);
  auto matrix = [&](std::size_t rows, std::size_t dim) {
    std::vector<float> w(rows * dim);
    for (auto& v : w) {
      v = weight(rng);
    }
    return q4nx::PackFloat(w.data(), rows, dim);
  };
  const std::size_t hidden = npu::kGemvHidden;
  // DeltaNet input (qkv + z) and output projections of Qwen3.6-35B-A3B.
  npu::Q4nxGemv in(rt, matrix(12288, hidden), Prologue::kResNorm,
                   Epilogue::kStore);
  npu::Q4nxGemv out(rt, matrix(hidden, 4096), Prologue::kPlain,
                    Epilogue::kStore);
  npu::Q4nxGemv next(rt, matrix(12288, hidden), Prologue::kResNorm,
                     Epilogue::kStore);
  const npu::ExpertArena arena{kArena};
  Buffer arena_buf(rt, arena.bytes());
  for (int e = 0; e < kArena; ++e) {
    std::vector<float> w(2 * npu::kExpertFfn * hidden);
    for (auto& v : w) {
      v = weight(rng);
    }
    std::vector<std::uint16_t> bits(w.size());
    for (std::size_t i = 0; i < w.size(); ++i) {
      bits[i] = quant::FloatToBf16(w[i]);
    }
    const auto up =
        q4nx::PackRows({{gufo_npu::gguf::GgmlType::kBF16,
                         reinterpret_cast<const std::uint8_t*>(bits.data()),
                         2 * npu::kExpertFfn}},
                       hidden, q4nx::SwiGluOrder(npu::kExpertFfn));
    const auto down = matrix(hidden, npu::kExpertFfn);
    std::memcpy(arena_buf.as<std::uint8_t>() + arena.UpOffset(e),
                up.bytes.data(), up.bytes.size());
    std::memcpy(arena_buf.as<std::uint8_t>() + arena.DownOffset(e),
                down.bytes.data(), down.bytes.size());
  }
  arena_buf.ToDevice();
  npu::Q4nxGather gate_up(rt, npu::Q4nxGather::Kind::kGateUp, arena);
  npu::Q4nxGather gather_down(rt, npu::Q4nxGather::Kind::kDown, arena);

  const std::size_t slotted =
      npu::kGemvSlot * sizeof(float) + npu::kGemvHidden * sizeof(float);
  const std::size_t d_offset = npu::kGemvHidden * sizeof(float);
  Buffer in_x(rt, npu::kGemvInputBytes);
  Buffer in_y(rt, in.y_bytes());
  Buffer out_x(rt, npu::kGemvInputBytes);
  Buffer ffn_x(rt, npu::kGemvInputBytes);
  Buffer mid(rt, gate_up.y_bytes());
  Buffer next_x(rt, npu::kGemvInputBytes);
  Buffer next_y(rt, next.y_bytes());
  Buffer next_r(rt, npu::kGemvResidualBytes);
  Buffer scratch(rt, npu::kGemvResidualBytes);
  for (Buffer* b : {&in_x, &ffn_x, &next_x}) {
    auto* norm = reinterpret_cast<std::uint16_t*>(b->as<std::uint8_t>() +
                                                  npu::kGemvNormOffset);
    for (std::size_t i = 0; i < hidden; ++i) {
      norm[i] = quant::FloatToBf16(1.0f + 0.2f * act(rng));
    }
  }
  for (std::size_t i = 0; i < 2 * hidden; ++i) {
    in_x.as<float>()[i] = act(rng);  // H and D
    out_x.as<float>()[i] = act(rng);
  }
  for (Buffer* b : {&in_x, &out_x, &ffn_x, &next_x}) {
    b->ToDevice();
  }
  Buffer ffn_h = ffn_x.Sub(0, slotted);
  Buffer ffn_d = ffn_x.Sub(d_offset, slotted);
  Buffer next_h = next_x.Sub(0, slotted);
  Buffer next_d = next_x.Sub(d_offset, slotted);
  npu::Chain first(rt);
  first.Add(in, in_x, in_y, ffn_h);
  first.Add(out, out_x, ffn_d, scratch);
  npu::Chain second(rt);
  second.Add(gate_up, ffn_x, arena_buf, mid, next_h);
  second.Add(gather_down, mid, arena_buf, next_d, scratch);
  second.Add(next, next_x, next_y, next_r);

  // Fixed selections: 9 distinct arena experts and their scales.
  std::vector<std::vector<int>> ids(kSelections);
  std::vector<std::vector<float>> scales(kSelections);
  for (int s = 0; s < kSelections; ++s) {
    std::vector<int> all(kArena);
    for (int e = 0; e < kArena; ++e) {
      all[e] = e;
    }
    std::shuffle(all.begin(), all.end(), rng);
    ids[s].assign(all.begin(), all.begin() + npu::kGatherExperts);
    for (std::size_t i = 0; i < npu::kGatherExperts; ++i) {
      scales[s].push_back(
          std::uniform_real_distribution<float>(0.05f, 1.0f)(rng));
    }
  }

  struct Watched {
    const char* name;
    Buffer* buffer;
  };
  const Watched watched[] = {{"in_y", &in_y},     {"ffn_x", &ffn_x},
                             {"mid", &mid},       {"next_x", &next_x},
                             {"next_y", &next_y}, {"next_r", &next_r}};
  std::vector<std::vector<std::uint8_t>> want[kSelections];
  long bad_runs = 0;
  for (long it = 0; it < iterations; ++it) {
    const auto s = static_cast<std::size_t>(it % kSelections);
    first.Run();
    gate_up.Select(ids[s].data(), scales[s].data());
    gather_down.Select(ids[s].data());
    second.Run();
    bool bad = false;
    for (std::size_t w = 0; w < std::size(watched); ++w) {
      Buffer& b = *watched[w].buffer;
      b.FromDevice();
      const auto* p = b.as<std::uint8_t>();
      if (want[s].size() <= w) {
        want[s].emplace_back(p, p + b.size());
        continue;
      }
      if (std::memcmp(p, want[s][w].data(), b.size()) != 0 && !bad) {
        std::size_t i = 0;
        while (p[i] == want[s][w][i]) {
          ++i;
        }
        std::fprintf(stderr,
                     "run %ld (selection %zu): %s first differs at "
                     "byte %zu\n",
                     it, s, watched[w].name, i);
        bad = true;
      }
    }
    bad_runs += bad ? 1 : 0;
  }
  // Different selections must give different outputs (the patches apply):
  // next_y of the first two.
  const bool distinct =
      iterations < 2 ||
      std::memcmp(want[0][4].data(), want[1][4].data(), want[0][4].size()) != 0;
  std::printf(
      "%s MoE layer chain: %ld runs over %d expert selections, %ld with any "
      "output differing%s\n",
      bad_runs == 0 && distinct ? "ok  " : "FAIL", iterations, kSelections,
      bad_runs, distinct ? "" : "; selections gave the same output");
  return bad_runs == 0 && distinct ? EXIT_SUCCESS : EXIT_FAILURE;
}
