#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/gemv.hpp"

// Determinism of an MoE layer's calls, wired as NpuDecoder wires a
// Qwen3.6-35B-A3B layer (instruction-buffer flow): the input projection and
// the mixer's output projection in one submission, then, with the gathers
// pointed at a new selection, gate + up and down gathers and the next
// layer's input projection in another. The selections cycle through fixed
// ones, for one token (9 experts) and for two (unions of 9, 13 and 17
// experts, each token weighing some of them 0); every output must be
// bit-identical to the first run with the same selection. Guards the DMA
// wait-order rule (AGENTS.md) for the gather sequences, whose many segment
// fills reuse BDs. argv[2] overrides the iteration count. Exits 77 (skip)
// without an NPU.
namespace {

namespace npu = gufo_npu::npu;
namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
using npu::Buffer;
using npu::Epilogue;
using npu::Prologue;

constexpr long kDefaultIterations = 20000;
constexpr int kArena = 24;

// A selection: experts per call and tokens.
struct Selection {
  std::size_t experts;
  std::size_t tokens;
};
constexpr Selection kSelections[] = {{9, 1}, {9, 1}, {9, 2}, {13, 2}, {17, 2}};
constexpr std::size_t kCount = std::size(kSelections);

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

  const std::size_t slotted =
      npu::kGemvSlot * sizeof(float) + npu::kGemvHidden * sizeof(float);
  const std::size_t d_offset = npu::kGemvHidden * sizeof(float);
  Buffer in_x(rt, npu::kGemvInputBytes);
  Buffer in_y(rt, in.y_bytes());
  Buffer out_x(rt, npu::kGemvInputBytes);
  Buffer ffn_x(rt, npu::kGemvInputBytes);
  Buffer mid(
      rt, npu::Q4nxGather(rt, npu::Q4nxGather::Kind::kGateUp, arena).y_bytes());
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
  for (std::size_t t = 0; t < npu::kGemvTokens; ++t) {
    for (std::size_t i = 0; i < 2 * hidden; ++i) {
      in_x.as<float>()[t * npu::kGemvSlot + i] = act(rng);  // H and D
      out_x.as<float>()[t * npu::kGemvSlot + i] = act(rng);
    }
  }
  for (Buffer* b : {&in_x, &out_x, &ffn_x, &next_x}) {
    b->ToDevice();
  }
  Buffer ffn_h = ffn_x.Sub(0, slotted);
  Buffer ffn_d = ffn_x.Sub(d_offset, slotted);
  Buffer next_h = next_x.Sub(0, slotted);
  Buffer next_d = next_x.Sub(d_offset, slotted);
  // Per selection: its gathers and chains, distinct arena experts (sorted,
  // as the decoder orders them) and their scales.
  struct Setup {
    std::unique_ptr<npu::Q4nxGather> gate_up, down;
    std::unique_ptr<npu::Chain> first, second;
    std::vector<int> ids;
    std::vector<float> scales;
  };
  std::vector<Setup> setups;
  for (const Selection& sel : kSelections) {
    Setup& u = setups.emplace_back();
    u.gate_up = std::make_unique<npu::Q4nxGather>(
        rt, npu::Q4nxGather::Kind::kGateUp, arena, sel.experts, sel.tokens);
    u.down = std::make_unique<npu::Q4nxGather>(rt, npu::Q4nxGather::Kind::kDown,
                                               arena, sel.experts, sel.tokens);
    u.first = std::make_unique<npu::Chain>(rt);
    u.first->Add(in, in_x, in_y, ffn_h, sel.tokens);
    u.first->Add(out, out_x, ffn_d, scratch, sel.tokens);
    u.second = std::make_unique<npu::Chain>(rt);
    u.second->Add(*u.gate_up, ffn_x, arena_buf, mid, next_h);
    u.second->Add(*u.down, mid, arena_buf, next_d, scratch);
    u.second->Add(next, next_x, next_y, next_r, sel.tokens);
    std::vector<int> all(kArena);
    for (int e = 0; e < kArena; ++e) {
      all[e] = e;
    }
    std::shuffle(all.begin(), all.end(), rng);
    u.ids.assign(all.begin(), all.begin() + sel.experts);
    std::sort(u.ids.begin(), u.ids.end());
    for (std::size_t i = 0; i < sel.experts * sel.tokens; ++i) {
      const float v = std::uniform_real_distribution<float>(0.05f, 1.0f)(rng);
      const bool unused =
          sel.tokens == 2 && std::uniform_int_distribution<int>(0, 2)(rng) == 0;
      u.scales.push_back(unused ? 0.0f : v);
    }
  }

  // The bytes a selection writes in each output, per token (a smaller call
  // leaves the rest of a buffer as the previous one wrote it): `bytes` at
  // offset + t * stride.
  struct Watched {
    const char* name;
    Buffer* buffer;
    std::size_t offset;
    std::size_t stride;
    std::size_t bytes;  // 0: the selection's experts' SwiGLU outputs
  };
  const std::size_t slot = npu::kGemvSlot * sizeof(float);
  const std::size_t hidden_bytes = hidden * sizeof(float);
  const Watched watched[] = {
      {"in_y", &in_y, 0, in.y_token_stride() * sizeof(float),
       in.rows() * sizeof(float)},
      {"ffn_x", &ffn_x, 0, slot, 2 * hidden_bytes},
      {"mid", &mid, 0, npu::kGemvMaxDim / 2 * sizeof(float), 0},
      {"next_x", &next_x, 0, slot, 2 * hidden_bytes},
      {"next_y", &next_y, 0, next.y_token_stride() * sizeof(float),
       next.rows() * sizeof(float)},
      {"next_r", &next_r, 0, slot, hidden_bytes}};
  std::vector<std::vector<std::uint8_t>> want[kCount];
  long bad_runs = 0;
  for (long it = 0; it < iterations; ++it) {
    const auto s = static_cast<std::size_t>(it) % kCount;
    Setup& u = setups[s];
    u.first->Run();
    u.gate_up->Select(u.ids.data(), u.scales.data());
    u.down->Select(u.ids.data());
    u.second->Run();
    bool bad = false;
    for (std::size_t w = 0; w < std::size(watched); ++w) {
      const Watched& o = watched[w];
      o.buffer->FromDevice();
      const auto* p = o.buffer->as<std::uint8_t>();
      const std::size_t bytes = o.bytes != 0
                                    ? o.bytes
                                    : kSelections[s].experts * npu::kExpertFfn *
                                          sizeof(std::uint16_t);
      std::vector<std::uint8_t> got;
      for (std::size_t t = 0; t < kSelections[s].tokens; ++t) {
        const auto* q = p + o.offset + t * o.stride;
        got.insert(got.end(), q, q + bytes);
      }
      if (want[s].size() <= w) {
        want[s].push_back(std::move(got));
        continue;
      }
      if (got != want[s][w] && !bad) {
        std::size_t i = 0;
        while (got[i] == want[s][w][i]) {
          ++i;
        }
        std::fprintf(stderr,
                     "run %ld (selection %zu): %s first differs at "
                     "byte %zu of what it writes\n",
                     it, s, o.name, i);
        bad = true;
      }
    }
    bad_runs += bad ? 1 : 0;
  }
  // Different selections must give different outputs (the patches apply):
  // next_y of the two one-token selections.
  const bool distinct =
      iterations < 2 ||
      std::memcmp(want[0][4].data(), want[1][4].data(), want[0][4].size()) != 0;
  std::printf(
      "%s MoE layer chain: %ld runs over %zu expert selections (1 token: 9 "
      "experts; 2 tokens: 9, 13, 17), %ld with any output differing%s\n",
      bad_runs == 0 && distinct ? "ok  " : "FAIL", iterations, kCount, bad_runs,
      distinct ? "" : "; selections gave the same output");
  return bad_runs == 0 && distinct ? EXIT_SUCCESS : EXIT_FAILURE;
}
