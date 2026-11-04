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
#include "models/qwen36_35b_a3b/npu/gemv.hpp"
#include "npu/gemv.hpp"

// The chain sequences of Qwen3.6-35B-A3B's decode program
// (qwen36_35b_a3b_decode_chain_*) against the calls they replace: an MoE
// layer's gate + up gather, down gather and the call after them, wired as
// NpuDecoder wires them, two ways on two copies of the same activation
// buffer (npu/gemv.hpp's block layout):
//   per call  the two Q4nxGathers and the next Q4nxGemv, three runs in one
//             runlist;
//   chain     the Q4nxGatherChain, one run.
// Every chain the decoder runs is checked: into the next layer's DeltaNet or
// attention input, the LM head, the MTP draft head, and the bare chain
// (stops after the down gather). Runs alternate 1 token (9 experts) and 2
// (unions of 9, 13 and 17 experts, each token weighing some of them 0), with
// a new random selection and new router scales every run, given to both. After
// every run both copies must be bit for bit equal. argv[2] sets every chain's
// iteration count. Exits 77 (skip) without an NPU.
namespace {

namespace npu = gufo_npu::npu;
namespace geo = gufo_npu::qwen36_35b_a3b::npu;
namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
using npu::Buffer;
using npu::Epilogue;
using npu::Prologue;

// Experts in the synthetic arena (the model has 256 and the shared one).
constexpr int kArena = 32;

// Each chain and its default run length (the LM head's call takes ~5 ms).
struct Case {
  geo::ChainNext next;
  long iterations;
  std::size_t rows;  // the next call's real rows (the LM head pads its own)
};
constexpr Case kCases[] = {
    {geo::ChainNext::kDeltaNet, 20000, 12288},
    {geo::ChainNext::kAttention, 4000, 9216},
    {geo::ChainNext::kLmHead, 300, 248320},
    {geo::ChainNext::kDraftHead, 2000, 32768},
    {geo::ChainNext::kNone, 4000, 0},
};

// The selections runs cycle through: experts per call and tokens.
struct Selection {
  std::size_t experts;
  std::size_t tokens;
};
constexpr Selection kSelections[] = {{9, 1},  {9, 2}, {9, 1},
                                     {13, 2}, {9, 1}, {17, 2}};
constexpr std::size_t kCount = std::size(kSelections);

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(
        stderr,
        "usage: qwen36_moe_chain_sequence_test <kernel-dir> [iterations]\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const long forced = argc > 2 ? std::atol(argv[2]) : 0;
  npu::GemvRuntime rt(argv[1], npu::Flow::kInstructionBuffer,
                      geo::DecodeDesign());
  std::mt19937 rng(5);
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> act(0.0f, 1.0f);
  auto matrix = [&](std::size_t rows, std::size_t dim) {
    std::vector<float> w(rows * dim);
    for (auto& v : w) {
      v = weight(rng);
    }
    return q4nx::PackFloat(w.data(), rows, dim, q4nx::Fit::kMinMax);
  };
  const std::size_t hidden = npu::kGemvHidden;

  // The arena: kArena experts' gate + up and down.
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
    const auto up = q4nx::PackRows(
        {{gufo_npu::gguf::GgmlType::kBF16,
          reinterpret_cast<const std::uint8_t*>(bits.data()),
          2 * npu::kExpertFfn}},
        hidden, q4nx::Fit::kMinMax, q4nx::SwiGluOrder(npu::kExpertFfn));
    const auto down = matrix(hidden, npu::kExpertFfn);
    std::memcpy(arena_buf.as<std::uint8_t>() + arena.UpOffset(e),
                up.bytes.data(), up.bytes.size());
    std::memcpy(arena_buf.as<std::uint8_t>() + arena.DownOffset(e),
                down.bytes.data(), down.bytes.size());
  }
  arena_buf.ToDevice();

  // The per-call gathers, one pair per selection (shared by every case).
  struct Gathers {
    std::unique_ptr<npu::Q4nxGather> gate_up, down;
  };
  std::vector<Gathers> gathers;
  for (const Selection& sel : kSelections) {
    gathers.push_back(
        {std::make_unique<npu::Q4nxGather>(rt, npu::Q4nxGather::Kind::kGateUp,
                                           arena, sel.experts, sel.tokens),
         std::make_unique<npu::Q4nxGather>(rt, npu::Q4nxGather::Kind::kDown,
                                           arena, sel.experts, sel.tokens)});
  }
  // A new selection: distinct arena experts (sorted, as the decoder orders
  // them) and their scales, scales[i * tokens + t].
  std::vector<int> all(kArena);
  for (int e = 0; e < kArena; ++e) {
    all[e] = e;
  }
  auto choose = [&](const Selection& sel, std::vector<int>& ids,
                    std::vector<float>& scales) {
    std::shuffle(all.begin(), all.end(), rng);
    ids.assign(all.begin(), all.begin() + sel.experts);
    std::sort(ids.begin(), ids.end());
    scales.clear();
    for (std::size_t i = 0; i < sel.experts * sel.tokens; ++i) {
      const float v = std::uniform_real_distribution<float>(0.05f, 1.0f)(rng);
      const bool unused =
          sel.tokens == 2 && std::uniform_int_distribution<int>(0, 2)(rng) == 0;
      scales.push_back(unused ? 0.0f : v);
    }
  };

  const std::size_t hidden_bytes = hidden * sizeof(float);
  const std::size_t slotted = npu::kGemvSlot * sizeof(float) + hidden_bytes;
  const std::size_t b1 = geo::kBlockBytes;
  auto first_difference = [](const std::uint8_t* a, const std::uint8_t* b) {
    std::size_t i = 0;
    while (a[i] == b[i]) {
      ++i;
    }
    return i;
  };
  bool all_ok = true;
  for (const Case& k : kCases) {
    const long iterations = forced > 0 ? forced : k.iterations;
    const char* name = geo::ChainNextName(k.next);
    // The next call (none for the bare chain), on the regions after the
    // block: the next block's in_x, ffn_x and in_y, or a head's x, r, y.
    std::unique_ptr<npu::Q4nxGemv> next;
    if (k.rows > 0) {
      next = std::make_unique<npu::Q4nxGemv>(
          rt, matrix(k.rows, hidden), Prologue::kResNorm, Epilogue::kStore);
      if (next->padded_rows() != geo::ChainNextRows(k.next)) {
        std::printf(
            "FAIL %s: the next call has %zu padded rows, the chain %zu\n", name,
            next->padded_rows(), geo::ChainNextRows(k.next));
        all_ok = false;
        continue;
      }
    }
    Buffer next_w = next ? next->weights() : Buffer(rt, 4096);
    const std::size_t acts_bytes =
        geo::kBlockFfnX + geo::ChainWindowBytes(k.next);

    // The two copies' inputs: the FFN input region (H and D per token, its
    // norm weight) and the next region's norm weight(s); the rest zero.
    Buffer per_call(rt, acts_bytes);
    Buffer chained(rt, acts_bytes);
    {
      std::vector<std::uint8_t> init(acts_bytes, 0);
      auto norm = [&](std::size_t region) {
        auto* w = reinterpret_cast<std::uint16_t*>(
            &init[region + npu::kGemvNormOffset]);
        for (std::size_t i = 0; i < hidden; ++i) {
          w[i] = quant::FloatToBf16(1.0f + 0.2f * act(rng));
        }
      };
      norm(geo::kBlockFfnX);
      norm(b1 + geo::kBlockInX);
      if (k.next == geo::ChainNext::kDeltaNet ||
          k.next == geo::ChainNext::kAttention) {
        norm(b1 + geo::kBlockFfnX);
      }
      for (std::size_t t = 0; t < npu::kGemvTokens; ++t) {
        auto* x = reinterpret_cast<float*>(&init[geo::kBlockFfnX]) +
                  t * npu::kGemvSlot;
        for (std::size_t i = 0; i < 2 * hidden; ++i) {
          x[i] = act(rng);  // H and D
        }
      }
      for (Buffer* b : {&per_call, &chained}) {
        std::memcpy(b->data(), init.data(), acts_bytes);
        b->ToDevice();
      }
    }

    // Region `offset` of `acts`, `bytes` long, kept alive for the chains.
    std::vector<std::unique_ptr<Buffer>> regions;
    auto region = [&](Buffer& acts, std::size_t offset,
                      std::size_t bytes) -> Buffer& {
      regions.push_back(std::make_unique<Buffer>(acts.Sub(offset, bytes)));
      return *regions.back();
    };
    Buffer& window =
        region(chained, geo::kBlockFfnX, geo::ChainWindowBytes(k.next));
    // Per selection: the per-call runlist, the chain sequence and its run.
    struct Setup {
      std::unique_ptr<npu::Chain> calls, chain;
      std::unique_ptr<npu::Q4nxGatherChain> sequence;
    };
    std::vector<Setup> setups;
    for (std::size_t s = 0; s < kCount; ++s) {
      const Selection& sel = kSelections[s];
      Setup& u = setups.emplace_back();
      Gathers& g = gathers[s];
      u.calls = std::make_unique<npu::Chain>(rt);
      u.calls->Add(
          *g.gate_up, region(per_call, geo::kBlockFfnX, npu::kGemvInputBytes),
          arena_buf, region(per_call, geo::kBlockFfnY, g.gate_up->y_bytes()),
          region(per_call, b1 + geo::kBlockInX, slotted));
      u.calls->Add(
          *g.down, region(per_call, geo::kBlockFfnY, g.gate_up->y_bytes()),
          arena_buf,
          region(per_call, b1 + geo::kBlockInX + hidden_bytes, slotted),
          region(per_call, geo::kBlockScratch, geo::kScratchBytes));
      if (next) {
        u.calls->Add(
            *next, region(per_call, b1 + geo::kBlockInX, npu::kGemvInputBytes),
            region(per_call, b1 + geo::kBlockInY, next->y_bytes()),
            region(per_call, b1 + geo::kBlockFfnX, slotted), sel.tokens);
      }
      u.sequence = std::make_unique<npu::Q4nxGatherChain>(
          rt, geo::ChainSequence(k.next, sel.experts, sel.tokens), arena,
          sel.experts, sel.tokens);
      u.chain = std::make_unique<npu::Chain>(rt);
      u.chain->Add(*u.sequence, window, arena_buf, next_w);
    }

    std::vector<int> ids;
    std::vector<float> scales;
    long mismatched_runs = 0;
    long changed_runs = 0;  // runs whose output differs from the run before
    std::vector<std::uint8_t> last;
    for (long it = 0; it < iterations; ++it) {
      const auto s = static_cast<std::size_t>(it) % kCount;
      Setup& u = setups[s];
      choose(kSelections[s], ids, scales);
      gathers[s].gate_up->Select(ids.data(), scales.data());
      gathers[s].down->Select(ids.data());
      u.sequence->Select(ids.data(), scales.data());
      u.calls->Run();
      u.chain->Run();
      per_call.FromDevice();
      chained.FromDevice();
      const auto* p = per_call.as<std::uint8_t>();
      const auto* c = chained.as<std::uint8_t>();
      if (std::memcmp(c, p, acts_bytes) != 0 && ++mismatched_runs <= 5) {
        std::fprintf(stderr,
                     "%s run %ld (%zu experts, %zu tokens): chain sequence "
                     "differs from the per-call chain at byte %zu\n",
                     name, it, kSelections[s].experts, kSelections[s].tokens,
                     first_difference(c, p));
      }
      if (!last.empty() && std::memcmp(p, last.data(), acts_bytes) != 0) {
        ++changed_runs;
      }
      last.assign(p, p + acts_bytes);
    }
    // New selections must move the output (the patches apply).
    const bool moved = iterations < 2 || changed_runs > 0;
    const bool ok = mismatched_runs == 0 && moved;
    all_ok = all_ok && ok;
    std::printf(
        "%s %s chain: %ld runs (1 token: 9 experts; 2 tokens: 9, 13, 17), "
        "%ld chain-sequence runs differing from the per-call chain%s\n",
        ok ? "ok  " : "FAIL", name, iterations, mismatched_runs,
        moved ? "" : "; new selections never changed the output");
  }
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
