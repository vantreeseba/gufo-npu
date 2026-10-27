#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/gemv.hpp"
#include "npu/xrt_kernel.hpp"

// MoE FFN on the NPU with experts chosen per call (docs/research/NPU-MOE.md):
// a gate + up gather (one expert per round, SwiGLU scaled by its router
// weight) and a down gather (K = the experts' 512-wide outputs back to back),
// whose weight offsets and scales are rewritten in their instruction streams
// before each call. Checked against the CPU over the same q4nx weights for
// random selections; also prints the patch and layer times. Exits 77 (skip)
// without an NPU.
namespace {

namespace npu = gufo_npu::npu;
namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
using Clock = std::chrono::steady_clock;

constexpr int kCols = 8;
constexpr int kTokens = 2;
constexpr int kExperts = 9;  // per call: 8 routed and the shared expert
constexpr int kArena = 16;   // experts to choose from
constexpr int kFfn = 512;
constexpr int kHidden = 2048;
constexpr std::uint32_t kScaleSentinel = 0x7FA50000;  // see the design
constexpr double kMaxError = 1e-2;

double MicrosSince(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start)
      .count();
}

std::vector<std::uint32_t> ReadWords(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
  std::vector<std::uint32_t> words(bytes.size() / 4);
  std::memcpy(words.data(), bytes.data(), words.size() * 4);
  return words;
}

// Word positions of the DDR patches (opcode 0x81) of buffer argument `arg`,
// in stream order (layout: aie/Runtime/TxnEncoding.h).
std::vector<std::size_t> Patches(const std::vector<std::uint32_t>& w,
                                 std::uint32_t arg) {
  std::vector<std::size_t> out;
  std::size_t pos = 4;  // after the header
  while (pos < w.size()) {
    const std::uint32_t op = w[pos] & 0xFF;
    std::size_t len = 0;
    if (op == 0) {
      len = 6;  // write32
    } else if (op == 3 || op == 4) {
      len = 7;  // maskwrite32, maskpoll32
    } else if (op == 1) {
      len = w[pos + 3] / 4;  // blockwrite
    } else if (op == 0x80 || op == 0x81) {
      len = w[pos + 1] / 4;  // tct, ddr patch
    } else {
      throw std::runtime_error("unexpected instruction opcode");
    }
    if (op == 0x81 && w[pos + 8] == arg) {
      out.push_back(pos);
    }
    pos += len;
  }
  return out;
}

void SetPlus(std::uint32_t* w, std::size_t site, std::uint64_t offset) {
  w[site + 10] = static_cast<std::uint32_t>(offset);
  w[site + 11] = static_cast<std::uint32_t>(offset >> 32);
}

// A gather call: its own instruction buffer (legacy flow), the weight patch
// sites with their placeholder offsets, and the scale words.
struct Gather {
  std::vector<std::uint32_t> words;
  std::vector<std::size_t> sites;
  std::vector<std::uint64_t> placeholders;
  std::vector<std::pair<std::size_t, int>> scales;  // (word, round * 2 + t)
  xrt::bo instr;

  Gather(npu::HwContext& hw, xrt::kernel& kernel,
         const std::filesystem::path& path)
      : words(ReadWords(path)), sites(Patches(words, 1)) {
    for (std::size_t site : sites) {
      placeholders.push_back(words[site + 10] |
                             (std::uint64_t{words[site + 11]} << 32));
    }
    for (std::size_t i = 0; i < words.size(); ++i) {
      if (words[i] >= kScaleSentinel && words[i] < kScaleSentinel + 256) {
        scales.emplace_back(i, static_cast<int>(words[i] - kScaleSentinel));
      }
    }
    instr = xrt::bo(hw.device, words.size() * 4, XCL_BO_FLAGS_CACHEABLE,
                    kernel.group_id(1));
    std::memcpy(instr.map<void*>(), words.data(), words.size() * 4);
    instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  xrt::run Run(xrt::kernel& kernel, xrt::bo& x, xrt::bo& w, xrt::bo& y,
               xrt::bo& r) {
    xrt::run run(kernel);
    run.set_arg(0, npu::kOpcodeRun);
    run.set_arg(1, instr);
    run.set_arg(2, static_cast<std::uint32_t>(words.size()));
    run.set_arg(3, x);
    run.set_arg(4, w);
    run.set_arg(5, y);
    run.set_arg(6, r);
    return run;
  }
};

float Bf16Round(float v) {
  return quant::Bf16ToFloat(quant::FloatToBf16(v));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: q4nx_moe_gather_test <kernel-dir>\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const std::filesystem::path dir = argv[1];
  npu::HwContext hw(dir, "q4nx_gemv");
  xrt::kernel kernel(*hw.context, hw.kernel_name);
  std::mt19937 rng(5);
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> act(0.0f, 1.0f);

  // The arena: every expert's gate + up (1024 x 2048, SwiGLU order), then
  // every expert's down (2048 x 512).
  std::vector<q4nx::Matrix> ups;
  std::vector<q4nx::Matrix> downs;
  for (int e = 0; e < kArena; ++e) {
    std::vector<std::uint16_t> bits(std::size_t{2} * kFfn * kHidden);
    for (auto& b : bits) {
      b = quant::FloatToBf16(weight(rng));
    }
    ups.push_back(q4nx::PackRows(
        {{gufo_npu::gguf::GgmlType::kBF16,
          reinterpret_cast<const std::uint8_t*>(bits.data()), 2 * kFfn}},
        kHidden, q4nx::SwiGluOrder(kFfn)));
    std::vector<float> d(std::size_t{kHidden} * kFfn);
    for (auto& v : d) {
      v = weight(rng);
    }
    downs.push_back(q4nx::PackFloat(d.data(), kHidden, kFfn));
  }
  const std::size_t up_bytes = ups[0].bytes.size();
  const std::size_t down_bytes = downs[0].bytes.size();
  xrt::bo arena = hw.Buffer(kArena * (up_bytes + down_bytes));
  for (int e = 0; e < kArena; ++e) {
    std::memcpy(arena.map<char*>() + e * up_bytes, ups[e].bytes.data(),
                up_bytes);
    std::memcpy(arena.map<char*>() + kArena * up_bytes + e * down_bytes,
                downs[e].bytes.data(), down_bytes);
  }
  arena.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // One token's resnorm input: H, D and the norm weight.
  xrt::bo x = hw.Buffer(npu::kGemvInputBytes);
  std::memset(x.map<void*>(), 0, npu::kGemvInputBytes);
  float* xf = x.map<float*>();
  auto* norm_bits = reinterpret_cast<std::uint16_t*>(x.map<std::uint8_t*>() +
                                                     npu::kGemvNormOffset);
  std::vector<float> h(kHidden);
  double squares = 0;
  for (int i = 0; i < kHidden; ++i) {
    xf[i] = act(rng) * 3;
    xf[kHidden + i] = act(rng);
    h[i] = xf[i] + xf[kHidden + i];
    squares += static_cast<double>(h[i]) * h[i];
    norm_bits[i] = quant::FloatToBf16(1.0f + 0.2f * act(rng));
  }
  x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  const float inv =
      1.0f / std::sqrt(static_cast<float>(squares / kHidden) + 1e-6f);
  std::vector<float> xn(kHidden);  // the bf16 input the kernel multiplies
  for (int i = 0; i < kHidden; ++i) {
    xn[i] = Bf16Round(h[i] * inv * quant::Bf16ToFloat(norm_bits[i]));
  }

  xrt::bo mid = hw.Buffer(std::size_t{kTokens} * npu::kGemvMaxDim / 2 * 4 * 4);
  xrt::bo y = hw.Buffer((npu::kGemvSlot + kHidden) * sizeof(float));
  xrt::bo r = hw.Buffer(npu::kGemvResidualBytes);
  xrt::bo scratch = hw.Buffer(npu::kGemvResidualBytes);
  Gather up(hw, kernel,
            dir /
                "q4nx_gemv_k2048_n9216_resnorm_swiglu_scaled_rounds9_t1"
                ".insts.bin");
  Gather down(hw, kernel,
              dir / "q4nx_gemv_k4608_n2048_act_store_k9_t1.insts.bin");
  const std::uint64_t up_seg = up_bytes / kCols;            // column slice
  const std::uint64_t down_seg = down_bytes / (2 * kCols);  // (round, col)
  if (up.sites.size() != std::size_t{kExperts} * kCols ||
      down.sites.size() != std::size_t{2} * kExperts * kCols ||
      up.scales.size() != std::size_t{32} * kExperts * kTokens) {
    std::fprintf(stderr, "FAIL: unexpected patch sites\n");
    return 1;
  }

  auto patch = [&](const std::vector<int>& experts,
                   const std::vector<float>& scales) {
    std::uint32_t* w = up.instr.map<std::uint32_t*>();
    for (std::size_t i = 0; i < up.sites.size(); ++i) {
      const std::uint64_t index = up.placeholders[i] / up_seg;  // rnd*COLS+col
      SetPlus(w, up.sites[i],
              experts[index / kCols] * up_bytes + (index % kCols) * up_seg);
    }
    for (auto [word, i] : up.scales) {
      const float s = scales[i / kTokens];
      std::memcpy(&w[word], &s, sizeof(s));
    }
    up.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w = down.instr.map<std::uint32_t*>();
    for (std::size_t i = 0; i < down.sites.size(); ++i) {
      // Placeholder index: (round * COLS + col) * experts + expert.
      const std::uint64_t index = down.placeholders[i] / down_seg;
      SetPlus(w, down.sites[i],
              kArena * up_bytes + experts[index % kExperts] * down_bytes +
                  (index / kExperts) * down_seg);
    }
    down.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  };
  xrt::runlist layer(*hw.context);
  layer.add(up.Run(kernel, x, arena, mid, r));
  layer.add(down.Run(kernel, mid, arena, y, scratch));

  std::vector<std::vector<float>> up_w;
  std::vector<std::vector<float>> down_w;
  for (int e = 0; e < kArena; ++e) {
    up_w.push_back(q4nx::Unpack(ups[e]));
    down_w.push_back(q4nx::Unpack(downs[e]));
  }
  double worst = 0;
  for (int trial = 0; trial < 20; ++trial) {
    std::vector<int> experts(kArena);
    for (int i = 0; i < kArena; ++i) {
      experts[i] = i;
    }
    std::shuffle(experts.begin(), experts.end(), rng);
    experts.resize(kExperts);
    std::vector<float> scales(kExperts);
    for (auto& s : scales) {
      s = std::uniform_real_distribution<float>(0.05f, 1.0f)(rng);
    }
    patch(experts, scales);
    layer.execute();
    layer.wait();
    y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<double> want(kHidden, 0.0);
    for (int i = 0; i < kExperts; ++i) {
      const auto& u = up_w[experts[i]];
      std::vector<float> a(kFfn);
      // SwiGLU order: each 32-row tile is 16 gate rows, then their up rows.
      for (int tile = 0; tile < 2 * kFfn / 32; ++tile) {
        for (int j = 0; j < 16; ++j) {
          double g = 0;
          double v = 0;
          const float* gate =
              &u[static_cast<std::size_t>(tile * 32 + j) * kHidden];
          const float* upr = gate + std::size_t{16} * kHidden;
          for (int k = 0; k < kHidden; ++k) {
            g += static_cast<double>(gate[k]) * xn[k];
            v += static_cast<double>(upr[k]) * xn[k];
          }
          a[tile * 16 + j] = Bf16Round(
              static_cast<float>(g / (1 + std::exp(-g)) * v * scales[i]));
        }
      }
      const auto& d = down_w[experts[i]];
      for (int o = 0; o < kHidden; ++o) {
        double acc = 0;
        for (int k = 0; k < kFfn; ++k) {
          acc +=
              static_cast<double>(d[static_cast<std::size_t>(o) * kFfn + k]) *
              a[k];
        }
        want[o] += acc;
      }
    }
    double diff = 0;
    double norm = 0;
    const float* got = y.map<float*>();
    for (int o = 0; o < kHidden; ++o) {
      diff += (got[o] - want[o]) * (got[o] - want[o]);
      norm += want[o] * want[o];
    }
    worst = std::max(worst, std::sqrt(diff / norm));
  }
  const bool ok = worst < kMaxError;
  std::printf(
      "%s MoE gather (9 of %d experts, router scales) vs CPU: worst "
      "rel err %.2e over 20 selections\n",
      ok ? "ok  " : "FAIL", kArena, worst);

  std::vector<int> experts{0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> scales(kExperts, 0.5f);
  double patch_us = 1e9;
  double layer_us = 1e9;
  for (int i = 0; i < 100; ++i) {
    auto start = Clock::now();
    patch(experts, scales);
    patch_us = std::min(patch_us, MicrosSince(start));
    start = Clock::now();
    layer.execute();
    layer.wait();
    layer_us = std::min(layer_us, MicrosSince(start));
  }
  std::printf(
      "     patch %.1f us; layer FFN (2 calls, 1 submission) %.1f us, "
      "%.1f GB/s\n",
      patch_us, layer_us,
      kExperts * static_cast<double>(up_bytes + down_bytes) / layer_us / 1e3);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
