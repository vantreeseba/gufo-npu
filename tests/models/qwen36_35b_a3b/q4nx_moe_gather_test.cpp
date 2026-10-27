#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "npu/gemv.hpp"

// MoE FFN on the NPU with experts chosen per call (npu::Q4nxGather,
// docs/research/NPU-MOE.md): a gate + up gather (one expert per round,
// SwiGLU scaled by its router weight) and a down gather (K = the experts'
// 512-wide outputs back to back), whose weight offsets and scales Select()
// rewrites before each call. Checked against the CPU over the same q4nx
// weights for random selections: one token over 9 experts, and two tokens
// over unions of 9, 13 and 17 experts in which each token weighs some
// experts 0 (those only the other token chose). Also prints the patch and
// layer times. Exits 77 (skip) without an NPU.
namespace {

namespace npu = gufo_npu::npu;
namespace q4nx = gufo_npu::q4nx;
namespace quant = gufo_npu::quant;
using Clock = std::chrono::steady_clock;

constexpr int kArena = 20;  // experts to choose from
constexpr int kFfn = static_cast<int>(npu::kExpertFfn);
constexpr int kHidden = static_cast<int>(npu::kGemvHidden);
constexpr double kMaxError = 1e-2;

double MicrosSince(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start)
      .count();
}

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
  npu::GemvRuntime rt(argv[1], npu::Flow::kInstructionBuffer);
  std::mt19937 rng(5);
  std::normal_distribution<float> weight(0.0f, 0.02f);
  std::normal_distribution<float> act(0.0f, 1.0f);

  // The arena: every expert's gate + up (1024 x 2048, SwiGLU order) and down
  // (2048 x 512).
  const npu::ExpertArena arena{kArena};
  npu::Buffer arena_buf(rt, arena.bytes());
  std::vector<std::vector<float>> up_w;
  std::vector<std::vector<float>> down_w;
  for (int e = 0; e < kArena; ++e) {
    std::vector<std::uint16_t> bits(std::size_t{2} * kFfn * kHidden);
    for (auto& b : bits) {
      b = quant::FloatToBf16(weight(rng));
    }
    const auto up = q4nx::PackRows(
        {{gufo_npu::gguf::GgmlType::kBF16,
          reinterpret_cast<const std::uint8_t*>(bits.data()), 2 * kFfn}},
        kHidden, q4nx::SwiGluOrder(kFfn));
    std::vector<float> d(std::size_t{kHidden} * kFfn);
    for (auto& v : d) {
      v = weight(rng);
    }
    const auto down = q4nx::PackFloat(d.data(), kHidden, kFfn);
    std::memcpy(arena_buf.as<std::uint8_t>() + arena.UpOffset(e),
                up.bytes.data(), up.bytes.size());
    std::memcpy(arena_buf.as<std::uint8_t>() + arena.DownOffset(e),
                down.bytes.data(), down.bytes.size());
    up_w.push_back(q4nx::Unpack(up));
    down_w.push_back(q4nx::Unpack(down));
  }
  arena_buf.ToDevice();

  // Both tokens' resnorm inputs: H, D and the norm weight.
  npu::Buffer x(rt, npu::kGemvInputBytes);
  auto* norm_bits = reinterpret_cast<std::uint16_t*>(x.as<std::uint8_t>() +
                                                     npu::kGemvNormOffset);
  for (int i = 0; i < kHidden; ++i) {
    norm_bits[i] = quant::FloatToBf16(1.0f + 0.2f * act(rng));
  }
  // xn[t]: the bf16 input the kernel multiplies for token t.
  std::vector<std::vector<float>> xn(npu::kGemvTokens,
                                     std::vector<float>(kHidden));
  for (std::size_t t = 0; t < npu::kGemvTokens; ++t) {
    float* slot = x.as<float>() + t * npu::kGemvSlot;
    std::vector<float> h(kHidden);
    double squares = 0;
    for (int i = 0; i < kHidden; ++i) {
      slot[i] = act(rng) * 3;
      slot[kHidden + i] = act(rng);
      h[i] = slot[i] + slot[kHidden + i];
      squares += static_cast<double>(h[i]) * h[i];
    }
    const float inv =
        1.0f / std::sqrt(static_cast<float>(squares / kHidden) + 1e-6f);
    for (int i = 0; i < kHidden; ++i) {
      xn[t][i] = Bf16Round(h[i] * inv * quant::Bf16ToFloat(norm_bits[i]));
    }
  }
  x.ToDevice();

  bool ok = true;
  double layer_us = 0;
  double patch_us = 0;
  struct Case {
    int experts;
    std::size_t tokens;
  };
  for (const Case call : {Case{9, 1}, Case{9, 2}, Case{13, 2}, Case{17, 2}}) {
    const int count = call.experts;
    const std::size_t tokens = call.tokens;
    npu::Q4nxGather up(rt, npu::Q4nxGather::Kind::kGateUp, arena,
                       static_cast<std::size_t>(count), tokens);
    npu::Q4nxGather down(rt, npu::Q4nxGather::Kind::kDown, arena,
                         static_cast<std::size_t>(count), tokens);
    npu::Buffer mid(rt, up.y_bytes());
    npu::Buffer y(rt, down.y_bytes());
    npu::Buffer r(rt, npu::kGemvResidualBytes);
    npu::Buffer scratch(rt, npu::kGemvResidualBytes);
    npu::Chain layer(rt);
    layer.Add(up, x, arena_buf, mid, r);
    layer.Add(down, mid, arena_buf, y, scratch);

    double worst = 0;
    std::vector<int> experts(kArena);
    for (int trial = 0; trial < 20; ++trial) {
      for (int i = 0; i < kArena; ++i) {
        experts[i] = i;
      }
      std::shuffle(experts.begin(), experts.end(), rng);
      // scales[i * tokens + t]: each token weighs the experts differently,
      // and with two tokens about a third of each token's are 0.
      std::vector<float> scales(count * tokens);
      for (auto& s : scales) {
        s = std::uniform_real_distribution<float>(0.05f, 1.0f)(rng);
        if (tokens == 2 && std::uniform_int_distribution<int>(0, 2)(rng) == 0) {
          s = 0.0f;
        }
      }
      up.Select(experts.data(), scales.data());
      down.Select(experts.data());
      layer.Run();
      y.FromDevice();
      for (std::size_t t = 0; t < tokens; ++t) {
        std::vector<double> want(kHidden, 0.0);
        for (int i = 0; i < count; ++i) {
          const auto& u = up_w[experts[i]];
          std::vector<float> a(kFfn);
          // SwiGLU order: each 32-row tile is 16 gate rows, then their up
          // rows.
          for (int tile = 0; tile < 2 * kFfn / 32; ++tile) {
            for (int j = 0; j < 16; ++j) {
              double g = 0;
              double v = 0;
              const float* gate =
                  &u[static_cast<std::size_t>(tile * 32 + j) * kHidden];
              const float* upr = gate + std::size_t{16} * kHidden;
              for (int k = 0; k < kHidden; ++k) {
                g += static_cast<double>(gate[k]) * xn[t][k];
                v += static_cast<double>(upr[k]) * xn[t][k];
              }
              a[tile * 16 + j] = Bf16Round(static_cast<float>(
                  g / (1 + std::exp(-g)) * v * scales[i * tokens + t]));
            }
          }
          const auto& d = down_w[experts[i]];
          for (int o = 0; o < kHidden; ++o) {
            double acc = 0;
            for (int k = 0; k < kFfn; ++k) {
              acc += static_cast<double>(
                         d[static_cast<std::size_t>(o) * kFfn + k]) *
                     a[k];
            }
            want[o] += acc;
          }
        }
        double diff = 0;
        double norm = 0;
        const float* got = y.as<float>() + t * npu::kGemvSlot;
        for (int o = 0; o < kHidden; ++o) {
          diff += (got[o] - want[o]) * (got[o] - want[o]);
          norm += want[o] * want[o];
        }
        worst = std::max(worst, std::sqrt(diff / norm));
      }
    }
    const bool pass = worst < kMaxError;
    ok = ok && pass;
    std::printf(
        "%s MoE gather, %zu token(s), %d of %d experts with router scales, "
        "vs CPU: worst rel err %.2e over 20 selections\n",
        pass ? "ok  " : "FAIL", tokens, count, kArena, worst);

    if (tokens == 1) {
      std::vector<float> scales(count, 0.5f);
      patch_us = 1e9;
      layer_us = 1e9;
      for (int i = 0; i < 100; ++i) {
        auto start = Clock::now();
        up.Select(experts.data(), scales.data());
        down.Select(experts.data());
        patch_us = std::min(patch_us, MicrosSince(start));
        start = Clock::now();
        layer.Run();
        layer_us = std::min(layer_us, MicrosSince(start));
      }
    }
  }
  std::printf(
      "     patch %.1f us; layer FFN (2 calls, 1 submission) %.1f us, "
      "%.1f GB/s\n",
      patch_us, layer_us,
      static_cast<double>(npu::kGatherExperts) *
          static_cast<double>(npu::ExpertArena::kUpBytes +
                              npu::ExpertArena::kDownBytes) /
          layer_us / 1e3);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
