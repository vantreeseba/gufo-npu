#include "models/qwen35_2b/deltanet.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"

// The fused DeltaNet decode step (DeltaNetStep::Run) against the path it
// replaces (DeltaNetStep::Reference: the decoder's alpha/beta rows, then
// qwen35::Mixer::Linear token by token), bit for bit, on every DeltaNet
// layer's real weights: outputs, conv window, state and, for pairs, the
// speculative state and saved window, over a run of one- and two-token
// steps with drafts kept and rolled back (as NpuDecoder::Resolve does).
// With the AVX-512 delta rule and without it.
//   GUFO_NPU_QWEN35_2B_Q4_GGUF  UD-Q4_K_XL GGUF of Qwen3.5-2B
// Exits 77 (skip) when it is unset.
namespace {

namespace qwen35 = gufo_npu::qwen35;
namespace qwen35_2b = gufo_npu::qwen35_2b;

constexpr int kSteps = 120;
constexpr std::size_t kInStride = 8192;  // the input projection's rows
constexpr std::size_t kSlot = 4096;      // residual and output slots

// Bit for bit; prints the first difference.
bool SameSpan(const float* a, const float* b, std::size_t n) {
  if (std::memcmp(a, b, n * sizeof(float)) == 0) {
    return true;
  }
  std::size_t i = 0;
  while (std::memcmp(a + i, b + i, sizeof(float)) == 0) {
    ++i;
  }
  std::size_t count = 0;
  for (std::size_t j = 0; j < n; ++j) {
    count += std::memcmp(a + j, b + j, sizeof(float)) != 0;
  }
  std::printf("  first difference at %zu of %zu: %.9g vs %.9g (%zu differ)\n",
              i, n, a[i], b[i], count);
  return false;
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    std::printf("  sizes %zu vs %zu\n", a.size(), b.size());
    return false;
  }
  return SameSpan(a.data(), b.data(), a.size());
}

// Fails on the first difference, naming it.
int Check(bool same, const char* what, int il, int step, bool wide) {
  if (same) {
    return 0;
  }
  std::printf("FAIL: layer %d step %d (%s): %s differs\n", il, step,
              wide ? "AVX-512" : "AVX2", what);
  return 1;
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_NPU_QWEN35_2B_Q4_GGUF");
  if (model == nullptr || *model == '\0') {
    std::puts("SKIP: set GUFO_NPU_QWEN35_2B_Q4_GGUF");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(model);
  qwen35::Mixer mixer(*file);
  const qwen35::Config& c = mixer.config();
  qwen35_2b::DeltaNetStep step(*file, c, mixer.pool());
  const bool has_wide = step.wide();
  const std::size_t value_dim = static_cast<std::size_t>(c.LinearValueDim());

  int total_failures = 0;
  int layers = 0;
  for (const bool wide : {true, false}) {
    int failures = 0;
    if (wide && !has_wide) {
      std::puts("note: no AVX-512 on this CPU; the AVX2 path only");
      continue;
    }
    step.set_wide(wide);
    for (int il = 0; il < c.layers && failures == 0; ++il) {
      if (c.IsFullAttention(il)) {
        continue;
      }
      ++layers;
      qwen35::DecodeState fused_state = mixer.NewState();
      qwen35::DecodeState ref_state = mixer.NewState();
      auto& fused = fused_state.layers[il];
      auto& ref = ref_state.layers[il];
      std::mt19937 rng(1234 + il);
      std::normal_distribution<float> normal(0.0f, 1.0f);
      std::vector<float> in(2 * kInStride);
      std::vector<float> h(2 * kSlot);
      std::vector<float> y_fused(2 * kSlot);
      std::vector<float> y_ref(2 * kSlot);
      for (int s = 0; s < kSteps && failures == 0; ++s) {
        const int count = s % 3 == 0 ? 1 : 2;
        for (float& v : in) {
          v = normal(rng);
        }
        for (float& v : h) {
          v = 4.0f * normal(rng);
        }
        step.Run(il, count, in.data(), kInStride, h.data(), kSlot, fused,
                 y_fused.data(), kSlot);
        step.Reference(mixer, il, count, in.data(), kInStride, h.data(), kSlot,
                       ref, y_ref.data(), kSlot);
        for (int t = 0; t < count; ++t) {
          failures += Check(SameSpan(y_fused.data() + t * kSlot,
                                     y_ref.data() + t * kSlot, value_dim),
                            t == 0 ? "output (token 0)" : "output (token 1)",
                            il, s, wide);
        }
        failures +=
            Check(Same(fused.conv, ref.conv), "conv window", il, s, wide);
        failures += Check(Same(fused.ssm, ref.ssm), "state", il, s, wide);
        if (count == 2) {
          failures += Check(Same(fused.ssm_spec, ref.ssm_spec),
                            "speculative state", il, s, wide);
          failures += Check(Same(fused.conv_saved, ref.conv_saved),
                            "saved conv window", il, s, wide);
          // Keep four drafts in five, roll back the rest (Resolve).
          for (auto* st : {&fused, &ref}) {
            if (s % 5 != 0) {
              st->ssm.swap(st->ssm_spec);
            } else {
              st->conv.swap(st->conv_saved);
            }
          }
        }
      }
    }
    total_failures += failures;
  }
  if (total_failures != 0) {
    return 1;
  }
  std::printf("PASS: %d layer runs x %d steps bit-identical\n", layers, kSteps);
  return 0;
}
