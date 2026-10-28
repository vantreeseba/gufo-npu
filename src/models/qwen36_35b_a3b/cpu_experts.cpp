#include "models/qwen36_35b_a3b/cpu_experts.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/q4nx.hpp"
#include "core/simd.hpp"
#include "npu/gemv.hpp"

#if defined(__x86_64__)
#include <immintrin.h>
#define GUFO_CPU_EXPERTS_AVX512 1
#endif

namespace gufo_npu::qwen36_35b_a3b {
namespace {

using npu::ExpertArena;

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kFfn = 512;
constexpr std::size_t kTile = q4nx::kMTile;  // rows per tile
constexpr std::size_t kHalf = kTile / 2;
constexpr std::size_t kKTile = q4nx::kKTile;
constexpr std::size_t kGroups = kKTile / q4nx::kGroup;
constexpr std::size_t kUpKBlocks = kHidden / kKTile;
constexpr std::size_t kDownKBlocks = kFfn / kKTile;
constexpr std::size_t kUpTiles = 2 * kFfn / kTile;
constexpr std::size_t kDownTiles = kHidden / kTile;
// A tile's weights as bf16 pairs: per pair of inputs (k, k + 1), the 32 rows'
// two weights, so one 64-byte load gives 16 rows of a pair.
constexpr std::size_t kPairBf16 = kTile * 2;
constexpr std::size_t kUpPairs = kHidden / 2;
constexpr std::size_t kDownPairs = kFfn / 2;
// Tokens per inner pass (accumulators: 2 x kTokens vectors) and per gate +
// up job.
constexpr std::size_t kTokens = 8;
constexpr std::size_t kRun = 256;
// Byte offsets in a q4nx block (src/core/q4nx.hpp).
constexpr std::size_t kD = 0;
constexpr std::size_t kDmin = kD + kTile * 2;
constexpr std::size_t kSc = kDmin + kTile * 2;
constexpr std::size_t kMn = kSc + kGroups * kTile;
constexpr std::size_t kCodes = kMn + kGroups * kTile;
static_assert(kCodes + kTile * kKTile / 2 == q4nx::kBlockBytes);
static_assert(ExpertArena::kUpBytes ==
              kUpTiles * kUpKBlocks * q4nx::kBlockBytes);
static_assert(ExpertArena::kDownBytes ==
              kDownTiles * kDownKBlocks * q4nx::kBlockBytes);

// The block of row tile `tile` and k block `kb` (q4nx's GEMV order).
const std::uint8_t* BlockAt(const std::uint8_t* matrix, std::size_t k_blocks,
                            std::size_t tile, std::size_t kb) {
  const std::size_t index =
      ((tile / q4nx::kRows) * k_blocks + kb) * q4nx::kRows + tile % q4nx::kRows;
  return matrix + index * q4nx::kBlockBytes;
}

float Bf16At(const std::uint8_t* p) {
  std::uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return simd::FromBf16(v);
}

// Row tile `tile` of a q4nx matrix of `k_blocks` k blocks into `wt` (bf16
// pairs, k_blocks * kKTile / 2 pairs of kPairBf16): w = dmin * mn + d * sc *
// code, rounded once to bf16.
void UnpackTilePortable(const std::uint8_t* matrix, std::size_t k_blocks,
                        std::size_t tile, std::uint16_t* wt) {
  for (std::size_t kb = 0; kb < k_blocks; ++kb) {
    const std::uint8_t* block = BlockAt(matrix, k_blocks, tile, kb);
    for (std::size_t m = 0; m < kTile; ++m) {
      const float d = Bf16At(block + kD + m * 2);
      const float dmin = Bf16At(block + kDmin + m * 2);
      for (std::size_t k = 0; k < kKTile; ++k) {
        const std::size_t g = k / q4nx::kGroup;
        const float step = d * static_cast<float>(block[kSc + g * kTile + m]);
        const float min = dmin * static_cast<float>(static_cast<std::int8_t>(
                                     block[kMn + g * kTile + m]));
        const std::size_t i =
            (((m / 16) * kGroups + g) * 32 + k % 32) * 16 + m % 16;
        const std::uint8_t byte = block[kCodes + i / 2];
        const int code = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
        const std::size_t pair = (kb * kKTile + k) / 2;
        wt[(pair * kTile + m) * 2 + k % 2] =
            simd::ToBf16(min + step * static_cast<float>(code));
      }
    }
  }
}

// out[t][r] = sum over the pairs p of wt[p][r] . x[t][p] for `tokens` rows of
// bf16 x (`pairs` pairs each), r < kTile.
void DotsPortable(const std::uint16_t* wt, std::size_t pairs,
                  const std::uint16_t* const* x, std::size_t tokens,
                  float* out) {
  for (std::size_t t = 0; t < tokens; ++t) {
    float acc[kTile] = {};
    for (std::size_t p = 0; p < pairs; ++p) {
      const float x0 = simd::FromBf16(x[t][2 * p]);
      const float x1 = simd::FromBf16(x[t][2 * p + 1]);
      const std::uint16_t* w = wt + p * kPairBf16;
      for (std::size_t r = 0; r < kTile; ++r) {
        acc[r] +=
            simd::FromBf16(w[2 * r]) * x0 + simd::FromBf16(w[2 * r + 1]) * x1;
      }
    }
    std::copy_n(acc, kTile, out + t * kTile);
  }
}

#ifdef GUFO_CPU_EXPERTS_AVX512
#define GUFO_FAST_ISA "avx512f,avx512bw,avx512vl,avx512vbmi,avx512bf16,fma"
#define GUFO_FAST __attribute__((target(GUFO_FAST_ISA)))

bool HasFast() {
  static const bool has = __builtin_cpu_supports("avx512f") &&
                          __builtin_cpu_supports("avx512bw") &&
                          __builtin_cpu_supports("avx512vbmi") &&
                          __builtin_cpu_supports("avx512bf16");
  return has;
}

// Byte permutations from a 64-byte run of codes (8 inputs l of one half and
// group: low nibbles in `lo`, high in `hi`; lo[l * 8 + b] is row 2b's code,
// hi[l * 8 + b] row 2b + 1's) to pairs (l, l + 1) per row: byte
// q * 32 + r * 2 + s is row r's code of input 2q + s, for pairs q = 0, 1
// (kPermLow) and 2, 3 (kPermHigh).
struct Perms {
  alignas(64) std::uint8_t low[64];
  alignas(64) std::uint8_t high[64];
  Perms() {
    for (int q = 0; q < 4; ++q) {
      for (int r = 0; r < 16; ++r) {
        for (int s = 0; s < 2; ++s) {
          const int value = ((r & 1) != 0 ? 64 : 0) + (2 * q + s) * 8 + r / 2;
          const int byte = (q % 2) * 32 + r * 2 + s;
          (q < 2 ? low : high)[byte] = static_cast<std::uint8_t>(value);
        }
      }
    }
  }
};

// One pair's 16 rows: `codes` 32 bytes (row r's two codes at 2r, 2r + 1) to
// bf16 pairs at `dst`.
GUFO_FAST inline void StorePair(__m128i codes_a, __m128i codes_b, __m512 step_a,
                                __m512 min_a, __m512 step_b, __m512 min_b,
                                std::uint16_t* dst) {
  const __m512 wa = _mm512_fmadd_ps(
      _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(codes_a)), step_a, min_a);
  const __m512 wb = _mm512_fmadd_ps(
      _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(codes_b)), step_b, min_b);
  _mm512_storeu_si512(dst, (__m512i)_mm512_cvtne2ps_pbh(wb, wa));
}

GUFO_FAST void UnpackTileFast(const std::uint8_t* matrix, std::size_t k_blocks,
                              std::size_t tile, std::uint16_t* wt) {
  static const Perms perms;
  const __m512i perm_low = _mm512_load_si512(perms.low);
  const __m512i perm_high = _mm512_load_si512(perms.high);
  const __m512i nibble = _mm512_set1_epi8(0x0F);
  // Lane 2r + s takes row r's value (rows 0-7, then 8-15).
  const __m512i dup_a =
      _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i dup_b = _mm512_add_epi32(dup_a, _mm512_set1_epi32(8));
  for (std::size_t kb = 0; kb < k_blocks; ++kb) {
    const std::uint8_t* block = BlockAt(matrix, k_blocks, tile, kb);
    for (std::size_t h = 0; h < 2; ++h) {
      const __m512 d = _mm512_castsi512_ps(_mm512_slli_epi32(
          _mm512_cvtepu16_epi32(_mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(block + kD + h * kHalf * 2))),
          16));
      const __m512 dmin = _mm512_castsi512_ps(_mm512_slli_epi32(
          _mm512_cvtepu16_epi32(_mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(block + kDmin + h * kHalf * 2))),
          16));
      for (std::size_t g = 0; g < kGroups; ++g) {
        const __m512 sc = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                block + kSc + g * kTile + h * kHalf))));
        const __m512 mn = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                block + kMn + g * kTile + h * kHalf))));
        const __m512 step = _mm512_mul_ps(d, sc);    // exact
        const __m512 min = _mm512_mul_ps(dmin, mn);  // exact
        const __m512 step_a = _mm512_permutexvar_ps(dup_a, step);
        const __m512 step_b = _mm512_permutexvar_ps(dup_b, step);
        const __m512 min_a = _mm512_permutexvar_ps(dup_a, min);
        const __m512 min_b = _mm512_permutexvar_ps(dup_b, min);
        const std::uint8_t* codes = block + kCodes + (h * kGroups + g) * 32 * 8;
        for (std::size_t run = 0; run < 4; ++run) {
          const __m512i raw = _mm512_loadu_si512(codes + run * 64);
          const __m512i lo = _mm512_and_si512(raw, nibble);
          const __m512i hi =
              _mm512_and_si512(_mm512_srli_epi16(raw, 4), nibble);
          const __m512i p01 = _mm512_permutex2var_epi8(lo, perm_low, hi);
          const __m512i p23 = _mm512_permutex2var_epi8(lo, perm_high, hi);
          // Pairs of this run: inputs g * 32 + run * 8 + 2q, q = 0 .. 3.
          const std::size_t pair = kb * (kKTile / 2) + g * 16 + run * 4;
          std::uint16_t* dst = wt + pair * kPairBf16 + h * kHalf * 2;
          StorePair(_mm512_extracti32x4_epi32(p01, 0),
                    _mm512_extracti32x4_epi32(p01, 1), step_a, min_a, step_b,
                    min_b, dst);
          StorePair(_mm512_extracti32x4_epi32(p01, 2),
                    _mm512_extracti32x4_epi32(p01, 3), step_a, min_a, step_b,
                    min_b, dst + kPairBf16);
          StorePair(_mm512_extracti32x4_epi32(p23, 0),
                    _mm512_extracti32x4_epi32(p23, 1), step_a, min_a, step_b,
                    min_b, dst + 2 * kPairBf16);
          StorePair(_mm512_extracti32x4_epi32(p23, 2),
                    _mm512_extracti32x4_epi32(p23, 3), step_a, min_a, step_b,
                    min_b, dst + 3 * kPairBf16);
        }
      }
    }
  }
}

template<std::size_t T>
GUFO_FAST void DotsFastT(const std::uint16_t* wt, std::size_t pairs,
                         const std::uint16_t* const* x, float* out) {
  __m512 acc[T][2];
  for (std::size_t t = 0; t < T; ++t) {
    acc[t][0] = _mm512_setzero_ps();
    acc[t][1] = _mm512_setzero_ps();
  }
  for (std::size_t p = 0; p < pairs; ++p) {
    const __m512bh w0 = (__m512bh)_mm512_loadu_si512(wt + p * kPairBf16);
    const __m512bh w1 =
        (__m512bh)_mm512_loadu_si512(wt + p * kPairBf16 + kHalf * 2);
    for (std::size_t t = 0; t < T; ++t) {
      std::int32_t two;
      std::memcpy(&two, x[t] + 2 * p, sizeof(two));
      const __m512bh xv = (__m512bh)_mm512_set1_epi32(two);
      acc[t][0] = _mm512_dpbf16_ps(acc[t][0], w0, xv);
      acc[t][1] = _mm512_dpbf16_ps(acc[t][1], w1, xv);
    }
  }
  for (std::size_t t = 0; t < T; ++t) {
    _mm512_storeu_ps(out + t * kTile, acc[t][0]);
    _mm512_storeu_ps(out + t * kTile + kHalf, acc[t][1]);
  }
}

GUFO_FAST void DotsFast(const std::uint16_t* wt, std::size_t pairs,
                        const std::uint16_t* const* x, std::size_t tokens,
                        float* out) {
  switch (tokens) {
    case 1:
      DotsFastT<1>(wt, pairs, x, out);
      break;
    case 2:
      DotsFastT<2>(wt, pairs, x, out);
      break;
    case 3:
      DotsFastT<3>(wt, pairs, x, out);
      break;
    case 4:
      DotsFastT<4>(wt, pairs, x, out);
      break;
    case 5:
      DotsFastT<5>(wt, pairs, x, out);
      break;
    case 6:
      DotsFastT<6>(wt, pairs, x, out);
      break;
    case 7:
      DotsFastT<7>(wt, pairs, x, out);
      break;
    default:
      DotsFastT<8>(wt, pairs, x, out);
      break;
  }
}
#undef GUFO_FAST
#undef GUFO_FAST_ISA
#else
bool HasFast() {
  return false;
}
#endif

class CpuExperts : public ExpertStep {
public:
  CpuExperts(BurstPool& pool, std::vector<const std::uint8_t*> arenas,
             std::size_t arena_experts, std::size_t max_tokens, bool portable)
      : pool_(pool),
        arenas_(std::move(arenas)),
        arena_experts_(arena_experts),
        fast_(!portable && HasFast()),
        row0_(arena_experts + 1) {
    // Every token takes experts_used routed experts and the shared one; the
    // routing sets how many in all, at most (arena_experts) per token.
    act_.resize(max_tokens * std::min<std::size_t>(arena_experts, 9) * kFfn);
  }

  void Prepare(int, const ExpertRouting&, npu::Batch&) override {}

  void Run(int il, const ExpertRouting& routing, const std::uint16_t* x,
           float* h) override {
    const std::uint8_t* arena = arenas_.at(static_cast<std::size_t>(il));
    const ExpertArena layout{arena_experts_};
    // Each expert's SwiGLU rows, back to back in arena order.
    std::size_t rows = 0;
    for (std::size_t e = 0; e < arena_experts_; ++e) {
      row0_[e] = rows;
      rows += routing.tokens[e].size();
    }
    row0_[arena_experts_] = rows;
    if (rows * kFfn > act_.size()) {
      act_.resize(rows * kFfn);
    }
    // Gate + up jobs, the largest experts first.
    jobs_.clear();
    for (std::size_t e = 0; e < arena_experts_; ++e) {
      const std::size_t n = routing.tokens[e].size();
      for (std::size_t first = 0; first < n; first += kRun) {
        for (std::size_t tile = 0; tile < kUpTiles; ++tile) {
          jobs_.push_back({e, tile, first, std::min(kRun, n - first)});
        }
      }
    }
    std::stable_sort(jobs_.begin(), jobs_.end(),
                     [](const Job& a, const Job& b) { return a.n > b.n; });
    pool_.Run(jobs_.size(), [&](std::size_t i) {
      GateUp(jobs_[i], arena + layout.UpOffset(jobs_[i].expert), routing, x);
    });
    pool_.Run(kDownTiles,
              [&](std::size_t tile) { Down(tile, arena, layout, routing, h); });
  }

private:
  struct Job {
    std::size_t expert;
    std::size_t tile;
    std::size_t first;  // index in the expert's token list
    std::size_t n;      // tokens
  };

  void Unpack(const std::uint8_t* matrix, std::size_t k_blocks,
              std::size_t tile, std::uint16_t* wt) const {
#ifdef GUFO_CPU_EXPERTS_AVX512
    if (fast_) {
      UnpackTileFast(matrix, k_blocks, tile, wt);
      return;
    }
#endif
    UnpackTilePortable(matrix, k_blocks, tile, wt);
  }

  void Dots(const std::uint16_t* wt, std::size_t pairs,
            const std::uint16_t* const* rows, std::size_t tokens,
            float* out) const {
#ifdef GUFO_CPU_EXPERTS_AVX512
    if (fast_) {
      DotsFast(wt, pairs, rows, tokens, out);
      return;
    }
#endif
    DotsPortable(wt, pairs, rows, tokens, out);
  }

  // A gate + up tile over a run of an expert's tokens: 16 SwiGLU values per
  // token, times its weight, as bf16 in its row of act_.
  void GateUp(const Job& job, const std::uint8_t* up,
              const ExpertRouting& routing, const std::uint16_t* x) {
    thread_local std::vector<std::uint16_t> wt;
    wt.resize(kUpPairs * kPairBf16);
    Unpack(up, kUpKBlocks, job.tile, wt.data());
    const auto& tokens = routing.tokens[job.expert];
    const auto& weights = routing.weights[job.expert];
    const std::uint16_t* rows[kTokens];
    float out[kTokens * kTile];
    for (std::size_t s = 0; s < job.n; s += kTokens) {
      const std::size_t n = std::min(kTokens, job.n - s);
      for (std::size_t t = 0; t < n; ++t) {
        rows[t] =
            x + static_cast<std::size_t>(tokens[job.first + s + t]) * kHidden;
      }
      Dots(wt.data(), kUpPairs, rows, n, out);
      for (std::size_t t = 0; t < n; ++t) {
        const std::size_t i = job.first + s + t;
        const float scale = weights[i];
        std::uint16_t* dst =
            act_.data() + (row0_[job.expert] + i) * kFfn + job.tile * kHalf;
        const float* o = out + t * kTile;
        float a[kHalf];
        for (std::size_t j = 0; j < kHalf; ++j) {
          a[j] = o[kHalf + j] * scale;
        }
        simd::MulSiluRow(a, o, static_cast<int>(kHalf));
        simd::ToBf16Row(a, dst, static_cast<int>(kHalf));
      }
    }
  }

  // Output columns tile * 32 .. + 31 of every token: each expert's down over
  // its tokens, in arena order, added into h.
  void Down(std::size_t tile, const std::uint8_t* arena,
            const ExpertArena& layout, const ExpertRouting& routing, float* h) {
    thread_local std::vector<std::uint16_t> wt;
    wt.resize(kDownPairs * kPairBf16);
    const std::uint16_t* rows[kTokens];
    float out[kTokens * kTile];
    for (std::size_t e = 0; e < arena_experts_; ++e) {
      const auto& tokens = routing.tokens[e];
      if (tokens.empty()) {
        continue;
      }
      Unpack(arena + layout.DownOffset(e), kDownKBlocks, tile, wt.data());
      for (std::size_t s = 0; s < tokens.size(); s += kTokens) {
        const std::size_t n = std::min(kTokens, tokens.size() - s);
        for (std::size_t t = 0; t < n; ++t) {
          rows[t] = act_.data() + (row0_[e] + s + t) * kFfn;
        }
        Dots(wt.data(), kDownPairs, rows, n, out);
        for (std::size_t t = 0; t < n; ++t) {
          float* dst = h + static_cast<std::size_t>(tokens[s + t]) * kHidden +
                       tile * kTile;
          const float* o = out + t * kTile;
          for (std::size_t r = 0; r < kTile; ++r) {
            dst[r] += o[r];
          }
        }
      }
    }
  }

  BurstPool& pool_;
  std::vector<const std::uint8_t*> arenas_;
  std::size_t arena_experts_;
  bool fast_;
  // Per expert, its first row in act_ (and the total after the last).
  std::vector<std::size_t> row0_;
  std::vector<std::uint16_t> act_;  // SwiGLU rows, kFfn bf16 each
  std::vector<Job> jobs_;
};

}  // namespace

std::unique_ptr<ExpertStep> MakeCpuExperts(
    BurstPool& pool, std::vector<const std::uint8_t*> arenas,
    std::size_t arena_experts, std::size_t max_tokens, bool portable) {
  return std::make_unique<CpuExperts>(pool, std::move(arenas), arena_experts,
                                      max_tokens, portable);
}

bool CpuExpertsFast() {
  return HasFast();
}

}  // namespace gufo_npu::qwen36_35b_a3b
