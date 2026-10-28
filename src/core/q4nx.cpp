#include "core/q4nx.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "core/parallel.hpp"
#include "core/quant.hpp"

namespace gufo_npu::q4nx {
namespace {

constexpr std::size_t kGroups = kKTile / kGroup;  // per block row
// Byte offsets in a block (see q4nx.hpp).
constexpr std::size_t kD = 0;
constexpr std::size_t kDmin = kD + kMTile * 2;
constexpr std::size_t kSc = kDmin + kMTile * 2;
constexpr std::size_t kMn = kSc + kGroups * kMTile;
constexpr std::size_t kCodes = kMn + kGroups * kMTile;
static_assert(kCodes + kMTile * kKTile / 2 == kBlockBytes);
static_assert(kCodes % 64 == 0, "the kernel loads codes 64 bytes at a time");
constexpr std::size_t kQ4KBytes = 144;  // block_q4_K

void CheckShape(std::size_t rows, std::size_t dim) {
  if (rows == 0 || dim == 0 || dim % kKTile != 0) {
    throw std::invalid_argument("q4nx needs dim a multiple of " +
                                std::to_string(kKTile) + ", got " +
                                std::to_string(dim));
  }
}

Matrix Allocate(std::size_t rows, std::size_t dim) {
  CheckShape(rows, dim);
  Matrix m;
  m.rows = rows;
  m.padded_rows = PaddedRows(rows);
  m.dim = dim;
  m.bytes.assign(m.padded_rows / kMTile * (dim / kKTile) * kBlockBytes, 0);
  return m;
}

// The block of row tile `tile` (kMTile out-features) and k block `kb`.
std::uint8_t* Block(Matrix& m, std::size_t tile, std::size_t kb) {
  const std::size_t k_blocks = m.dim / kKTile;
  const std::size_t index =
      ((tile / kRows) * k_blocks + kb) * kRows + tile % kRows;
  return m.bytes.data() + index * kBlockBytes;
}

const std::uint8_t* Block(const Matrix& m, std::size_t tile, std::size_t kb) {
  return Block(const_cast<Matrix&>(m), tile, kb);
}

void StoreU16(std::uint8_t* p, std::uint16_t v) {
  std::memcpy(p, &v, sizeof(v));
}

std::uint16_t LoadU16(const std::uint8_t* p) {
  std::uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

std::size_t CodeIndex(std::size_t m, std::size_t k) {
  return (((m / 16) * kGroups + k / 32) * 32 + k % 32) * 16 + m % 16;
}

// Row m's super-scale and super-minimum (bf16 bits).
void SetSuper(std::uint8_t* block, std::size_t m, std::uint16_t d,
              std::uint16_t dmin) {
  StoreU16(block + kD + m * 2, d);
  StoreU16(block + kDmin + m * 2, dmin);
}

void SetSub(std::uint8_t* block, std::size_t m, std::size_t g, std::uint8_t sc,
            std::int8_t mn) {
  block[kSc + g * kMTile + m] = sc;
  block[kMn + g * kMTile + m] = static_cast<std::uint8_t>(mn);
}

// Group g's step (d * sc) and minimum (dmin * mn) for row m, exact in fp32.
void Params(const std::uint8_t* block, std::size_t m, std::size_t g,
            float* step, float* min) {
  const float d = quant::Bf16ToFloat(LoadU16(block + kD + m * 2));
  const float dmin = quant::Bf16ToFloat(LoadU16(block + kDmin + m * 2));
  *step = d * static_cast<float>(block[kSc + g * kMTile + m]);
  *min = dmin * static_cast<float>(
                    static_cast<std::int8_t>(block[kMn + g * kMTile + m]));
}

void SetCode(std::uint8_t* block, std::size_t m, std::size_t k,
             std::uint8_t code) {
  const std::size_t i = CodeIndex(m, k);
  std::uint8_t& byte = block[kCodes + i / 2];
  byte = (i % 2 == 0) ? static_cast<std::uint8_t>((byte & 0xF0) | code)
                      : static_cast<std::uint8_t>((byte & 0x0F) | (code << 4));
}

// The smallest bf16 at or above x (x >= 0).
std::uint16_t Bf16AtLeast(float x) {
  std::uint16_t bits = quant::FloatToBf16(x);
  if (quant::Bf16ToFloat(bits) < x) {
    ++bits;
  }
  return bits;
}

// One Q4_K super-block (256 in-features of out-feature m) into a q4nx block.
void PackQ4KSuper(const std::uint8_t* src, std::uint8_t* block, std::size_t m) {
  const float d = quant::HalfToFloat(LoadU16(src));
  const float dmin = quant::HalfToFloat(LoadU16(src + 2));
  const std::uint8_t* scales = src + 4;
  const std::uint8_t* qs = src + 16;
  SetSuper(block, m, quant::FloatToBf16(d), quant::FloatToBf16(dmin));
  for (std::size_t g = 0; g < kGroups; ++g) {
    std::uint8_t sc;
    std::uint8_t mn;
    quant::ScaleMinK4(g, scales, &sc, &mn);  // 6 bits each
    SetSub(block, m, g, sc, static_cast<std::int8_t>(-mn));
    // Sub-blocks pair up: 2p in the low nibbles of qs[32p..], 2p+1 high.
    const std::uint8_t* q = qs + (g / 2) * 32;
    const int shift = (g % 2) * 4;
    for (std::size_t l = 0; l < 32; ++l) {
      SetCode(block, m, g * 32 + l, (q[l] >> shift) & 0xF);
    }
  }
}

// The fit sums over a group in kLanes partial sums (element l into lane
// l % kLanes), added up in lane order at the end, so the AVX2 and scalar
// paths give the same result.
constexpr std::size_t kLanes = 8;

float Sum(const float* lanes) {
  float s = 0;
  for (std::size_t j = 0; j < kLanes; ++j) {
    s += lanes[j];
  }
  return s;
}

// The nearest code to (x - offset) * scale, clamped to [0, 15] first (ties
// round up).
float Code(float x, float offset, float scale) {
  const float v = std::min(std::max((x - offset) * scale, 0.0f), 15.0f);
  return static_cast<float>(static_cast<int>(v + 0.5f));
}

#if defined(__AVX2__)
float Sum(__m256 v) {
  alignas(32) float lanes[kLanes];
  _mm256_store_ps(lanes, v);
  return Sum(lanes);
}

__m256 Code(__m256 x, __m256 offset, __m256 scale) {
  __m256 v = _mm256_mul_ps(_mm256_sub_ps(x, offset), scale);
  v = _mm256_min_ps(_mm256_max_ps(v, _mm256_setzero_ps()),
                    _mm256_set1_ps(15.0f));
  return _mm256_cvtepi32_ps(
      _mm256_cvttps_epi32(_mm256_add_ps(v, _mm256_set1_ps(0.5f))));
}
#endif

// Error sum of squares of one group's weights `x` under (step, min) with
// their nearest codes, weighted by `w`.
float GroupError(const float* x, const float* w, float step, float min) {
  const float inverse = step > 0 ? 1.0f / step : 0.0f;
#if defined(__AVX2__)
  const __m256 vmin = _mm256_set1_ps(min);
  const __m256 vstep = _mm256_set1_ps(step);
  const __m256 vinverse = _mm256_set1_ps(inverse);
  __m256 sse = _mm256_setzero_ps();
  for (std::size_t l = 0; l < kGroup; l += kLanes) {
    const __m256 xv = _mm256_loadu_ps(x + l);
    const __m256 code = Code(xv, vmin, vinverse);
    const __m256 e =
        _mm256_sub_ps(_mm256_add_ps(vmin, _mm256_mul_ps(vstep, code)), xv);
    sse = _mm256_add_ps(
        sse, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(w + l), e), e));
  }
  return Sum(sse);
#else
  float sse[kLanes] = {};
  for (std::size_t l = 0; l < kGroup; l += kLanes) {
    for (std::size_t j = 0; j < kLanes; ++j) {
      const float code = Code(x[l + j], min, inverse);
      const float e = min + step * code - x[l + j];
      sse[j] += w[l + j] * e * e;
    }
  }
  return Sum(sse);
#endif
}

// For code = Code(x, lo, inverse) over one group, with t = x - lo (>= 0, so
// the clamp at 0 never binds): the sums of w * code, w * code^2 and
// w * code * x.
void CodeSums(const float* t, const float* x, const float* w, float inverse,
              float* sums) {
#if defined(__AVX2__)
  const __m256 vinverse = _mm256_set1_ps(inverse);
  __m256 l1 = _mm256_setzero_ps();
  __m256 l2 = _mm256_setzero_ps();
  __m256 xl = _mm256_setzero_ps();
  for (std::size_t l = 0; l < kGroup; l += kLanes) {
    const __m256 v = _mm256_min_ps(
        _mm256_mul_ps(_mm256_loadu_ps(t + l), vinverse), _mm256_set1_ps(15.0f));
    const __m256 code = _mm256_round_ps(_mm256_add_ps(v, _mm256_set1_ps(0.5f)),
                                        _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m256 wc = _mm256_mul_ps(_mm256_loadu_ps(w + l), code);
    l1 = _mm256_add_ps(l1, wc);
    l2 = _mm256_add_ps(l2, _mm256_mul_ps(wc, code));
    xl = _mm256_add_ps(xl, _mm256_mul_ps(wc, _mm256_loadu_ps(x + l)));
  }
  sums[0] = Sum(l1);
  sums[1] = Sum(l2);
  sums[2] = Sum(xl);
#else
  float l1[kLanes] = {};
  float l2[kLanes] = {};
  float xl[kLanes] = {};
  for (std::size_t l = 0; l < kGroup; l += kLanes) {
    for (std::size_t j = 0; j < kLanes; ++j) {
      const float v = std::min(t[l + j] * inverse, 15.0f);
      const float code = static_cast<float>(static_cast<int>(v + 0.5f));
      const float wc = w[l + j] * code;
      l1[j] += wc;
      l2[j] += wc * code;
      xl[j] += wc * x[l + j];
    }
  }
  sums[0] = Sum(l1);
  sums[1] = Sum(l2);
  sums[2] = Sum(xl);
#endif
}

// The AVX-512 path is chosen at run time on CPUs that have it. Building with
// GUFO_NPU_Q4NX_AVX2_ONLY leaves it out, to check the AVX2 path gives the
// same bytes on such a CPU.
#if defined(__AVX2__) && !defined(GUFO_NPU_Q4NX_AVX2_ONLY)
#define GUFO_NPU_Q4NX_AVX512 1
// AVX-512 versions that run two candidates at once, one per 256-bit half:
// each half's lane j adds the same elements in the same order as the AVX2
// code's lane j, so the results are bit-identical. The inputs come with each
// run of 8 values repeated, [v0..v7, v0..v7, v8..v15, v8..v15, ...].
#define GUFO_NPU_AVX512 __attribute__((target("avx512f,avx512dq")))

GUFO_NPU_AVX512 void Halves(__m512 v, float* first, float* second) {
  alignas(64) float lanes[2 * kLanes];
  _mm512_store_ps(lanes, v);
  *first = Sum(lanes);
  *second = Sum(lanes + kLanes);
}

GUFO_NPU_AVX512 __m512 Pair(float a, float b) {
  return _mm512_insertf32x8(_mm512_set1_ps(a), _mm256_set1_ps(b), 1);
}

// CodeSums for two inverses; sums[0..2] for `inverse_a`, sums[3..5] for
// `inverse_b`.
GUFO_NPU_AVX512 void CodeSumsPair(const float* t2, const float* x2,
                                  const float* w2, float inverse_a,
                                  float inverse_b, float* sums) {
  const __m512 vinverse = Pair(inverse_a, inverse_b);
  __m512 l1 = _mm512_setzero_ps();
  __m512 l2 = _mm512_setzero_ps();
  __m512 xl = _mm512_setzero_ps();
  for (std::size_t l = 0; l < 2 * kGroup; l += 2 * kLanes) {
    const __m512 v =
        _mm512_min_ps(_mm512_mul_ps(_mm512_loadu_ps(t2 + l), vinverse),
                      _mm512_set1_ps(15.0f));
    const __m512 code =
        _mm512_roundscale_ps(_mm512_add_ps(v, _mm512_set1_ps(0.5f)),
                             _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m512 wc = _mm512_mul_ps(_mm512_loadu_ps(w2 + l), code);
    l1 = _mm512_add_ps(l1, wc);
    l2 = _mm512_add_ps(l2, _mm512_mul_ps(wc, code));
    xl = _mm512_add_ps(xl, _mm512_mul_ps(wc, _mm512_loadu_ps(x2 + l)));
  }
  Halves(l1, &sums[0], &sums[3]);
  Halves(l2, &sums[1], &sums[4]);
  Halves(xl, &sums[2], &sums[5]);
}

// GroupError for two (step, min) pairs: errors[0] and errors[1].
GUFO_NPU_AVX512 void GroupErrorPair(const float* x2, const float* w2,
                                    float step_a, float min_a, float step_b,
                                    float min_b, float* errors) {
  const __m512 vmin = Pair(min_a, min_b);
  const __m512 vstep = Pair(step_a, step_b);
  const __m512 vinverse = Pair(step_a > 0 ? 1.0f / step_a : 0.0f,
                               step_b > 0 ? 1.0f / step_b : 0.0f);
  __m512 sse = _mm512_setzero_ps();
  for (std::size_t l = 0; l < 2 * kGroup; l += 2 * kLanes) {
    const __m512 xv = _mm512_loadu_ps(x2 + l);
    __m512 v = _mm512_mul_ps(_mm512_sub_ps(xv, vmin), vinverse);
    v = _mm512_min_ps(_mm512_max_ps(v, _mm512_setzero_ps()),
                      _mm512_set1_ps(15.0f));
    const __m512 code =
        _mm512_roundscale_ps(_mm512_add_ps(v, _mm512_set1_ps(0.5f)),
                             _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m512 e =
        _mm512_sub_ps(_mm512_add_ps(vmin, _mm512_mul_ps(vstep, code)), xv);
    sse = _mm512_add_ps(
        sse, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(w2 + l), e), e));
  }
  Halves(sse, &errors[0], &errors[1]);
}

// [v0..v7, v0..v7, v8..v15, ...]: each run of kLanes values twice.
void Repeat(const float* v, float* out) {
  for (std::size_t l = 0; l < kGroup; l += kLanes) {
    std::copy(v + l, v + l + kLanes, out + 2 * l);
    std::copy(v + l, v + l + kLanes, out + 2 * l + kLanes);
  }
}

// FitGroup's candidate scan (its 37 steps), with t = x - lo: the code sums two
// candidates at a time, then each candidate's step, minimum and error 16 at a
// time, every expression evaluated in the scalar code's order (IEEE division
// and multiplication give the same results in a vector), then the scalar
// code's in-order choice: the first strictly better candidate wins.
GUFO_NPU_AVX512 void FitCandidatesAvx512(const float* t, const float* x,
                                         const float* w, float range,
                                         float sum_w, float sum_x, float sum_x2,
                                         float* best, float* step_out,
                                         float* min_out) {
  constexpr int kCandidates = 37;
  constexpr int kPadded = 48;
  alignas(64) float inverse[kPadded];
  for (int b = 0; b < kPadded; b += 16) {
    const __m512 is = _mm512_cvtepi32_ps(_mm512_add_epi32(
        _mm512_set1_epi32(b), _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                10, 11, 12, 13, 14, 15)));
    const __m512 num =
        _mm512_add_ps(_mm512_add_ps(_mm512_set1_ps(-0.9f),
                                    _mm512_mul_ps(_mm512_set1_ps(0.05f), is)),
                      _mm512_set1_ps(15.0f));
    _mm512_store_ps(inverse + b, _mm512_div_ps(num, _mm512_set1_ps(range)));
  }
  alignas(64) float t2[2 * kGroup];
  alignas(64) float x2[2 * kGroup];
  alignas(64) float w2[2 * kGroup];
  Repeat(t, t2);
  Repeat(x, x2);
  Repeat(w, w2);
  alignas(64) float l1[kPadded] = {};
  alignas(64) float l2[kPadded] = {};
  alignas(64) float xl[kPadded] = {};
  for (int is = 0; is < kCandidates; is += 2) {
    float sums[6];
    CodeSumsPair(t2, x2, w2, inverse[is], inverse[is + 1], sums);
    l1[is] = sums[0];
    l2[is] = sums[1];
    xl[is] = sums[2];
    l1[is + 1] = sums[3];
    l2[is + 1] = sums[4];
    xl[is + 1] = sums[5];
  }
  alignas(64) float det[kPadded];
  alignas(64) float step[kPadded];
  alignas(64) float min[kPadded];
  alignas(64) float sse[kPadded];
  const __m512 vw = _mm512_set1_ps(sum_w);
  const __m512 vx = _mm512_set1_ps(sum_x);
  const __m512 vx2 = _mm512_set1_ps(sum_x2);
  for (int b = 0; b < kPadded; b += 16) {
    const __m512 sl = _mm512_load_ps(l1 + b);
    const __m512 sl2 = _mm512_load_ps(l2 + b);
    const __m512 sxl = _mm512_load_ps(xl + b);
    // det = sum_w * sum_l2 - sum_l * sum_l
    const __m512 d =
        _mm512_sub_ps(_mm512_mul_ps(vw, sl2), _mm512_mul_ps(sl, sl));
    // step = (sum_w * sum_xl - sum_x * sum_l) / det
    const __m512 s = _mm512_div_ps(
        _mm512_sub_ps(_mm512_mul_ps(vw, sxl), _mm512_mul_ps(vx, sl)), d);
    // min = (sum_l2 * sum_x - sum_l * sum_xl) / det
    const __m512 m = _mm512_div_ps(
        _mm512_sub_ps(_mm512_mul_ps(sl2, vx), _mm512_mul_ps(sl, sxl)), d);
    // sse = sum_x2 + step * step * sum_l2 + min * min * sum_w
    //       + 2 * (step * min * sum_l - step * sum_xl - min * sum_x)
    const __m512 a = _mm512_add_ps(
        _mm512_add_ps(vx2, _mm512_mul_ps(_mm512_mul_ps(s, s), sl2)),
        _mm512_mul_ps(_mm512_mul_ps(m, m), vw));
    const __m512 c =
        _mm512_sub_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_mul_ps(s, m), sl),
                                    _mm512_mul_ps(s, sxl)),
                      _mm512_mul_ps(m, vx));
    _mm512_store_ps(det + b, d);
    _mm512_store_ps(step + b, s);
    _mm512_store_ps(min + b, m);
    _mm512_store_ps(sse + b,
                    _mm512_add_ps(a, _mm512_mul_ps(_mm512_set1_ps(2.0f), c)));
  }
  for (int is = 0; is < kCandidates; ++is) {
    if (det[is] <= 0 || step[is] <= 0) {
      continue;
    }
    if (sse[is] < *best) {
      *best = sse[is];
      *step_out = step[is];
      *min_out = min[is];
    }
  }
}

bool HasAvx512() {
  static const bool has =
      __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
  return has;
}
#endif

// One group's step and minimum, before they round onto the block's grids:
// the weighted least-squares fit over the code assignments of 37 candidate
// steps around the min/max one (llama.cpp's make_qkx3_quants, without its
// minimum <= 0 clamp, since q4nx's sub-minimum is signed).
void FitGroup(const float* x, const float* w, float* step_out, float* min_out) {
  const auto [lo_it, hi_it] = std::minmax_element(x, x + kGroup);
  const float lo = *lo_it;
  const float hi = *hi_it;
  *step_out = 0;
  *min_out = lo;
  if (hi <= lo) {
    return;
  }
  float lanes_w[kLanes] = {};
  float lanes_x[kLanes] = {};
  float lanes_x2[kLanes] = {};
  for (std::size_t l = 0; l < kGroup; l += kLanes) {
    for (std::size_t j = 0; j < kLanes; ++j) {
      lanes_w[j] += w[l + j];
      lanes_x[j] += w[l + j] * x[l + j];
      lanes_x2[j] += w[l + j] * x[l + j] * x[l + j];
    }
  }
  const float sum_w = Sum(lanes_w);
  const float sum_x = Sum(lanes_x);
  const float sum_x2 = Sum(lanes_x2);
  float best = GroupError(x, w, (hi - lo) / 15.0f, lo);
  *step_out = (hi - lo) / 15.0f;
  // Candidate `is`'s step from its codes' sums; candidates are considered in
  // order and replace the best only when strictly better.
  const auto consider = [&](const float* sums) {
    const float sum_l = sums[0];
    const float sum_l2 = sums[1];
    const float sum_xl = sums[2];
    const float det = sum_w * sum_l2 - sum_l * sum_l;
    if (det <= 0) {
      return;
    }
    const float step = (sum_w * sum_xl - sum_x * sum_l) / det;
    const float min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
    if (step <= 0) {
      return;
    }
    // sum w (min + step * code - x)^2, expanded over the sums above.
    const float sse = sum_x2 + step * step * sum_l2 + min * min * sum_w +
                      2 * (step * min * sum_l - step * sum_xl - min * sum_x);
    if (sse < best) {
      best = sse;
      *step_out = step;
      *min_out = min;
    }
  };
  constexpr int kSteps = 36;
  const auto inverse = [&](int is) {
    return (-0.9f + 0.05f * static_cast<float>(is) + 15.0f) / (hi - lo);
  };
  float t[kGroup];
  for (std::size_t l = 0; l < kGroup; ++l) {
    t[l] = x[l] - lo;
  }
#if defined(GUFO_NPU_Q4NX_AVX512)
  if (HasAvx512()) {
    FitCandidatesAvx512(t, x, w, hi - lo, sum_w, sum_x, sum_x2, &best, step_out,
                        min_out);
    return;
  }
#endif
  float sums[3];
  for (int is = 0; is <= kSteps; ++is) {
    CodeSums(t, x, w, inverse(is), sums);
    consider(sums);
  }
}

// Fit::kMinMax: row m of a block from its kKTile fp32 weights. Per group a
// min/max fit, with the minimum rounded down onto dmin's grid (int8) and the
// step rounded up onto d's (uint8), so every weight is within the group's 16
// levels.
void PackRowBlockMinMax(const float* w, std::uint8_t* block, std::size_t m) {
  float lo[kGroups];
  float hi[kGroups];
  float max_abs_min = 0;
  for (std::size_t g = 0; g < kGroups; ++g) {
    const auto [lo_it, hi_it] =
        std::minmax_element(w + g * kGroup, w + (g + 1) * kGroup);
    lo[g] = *lo_it;
    hi[g] = *hi_it;
    max_abs_min = std::max(max_abs_min, std::abs(lo[g]));
  }
  const std::uint16_t dmin_bits = Bf16AtLeast(max_abs_min / 127.0f);
  const float dmin = quant::Bf16ToFloat(dmin_bits);
  std::int8_t mn[kGroups];
  float need[kGroups];  // the step each group needs from its minimum
  float max_need = 0;
  for (std::size_t g = 0; g < kGroups; ++g) {
    const float q = dmin > 0 ? std::floor(lo[g] / dmin) : 0.0f;
    mn[g] = static_cast<std::int8_t>(std::clamp(q, -127.0f, 127.0f));
    need[g] = (hi[g] - dmin * static_cast<float>(mn[g])) / 15.0f;
    max_need = std::max(max_need, need[g]);
  }
  const std::uint16_t d_bits = Bf16AtLeast(max_need / 255.0f);
  const float d = quant::Bf16ToFloat(d_bits);
  SetSuper(block, m, d_bits, dmin_bits);
  for (std::size_t g = 0; g < kGroups; ++g) {
    const float sc = d > 0 ? std::ceil(need[g] / d) : 0.0f;
    SetSub(block, m, g, static_cast<std::uint8_t>(std::clamp(sc, 0.0f, 255.0f)),
           mn[g]);
    float step;
    float min;
    Params(block, m, g, &step, &min);
    for (std::size_t l = 0; l < kGroup; ++l) {
      float code =
          step > 0 ? std::nearbyint((w[g * kGroup + l] - min) / step) : 0.0f;
      code = std::clamp(code, 0.0f, 15.0f);
      SetCode(block, m, g * kGroup + l, static_cast<std::uint8_t>(code));
    }
  }
}

// Fit::kSearch: row m of a block from its kKTile fp32 weights. Each group's
// step and minimum come from FitGroup, weighting each weight by av + |w|
// (llama.cpp's Q4_K weighting: larger weights count more); d and dmin are
// set from the largest of them, and each group then takes the sub-scale and
// sub-minimum within one grid step of its fit, and the nearest codes, with
// the lowest weighted error.
void PackRowBlockSearch(const float* w, std::uint8_t* block, std::size_t m) {
  float sum_x2 = 0;
  for (std::size_t k = 0; k < kKTile; ++k) {
    sum_x2 += w[k] * w[k];
  }
  const float av = std::sqrt(2 * sum_x2 / static_cast<float>(kKTile));
  float weight[kKTile];
  for (std::size_t k = 0; k < kKTile; ++k) {
    weight[k] = av + std::abs(w[k]);
  }
  float step[kGroups];
  float min[kGroups];
  float max_step = 0;
  float max_abs_min = 0;
  for (std::size_t g = 0; g < kGroups; ++g) {
    FitGroup(w + g * kGroup, weight + g * kGroup, &step[g], &min[g]);
    max_step = std::max(max_step, step[g]);
    max_abs_min = std::max(max_abs_min, std::abs(min[g]));
  }
  const std::uint16_t d_bits = Bf16AtLeast(max_step / 255.0f);
  const std::uint16_t dmin_bits = Bf16AtLeast(max_abs_min / 127.0f);
  const float d = quant::Bf16ToFloat(d_bits);
  const float dmin = quant::Bf16ToFloat(dmin_bits);
  SetSuper(block, m, d_bits, dmin_bits);
  for (std::size_t g = 0; g < kGroups; ++g) {
    const float* x = w + g * kGroup;
    const float* wg = weight + g * kGroup;
    const float sc0 = d > 0 ? std::nearbyint(step[g] / d) : 0.0f;
    const float mn0 = dmin > 0 ? std::nearbyint(min[g] / dmin) : 0.0f;
    // The 9 sub-scale and sub-minimum choices, in order; the first with the
    // lowest error wins.
    constexpr int kChoices = 9;
    float sc[kChoices];
    float mn[kChoices];
    float sse[kChoices];
    for (int i = 0; i < kChoices; ++i) {
      sc[i] = std::clamp(sc0 + static_cast<float>(i / 3 - 1), 0.0f, 255.0f);
      mn[i] = std::clamp(mn0 + static_cast<float>(i % 3 - 1), -127.0f, 127.0f);
    }
#if defined(GUFO_NPU_Q4NX_AVX512)
    if (HasAvx512()) {
      alignas(64) float x2[2 * kGroup];
      alignas(64) float w2[2 * kGroup];
      Repeat(x, x2);
      Repeat(wg, w2);
      for (int i = 0; i < kChoices; i += 2) {
        const int j = std::min(i + 1, kChoices - 1);
        float pair[2];
        GroupErrorPair(x2, w2, d * sc[i], dmin * mn[i], d * sc[j], dmin * mn[j],
                       pair);
        sse[i] = pair[0];
        sse[j] = pair[1];
      }
    } else
#endif
    {
      for (int i = 0; i < kChoices; ++i) {
        sse[i] = GroupError(x, wg, d * sc[i], dmin * mn[i]);
      }
    }
    float best = INFINITY;
    float best_sc = 0;
    float best_mn = 0;
    for (int i = 0; i < kChoices; ++i) {
      if (sse[i] < best) {
        best = sse[i];
        best_sc = sc[i];
        best_mn = mn[i];
      }
    }
    SetSub(block, m, g, static_cast<std::uint8_t>(best_sc),
           static_cast<std::int8_t>(best_mn));
    float step_g;
    float min_g;
    Params(block, m, g, &step_g, &min_g);
    const float inverse = step_g > 0 ? 1.0f / step_g : 0.0f;
    for (std::size_t l = 0; l < kGroup; ++l) {
      SetCode(block, m, g * kGroup + l,
              static_cast<std::uint8_t>(Code(x[l], min_g, inverse)));
    }
  }
}

// Packs tile `tile` (kMTile rows) of `out`. row_fn(r, buffer) returns row
// r's Q4_K data when it can be copied code for code, or fills `buffer` (at
// least out.dim values) with its fp32 weights and returns nullptr. The rows of
// a tile share code bytes, so one thread packs a whole tile.
template<typename RowFn>
void PackTile(Matrix& out, std::size_t tile, Fit fit, const RowFn& row_fn,
              float* row) {
  const auto pack_row_block =
      fit == Fit::kSearch ? PackRowBlockSearch : PackRowBlockMinMax;
  for (std::size_t m = 0; m < kMTile; ++m) {
    const std::size_t r = tile * kMTile + m;
    if (r >= out.rows) {
      break;
    }
    const std::uint8_t* q4k = row_fn(r, row);
    for (std::size_t kb = 0; kb < out.dim / kKTile; ++kb) {
      std::uint8_t* block = Block(out, tile, kb);
      if (q4k != nullptr) {
        PackQ4KSuper(q4k + kb * kQ4KBytes, block, m);
        continue;
      }
      pack_row_block(row + kb * kKTile, block, m);
    }
  }
}

std::size_t Tiles(const Matrix& m) {
  return (m.rows + kMTile - 1) / kMTile;
}

// Every tile of `out`, in parallel.
template<typename RowFn>
void PackTiles(Matrix& out, Fit fit, RowFn&& row_fn) {
  ParallelFor(Tiles(out), [&](std::size_t begin, std::size_t end) {
    std::vector<float> row(out.dim);
    for (std::size_t tile = begin; tile < end; ++tile) {
      PackTile(out, tile, fit, row_fn, row.data());
    }
  });
}

// A PackJob's rows: stacked segments, picked through its order.
struct JobRows {
  const PackJob* job;
  std::vector<std::size_t> starts;  // first row of each segment

  const std::uint8_t* operator()(std::size_t packed, float* row) const {
    const std::size_t r = job->order.empty() ? packed : job->order[packed];
    const auto it = std::upper_bound(starts.begin(), starts.end(), r) - 1;
    const auto& seg =
        job->segments[static_cast<std::size_t>(it - starts.begin())];
    const std::uint8_t* src =
        seg.data + (r - *it) * quant::RowBytes(seg.type, job->dim);
    if (seg.type == gguf::GgmlType::kQ4_K) {
      return src;
    }
    quant::Dequantize(seg.type, src, row, job->dim);
    return nullptr;
  }
};

}  // namespace

std::size_t PaddedRows(std::size_t rows) {
  return (rows + kOutPerRound - 1) / kOutPerRound * kOutPerRound;
}

std::vector<Matrix> PackEach(const std::vector<PackJob>& jobs, Fit fit) {
  std::vector<Matrix> out;
  std::vector<JobRows> rows;
  out.reserve(jobs.size());
  rows.reserve(jobs.size());
  std::size_t max_dim = 0;
  // (job, tile) work items, so many small matrices share one parallel region.
  std::vector<std::pair<std::size_t, std::size_t>> items;
  for (const auto& job : jobs) {
    std::size_t total = 0;
    bool exact = true;
    JobRows job_rows{&job, {}};
    for (const auto& seg : job.segments) {
      job_rows.starts.push_back(total);
      total += seg.rows;
      exact = exact && seg.type == gguf::GgmlType::kQ4_K;
    }
    if (!job.order.empty() && job.order.size() != total) {
      throw std::invalid_argument("q4nx: row order does not cover every row");
    }
    out.push_back(Allocate(total, job.dim));
    out.back().exact = exact;
    rows.push_back(std::move(job_rows));
    max_dim = std::max(max_dim, job.dim);
    for (std::size_t tile = 0; tile < Tiles(out.back()); ++tile) {
      items.emplace_back(out.size() - 1, tile);
    }
  }
  ParallelFor(items.size(), [&](std::size_t begin, std::size_t end) {
    std::vector<float> row(max_dim);
    for (std::size_t i = begin; i < end; ++i) {
      const auto [j, tile] = items[i];
      PackTile(out[j], tile, fit, rows[j], row.data());
    }
  });
  return out;
}

Matrix PackRows(const std::vector<Segment>& segments, std::size_t dim, Fit fit,
                const std::vector<std::size_t>& order) {
  return std::move(PackEach({{segments, dim, order}}, fit).front());
}

std::vector<std::size_t> SwiGluOrder(std::size_t ffn) {
  constexpr std::size_t kHalf = kMTile / 2;
  if (ffn % kHalf != 0) {
    throw std::invalid_argument("q4nx: SwiGLU width must be a multiple of 16");
  }
  std::vector<std::size_t> order(2 * ffn);
  for (std::size_t n = 0; n < order.size(); ++n) {
    const std::size_t tile = n / kMTile;
    const std::size_t m = n % kMTile;
    order[n] = (m < kHalf ? 0 : ffn) + tile * kHalf + m % kHalf;
  }
  return order;
}

Matrix Pack(gguf::GgmlType type, const std::uint8_t* data, std::size_t rows,
            std::size_t dim, Fit fit) {
  return PackRows({{type, data, rows}}, dim, fit);
}

Matrix PackFloat(const float* weights, std::size_t rows, std::size_t dim,
                 Fit fit) {
  Matrix out = Allocate(rows, dim);
  PackTiles(out, fit, [&](std::size_t r, float* row) -> const std::uint8_t* {
    std::memcpy(row, weights + r * dim, dim * sizeof(float));
    return nullptr;
  });
  return out;
}

std::vector<float> Unpack(const Matrix& matrix) {
  std::vector<float> w(matrix.padded_rows * matrix.dim);
  const std::size_t tiles = matrix.padded_rows / kMTile;
  ParallelFor(tiles, [&](std::size_t begin, std::size_t end) {
    for (std::size_t tile = begin; tile < end; ++tile) {
      for (std::size_t kb = 0; kb < matrix.dim / kKTile; ++kb) {
        const std::uint8_t* block = Block(matrix, tile, kb);
        for (std::size_t m = 0; m < kMTile; ++m) {
          float* row =
              w.data() + (tile * kMTile + m) * matrix.dim + kb * kKTile;
          for (std::size_t k = 0; k < kKTile; ++k) {
            float step;
            float min;
            Params(block, m, k / kGroup, &step, &min);
            const std::size_t i = CodeIndex(m, k);
            const std::uint8_t byte = block[kCodes + i / 2];
            const int code = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            row[k] = min + step * static_cast<float>(code);
          }
        }
      }
    }
  });
  return w;
}

void Gemv(const Matrix& matrix, const float* x, float* y) {
  const std::size_t tiles = (matrix.rows + kMTile - 1) / kMTile;
  const std::size_t k_blocks = matrix.dim / kKTile;
  // Per-32 sums of x fold each group's minimum into one multiply.
  std::vector<double> sums(matrix.dim / kGroup);
  for (std::size_t g = 0; g < sums.size(); ++g) {
    for (std::size_t l = 0; l < kGroup; ++l) {
      sums[g] += x[g * kGroup + l];
    }
  }
  ParallelFor(tiles, [&](std::size_t begin, std::size_t end) {
    for (std::size_t tile = begin; tile < end; ++tile) {
      double acc[kMTile] = {};
      for (std::size_t kb = 0; kb < k_blocks; ++kb) {
        const std::uint8_t* block = Block(matrix, tile, kb);
        const std::uint8_t* codes = block + kCodes;
        const float* xk = x + kb * kKTile;
        for (std::size_t g = 0; g < kGroups; ++g) {
          // Codes of group g: [m / 16][g][l][m % 16], two per byte.
          for (std::size_t half = 0; half < kMTile / 16; ++half) {
            double dot[16] = {};
            const std::uint8_t* q = codes + ((half * kGroups + g) * 32) * 8;
            for (std::size_t l = 0; l < 32; ++l) {
              const double xv = xk[g * 32 + l];
              for (std::size_t i = 0; i < 8; ++i) {
                const std::uint8_t byte = q[l * 8 + i];
                dot[2 * i] += (byte & 0xF) * xv;
                dot[2 * i + 1] += (byte >> 4) * xv;
              }
            }
            for (std::size_t i = 0; i < 16; ++i) {
              const std::size_t m = half * 16 + i;
              float step;
              float min;
              Params(block, m, g, &step, &min);
              acc[m] += step * dot[i] + min * sums[kb * kGroups + g];
            }
          }
        }
      }
      for (std::size_t m = 0; m < kMTile; ++m) {
        const std::size_t r = tile * kMTile + m;
        if (r < matrix.rows) {
          y[r] = static_cast<float>(acc[m]);
        }
      }
    }
  });
}

std::vector<double> Reference(const std::vector<float>& unpacked,
                              const Matrix& matrix, const float* x) {
  std::vector<double> y(matrix.padded_rows);
  ParallelFor(matrix.padded_rows, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = begin; r < end; ++r) {
      const float* row = unpacked.data() + r * matrix.dim;
      double acc = 0;
      for (std::size_t k = 0; k < matrix.dim; ++k) {
        acc += static_cast<double>(row[k]) * x[k];
      }
      y[r] = acc;
    }
  });
  return y;
}

}  // namespace gufo_npu::q4nx
