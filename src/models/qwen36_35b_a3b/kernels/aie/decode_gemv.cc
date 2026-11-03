// q4nx matrix-vector product for decode: y = W x for one or two tokens per
// weight read (two verify a speculative draft).
//
// Each core owns Q4NX_M_TILE out-features per round and streams their q4nx
// blocks (Q4NX_M_TILE x Q4NX_K_TILE) past the tokens. The k block count, the
// token count and the input and output stages arrive at run time, so one core
// program serves every call:
//   input:  plain (fp32), act (bf16, a compact SwiGLU output), or resnorm:
//           rms_norm(H + D) * w, with H + D also written out as the next
//           residual by one column's cores.
//   output: store (fp32), or swiglu: silu(gate) * up from paired rows, bf16.
//
// The token buffer (raw) holds GEMV_TOKENS slots of SLOT fp32 each, then the
// norm weight; act tokens sit back to back as bf16 instead (MAX_DIM apart).
// Each output object carries GEMV_TOKENS * M_TILE values, token-major.
//
// The block (q4nx_block below, src/core/q4nx.hpp) holds 4-bit codes, an 8-bit
// sub-scale and sub-minimum per row and 32 inputs, and a bf16 super-scale and
// super-minimum per row: w = dmin * mn + d * sc * code. The accumulation is
// adapted from IRON's q4nx_accumulate (flm_gemma4/q4nx.h), keeping more
// precision: each block's dot product and each per-32 token sum are carried
// as two bf16 halves (hi + lo) instead of one, and the output stays fp32.
// Geometry arrives as -D flags.
#include "aie_kernels/aie_kernel_utils.h"
#include "aie_kernels/common/exp2_poly.h"
#include "aie_kernels/common/zero.h"
#include "aie_kernels/flm_gemma4/utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

#if !defined(Q4NX_M_TILE) || !defined(Q4NX_K_TILE) || !defined(Q4NX_GROUP) || \
    !defined(GEMV_MAX_DIM) || !defined(GEMV_RMS_EPS) ||                      \
    !defined(GEMV_HIDDEN) || !defined(GEMV_TOKENS) || !defined(GEMV_SLOT)
#error "q4nx_gemv.cc needs its geometry -D defined"
#endif

// One M by N block. Per row: d and dmin (bf16). Per row and 32 inputs,
// indexed [n / 32][m]: the sub-scale sc (uint8) and sub-minimum mn (int8).
// Then the codes in strips of 16 rows, each strip column-major.
template <int M, int N>
struct q4nx_block {
  bf16 d[M];
  bf16 dmin[M];
  uint8_t sc[M * N / 32];
  int8_t mn[M * N / 32];
  uint4 qs[M * N];
};

using q4nx_block_t = q4nx_block<Q4NX_M_TILE, Q4NX_K_TILE>;
// The codes take half a byte each in memory (the block is 4736 bytes), though
// a uint4 array's sizeof counts a byte each; only the offsets matter.
static_assert(__builtin_offsetof(q4nx_block_t, qs) ==
                  Q4NX_M_TILE * 4 + Q4NX_M_TILE * Q4NX_K_TILE / 32 * 2,
              "d, dmin, sc and mn precede the codes");

namespace {

constexpr int M_TILE = Q4NX_M_TILE;
constexpr int K_TILE = Q4NX_K_TILE;
constexpr int MAX_DIM = GEMV_MAX_DIM;
constexpr int TOKENS = GEMV_TOKENS;  // tokens a call can carry
constexpr int SLOT = GEMV_SLOT;      // fp32 per plain/resnorm token slot
constexpr int SUMS = 2 * MAX_DIM / 32;  // hi then lo, per token

static_assert(Q4NX_GROUP == 32, "a column sum covers exactly one group");
static_assert(K_TILE % 32 == 0, "a k tile must hold whole groups");
static_assert(MAX_DIM % K_TILE == 0, "the token must cover whole k tiles");
static_assert(M_TILE % 16 == 0, "rows are processed 16 at a time");
static_assert(TOKENS == 1 || TOKENS == 2, "one or two tokens per call");

// c_t[0:M] += dequantize(A) * B_t[0:N] for T tokens, with the minima folded
// in through the per-32 sums of each B_t (hi and lo bf16 halves). The codes
// are widened once and used for every token. Per 16 rows the integer
// sub-scales and sub-minima weight each group's dot product and token sum
// (exact in bf16), and the row's d and dmin multiply the two totals once.
template <int M, int N, int T>
void accumulate(const q4nx_block<M, N> *A, const bf16 *const *B,
                float *const *c, const bf16 *const *sums_hi,
                const bf16 *const *sums_lo) {
  constexpr int pr = 16;
  const uint4 *qs_ptr = A->qs;

  AIE_LOOP_RANGE(M / pr, M / pr)
  for (int row = 0; row < M; row += pr) {
    aie::accum<accfloat, pr> c_accum[T];
    aie::accum<accfloat, pr> scaled[T];  // sum over groups of sc * dot
    aie::accum<accfloat, pr> minned[T];  // sum over groups of mn * token sum
#pragma clang loop unroll(full)
    for (int t = 0; t < T; t++) {
      c_accum[t].from_vector(aie::load_v<pr>(c[t] + row));
      scaled[t] = aie::zeros<accfloat, pr>();
      minned[t] = aie::zeros<accfloat, pr>();
    }
    uint32_t scale_min_offset = row;

    AIE_LOOP_RANGE(N / 32, N / 32)
    for (int sub_chunk = 0; sub_chunk < N / 32; sub_chunk++) {
      aie::accum<accfloat, pr> sc_acc;
      sc_acc.from_vector(
          aie::to_float(aie::load_v<pr>(A->sc + scale_min_offset), 0));
      const aie::vector<bf16, pr> a_sc = sc_acc.template to_vector<bf16>();
      aie::accum<accfloat, pr> mn_acc;
      mn_acc.from_vector(
          aie::to_float(aie::load_v<pr>(A->mn + scale_min_offset), 0));
      const aie::vector<bf16, pr> a_mn = mn_acc.template to_vector<bf16>();
      scale_min_offset += M;

      // 32 columns = 4 uint4 sub-vectors x 8 lanes each, fully unrolled so
      // the scheduler pipelines one flat 32-long MAC chain per token.
      aie::vector<bf16, pr * 8> a_cc_bf16[4];
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
        aie::vector<uint4, pr * 8> a_cc = aie::load_v<pr * 8>(qs_ptr);
        qs_ptr += pr * 4;
        aie::accum<accfloat, pr * 8> acc;
        acc.from_vector(aie::to_float(a_cc, 0));
        a_cc_bf16[sv] = acc.template to_vector<bf16>();
      }
#pragma clang loop unroll(full)
      for (int t = 0; t < T; t++) {
        aie::vector<bf16, 32> b_col = aie::load_v<32>(B[t] + sub_chunk * 32);
        aie::accum<accfloat, pr> dot =
            aie::mul(a_cc_bf16[0].extract<pr>(0), b_col.get(0));
#pragma clang loop unroll(full)
        for (int sv = 0; sv < 4; sv++) {
#pragma clang loop unroll(full)
          for (int cc = 0; cc < 8; cc++) {
            if (sv == 0 && cc == 0)
              continue;
            dot = aie::mac(dot, a_cc_bf16[sv].extract<pr>(cc),
                           b_col.get(sv * 8 + cc));
          }
        }
        // The sub-scale multiply takes bf16 operands, so split the fp32 dot
        // into hi + lo: the residual error drops from 2^-9 to about 2^-17.
        aie::vector<bf16, pr> dot_hi = dot.template to_vector<bf16>();
        aie::accum<accfloat, pr> rest = aie::sub(dot, dot_hi);
        aie::vector<bf16, pr> dot_lo = rest.template to_vector<bf16>();
        scaled[t] = aie::mac(scaled[t], dot_hi, a_sc);
        scaled[t] = aie::mac(scaled[t], dot_lo, a_sc);
        minned[t] = aie::mac(minned[t], a_mn, sums_hi[t][sub_chunk]);
        minned[t] = aie::mac(minned[t], a_mn, sums_lo[t][sub_chunk]);
      }
    }
    // The row's super-scale and super-minimum, again on hi + lo halves.
    const aie::vector<bf16, pr> a_d = aie::load_v<pr>(A->d + row);
    const aie::vector<bf16, pr> a_dmin = aie::load_v<pr>(A->dmin + row);
#pragma clang loop unroll(full)
    for (int t = 0; t < T; t++) {
      aie::vector<bf16, pr> hi = scaled[t].template to_vector<bf16>();
      aie::vector<bf16, pr> lo =
          aie::sub(scaled[t], hi).template to_vector<bf16>();
      c_accum[t] = aie::mac(c_accum[t], hi, a_d);
      c_accum[t] = aie::mac(c_accum[t], lo, a_d);
      hi = minned[t].template to_vector<bf16>();
      lo = aie::sub(minned[t], hi).template to_vector<bf16>();
      c_accum[t] = aie::mac(c_accum[t], hi, a_dmin);
      c_accum[t] = aie::mac(c_accum[t], lo, a_dmin);
      aie::store_v(c[t] + row, c_accum[t].template to_vector<float>());
    }
  }
}

// Input stage modes (RTP word 2).
constexpr int PROLOGUE_PLAIN = 0;    // slot: dim fp32
constexpr int PROLOGUE_RESNORM = 1;  // slot: [H | D] fp32; w bf16 after slots
constexpr int PROLOGUE_ACT = 2;      // dim bf16 per token, MAX_DIM apart
// Output stage modes (RTP word 3).
constexpr int EPILOGUE_STORE = 0;   // per token: M_TILE fp32
constexpr int EPILOGUE_SWIGLU = 1;  // per token: 16 bf16 silu(g) * u, then pad
constexpr int EPILOGUE_SWIGLU_SCALED = 2;  // the same times a per-round scale
// RTP words (kernels/designs/q4nx_gemv.py): the output stage, and the first of
// the swiglu_scaled scales (round-major, then token).
constexpr int RTP_EPILOGUE = 3;
constexpr int RTP_SCALES = 7;

constexpr float RMS_EPS = GEMV_RMS_EPS;
constexpr int HIDDEN = GEMV_HIDDEN;
static_assert(HIDDEN % (4 * M_TILE) == 0, "four cores hand out the residual");

// sigmoid(g) in fp32: 1 / (1 + 2^(-g log2 e)), with MLIR-AIE's exp2
// polynomial and three Newton steps for the reciprocal. (The vtanh
// instruction passes |x| <= 0.5 through unchanged, ~1% error on SwiGLU.)
aie::vector<float, 16> sigmoid(aie::vector<float, 16> g) {
  aie::vector<float, 16> x =
      aie::mul(g, -1.4426950409f).template to_vector<float>();
  x = aie::max(aie::min(x, 80.0f), -80.0f);
  aie::vector<int32_t, 16> k;
  const aie::vector<float, 16> p = exp2_poly<16>(x, k);
  // 2^k as a float: (k + 127) in the exponent field.
  const aie::vector<float, 16> scale =
      aie::upshift(aie::add(k, 127), 23).template cast_to<float>();
  const aie::vector<float, 16> d =
      aie::add(aie::mul(p, scale).template to_vector<float>(), 1.0f);
  // 1 / d from the bit-trick estimate.
  aie::vector<float, 16> rcp =
      aie::sub(aie::broadcast<int32_t, 16>(0x7EF311C7),
               d.template cast_to<int32_t>())
          .template cast_to<float>();
  for (int it = 0; it < 3; it++) {
    const aie::vector<float, 16> dr = aie::mul(d, rcp).template to_vector<float>();
    rcp = aie::mul(rcp, aie::sub(aie::broadcast<float, 16>(2.0f), dr))
              .template to_vector<float>();
  }
  return rcp;
}

// The fp32 values at `in` to bf16 at `out`, which may be the same buffer:
// each 16-value chunk is read before its half-size write lands.
void to_bf16(bfloat16 *out, const float *in, int n) {
  for (int i = 0; i < n; i += 16) {
    aie::accum<accfloat, 16> acc;
    acc.from_vector(aie::load_v<16>(in + i));
    aie::store_v(out + i, acc.template to_vector<bfloat16>());
  }
}

// x = rms(h) * w as bf16 over the start of h's buffer (h is fp32).
void norm(float *h, const bfloat16 *w, int n) {
  aie::accum<accfloat, 16> squares = aie::zeros<accfloat, 16>();
  for (int i = 0; i < n; i += 16) {
    squares = aie::mac_square(squares, aie::load_v<16>(h + i));
  }
  const float mean =
      aie::reduce_add(squares.template to_vector<float>()) / (float)n;
  // 1/sqrt in float: AIE2P has no scalar sqrt, so Newton from the bit trick,
  // three steps (to fp32 precision).
  const float s = mean + RMS_EPS;
  float r = __builtin_bit_cast(
      float, 0x5f3759df - (__builtin_bit_cast(uint32_t, s) >> 1));
  for (int it = 0; it < 3; it++) {
    r = r * (1.5f - 0.5f * s * r * r);
  }
  bfloat16 *x = (bfloat16 *)h;
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(h + i);
    aie::accum<accfloat, 16> wv;
    wv.from_vector(aie::load_v<16>(w + i));
    aie::vector<float, 16> scaled =
        aie::mul(aie::mul(v, r).template to_vector<float>(),
                 wv.template to_vector<float>())
            .template to_vector<float>();
    aie::accum<accfloat, 16> out;
    out.from_vector(scaled);
    aie::store_v(x + i, out.template to_vector<bfloat16>());
  }
}

// Where token t's bf16 x sits in the token buffer after the input stage.
bfloat16 *token_x(float *raw, int prologue, int t) {
  return prologue == PROLOGUE_ACT ? (bfloat16 *)raw + t * MAX_DIM
                                  : (bfloat16 *)(raw + t * SLOT);
}

// The rest of the input stage for token t, then the per-32 sums of its x
// (hi and lo bf16 halves) that fold every block's minima into one MAC.
void prep(float *raw, bfloat16 *sums, int k_blocks, int prologue, int t) {
  const int n = k_blocks * K_TILE;
  bfloat16 *x = token_x(raw, prologue, t);
  if (prologue == PROLOGUE_RESNORM) {
    norm(raw + t * SLOT, (const bfloat16 *)(raw + TOKENS * SLOT), n);
  } else if (prologue == PROLOGUE_PLAIN) {
    to_bf16(x, raw + t * SLOT, n);
  }
  bfloat16 *hi_out = sums + t * SUMS;
  bfloat16 *lo_out = hi_out + SUMS / 2;
  for (int g = 0; g < n / 32; g++) {
    const float sum = aie::reduce_add(
        aie::accum<accfloat, 32>(aie::load_v<32>(x + 32 * g))
            .template to_vector<float>());
    const bfloat16 hi = (bfloat16)sum;
    hi_out[g] = hi;
    lo_out[g] = (bfloat16)(sum - (float)hi);
  }
}

}  // namespace

extern "C" {

// Once per call, first: with PROLOGUE_RESNORM, h = H + D in place over H for
// each of `tokens`. Also selects round-to-nearest-even for every
// float-to-bf16 narrowing that follows: a core powers up in floor mode, which
// truncates each block's dot products and biases the sum over the k blocks
// (about 1% relative error at dim 2048); the mode is core-global and sticky.
void gemv_residual(float *raw, int k_blocks, int prologue, int tokens) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  if (prologue == PROLOGUE_RESNORM) {
    const int n = k_blocks * K_TILE;
    for (int t = 0; t < tokens; t++) {
      float *h = raw + t * SLOT;
      for (int i = 0; i < n; i += 16) {
        aie::accum<accfloat, 16> acc;
        acc.from_vector(aie::load_v<16>(h + i));
        acc = aie::add(acc, aie::load_v<16>(h + n + i));
        aie::store_v(h + i, acc.template to_vector<float>());
      }
    }
  }
  event1();
}

// On the residual column, between gemv_residual and gemv_prep: M_TILE values
// of each token's h, piece `piece` of this core's quarter (`row` of 4).
void gemv_residual_out(float *y, const float *raw, int row, int piece) {
  event0();
  for (int t = 0; t < TOKENS; t++) {
    copy_vectorized<float, M_TILE>(
        y + t * M_TILE, raw + (t * SLOT + row * (HIDDEN / 4) + piece * M_TILE));
  }
  event1();
}

// Once per call, after gemv_residual: the rest of the input stage.
void gemv_prep(float *raw, bfloat16 *sums, int k_blocks, int prologue,
               int tokens) {
  event0();
  for (int t = 0; t < tokens; t++) {
    prep(raw, sums, k_blocks, prologue, t);
  }
  event1();
}

// Before each round's k loop.
void gemv_zero(float *y_acc) {
  event0();
  zero_vectorized<float, TOKENS * M_TILE, 1, false>(y_acc);
  event1();
}

// y_acc[t] += block * x_t[k * K_TILE : (k + 1) * K_TILE] for each token.
void gemv_block(const q4nx_block_t *w, float *raw, float *y_acc,
                const bfloat16 *sums, int k, int prologue, int tokens) {
  event0();
  const bf16 *b[TOKENS];
  float *c[TOKENS];
  const bf16 *hi[TOKENS];
  const bf16 *lo[TOKENS];
  for (int t = 0; t < TOKENS; t++) {
    b[t] = token_x(raw, prologue, t) + k * K_TILE;
    c[t] = y_acc + t * M_TILE;
    hi[t] = sums + t * SUMS + k * (K_TILE / 32);
    lo[t] = hi[t] + SUMS / 2;
  }
  if (TOKENS == 2 && tokens == 2) {
    accumulate<M_TILE, K_TILE, TOKENS>(w, b, c, hi, lo);
  } else {
    accumulate<M_TILE, K_TILE, 1>(w, b, c, hi, lo);
  }
  event1();
}

// After each round's k loop, per token. EPILOGUE_STORE writes the fp32 sums;
// with EPILOGUE_SWIGLU the core's rows are 16 gate rows then their 16 up
// rows, and it writes silu(gate) * up as 16 bf16 at the start of the token's
// M_TILE values (the rest is padding the sequence discards).
// EPILOGUE_SWIGLU_SCALED also multiplies by the fp32 scale in RTP word
// RTP_SCALES + round * TOKENS + t (an MoE router weight; round = expert).
void gemv_store(float *y, const float *y_acc, const int32_t *rtp, int round) {
  event0();
  const int epilogue = rtp[RTP_EPILOGUE];
  for (int t = 0; t < TOKENS; t++) {
    float *yt = y + t * M_TILE;
    const float *acc = y_acc + t * M_TILE;
    if (epilogue == EPILOGUE_SWIGLU || epilogue == EPILOGUE_SWIGLU_SCALED) {
      static_assert(M_TILE == 32, "SwiGLU pairs 16 gate rows with 16 up rows");
      aie::vector<float, 16> g = aie::load_v<16>(acc);
      aie::vector<float, 16> u = aie::load_v<16>(acc + 16);
      if (epilogue == EPILOGUE_SWIGLU_SCALED) {
        u = aie::mul(u, __builtin_bit_cast(
                            float, rtp[RTP_SCALES + round * TOKENS + t]))
                .template to_vector<float>();
      }
      const aie::vector<float, 16> sig = sigmoid(g);
      aie::accum<accfloat, 16> out;
      out.from_vector(aie::mul(aie::mul(g, sig).template to_vector<float>(), u)
                          .template to_vector<float>());
      aie::store_v((bfloat16 *)yt, out.template to_vector<bfloat16>());
    } else {
      copy_vectorized<float, M_TILE>(yt, acc);
    }
  }
  event1();
}
}
