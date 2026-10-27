// q4nx matrix-vector product for decode: y = W x, one token.
//
// Each core owns Q4NX_M_TILE out-features per round and streams their q4nx
// blocks (Q4NX_M_TILE x Q4NX_K_TILE) past the token. The token buffer holds
// GEMV_MAX_DIM fp32 values; the number of k blocks in use and the input and
// output stages arrive at run time, so one core program serves every call:
//   input:  plain (x in fp32), or resnorm: rms_norm(H + D) * w, with H + D
//           also written out as the next residual by one column's cores.
//   output: store (fp32), or swiglu: silu(gate) * up from paired rows.
//
// The block layout is IRON's q4nx.h. The accumulation is adapted from its
// q4nx_accumulate, keeping more precision: each block's dot product and each
// per-32 token sum are carried as two bf16 halves (hi + lo) instead of one,
// and the output stays fp32. Geometry arrives as -D flags.
#include "aie_kernels/aie_kernel_utils.h"
#include "aie_kernels/common/exp2_poly.h"
#include "aie_kernels/common/zero.h"
#include "aie_kernels/flm_gemma4/q4nx.h"
#include "aie_kernels/flm_gemma4/utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

#if !defined(Q4NX_M_TILE) || !defined(Q4NX_K_TILE) || !defined(Q4NX_GROUP) || \
    !defined(GEMV_MAX_DIM) || !defined(GEMV_RMS_EPS) || !defined(GEMV_HIDDEN)
#error "q4nx_gemv.cc needs its geometry -D defined"
#endif

using q4nx_block_t = q4nx_block<Q4NX_M_TILE, Q4NX_K_TILE>;

namespace {

constexpr int M_TILE = Q4NX_M_TILE;
constexpr int K_TILE = Q4NX_K_TILE;
constexpr int MAX_DIM = GEMV_MAX_DIM;

static_assert(Q4NX_GROUP == 32, "a column sum covers exactly one group");
static_assert(K_TILE % 32 == 0, "a k tile must hold whole groups");
static_assert(MAX_DIM % K_TILE == 0, "the token must cover whole k tiles");
static_assert(M_TILE % 16 == 0, "rows are processed 16 at a time");

// c[0:M] += dequantize(A) * B[0:N], with the minima folded in through the
// per-32 sums of B (given as hi and lo bf16 halves).
template <int M, int N>
void accumulate(const q4nx_block<M, N> *A, const bf16 *B, float *c,
                const bf16 *sums_hi, const bf16 *sums_lo) {
  constexpr int pr = 16;
  const uint4 *qs_ptr = A->qs;

  AIE_LOOP_RANGE(M / pr, M / pr)
  for (int row = 0; row < M; row += pr) {
    aie::accum<accfloat, pr> c_accum;
    c_accum.from_vector(aie::load_v<pr>(c + row));
    const bf16 *it_B = B;
    uint32_t scale_min_offset = row;

    AIE_LOOP_RANGE(N / 32, N / 32)
    for (int sub_chunk = 0; sub_chunk < N / 32; sub_chunk++) {
      aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
      it_B += 32;
      aie::vector<bf16, pr> a_scales =
          aie::load_v<pr>(A->scales + scale_min_offset);
      aie::vector<bf16, pr> a_mins = aie::load_v<pr>(A->mins + scale_min_offset);
      scale_min_offset += M;

      // 32 columns = 4 uint4 sub-vectors x 8 lanes each, fully unrolled so
      // the scheduler pipelines one flat 32-long MAC chain.
      aie::vector<bf16, pr * 8> a_cc_bf16[4];
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
        aie::vector<uint4, pr * 8> a_cc = aie::load_v<pr * 8>(qs_ptr);
        qs_ptr += pr * 4;
        aie::accum<accfloat, pr * 8> acc;
        acc.from_vector(aie::to_float(a_cc, 0));
        a_cc_bf16[sv] = acc.template to_vector<bf16>();
      }
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

      // The scale multiply takes bf16 operands, so split the fp32 dot into
      // hi + lo: the residual error drops from 2^-9 to about 2^-17.
      aie::vector<bf16, pr> dot_hi = dot.template to_vector<bf16>();
      aie::accum<accfloat, pr> rest = aie::sub(dot, dot_hi);
      aie::vector<bf16, pr> dot_lo = rest.template to_vector<bf16>();
      c_accum = aie::mac(c_accum, dot_hi, a_scales);
      c_accum = aie::mac(c_accum, dot_lo, a_scales);
      c_accum = aie::mac(c_accum, a_mins, sums_hi[sub_chunk]);
      c_accum = aie::mac(c_accum, a_mins, sums_lo[sub_chunk]);
    }
    aie::store_v(c + row, c_accum.template to_vector<float>());
  }
}

// Input stage modes (RTP word 2).
constexpr int PROLOGUE_PLAIN = 0;    // x: fp32
constexpr int PROLOGUE_RESNORM = 1;  // [H fp32 | D fp32 | w bf16]: rms(H + D) w
// Output stage modes (RTP word 3).
constexpr int EPILOGUE_STORE = 0;   // y: M_TILE fp32
constexpr int EPILOGUE_SWIGLU = 1;  // y[m] = silu(acc[m]) acc[m + 16], m < 16

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

// The rest of the input stage, then the per-32 sums of x (hi and lo bf16
// halves at sums[0 : MAX_DIM / 32] and sums[MAX_DIM / 32 :]) that fold every
// block's minima into one MAC. x ends as bf16 at the start of `raw`.
void prep(float *raw, bfloat16 *sums, int k_blocks, int prologue) {
  const int n = k_blocks * K_TILE;
  bfloat16 *x = (bfloat16 *)raw;
  if (prologue == PROLOGUE_RESNORM) {
    norm(raw, (const bfloat16 *)(raw + 2 * n), n);
  } else {
    to_bf16(x, raw, n);
  }
  bfloat16 *lo = sums + MAX_DIM / 32;
  for (int g = 0; g < n / 32; g++) {
    const float sum = aie::reduce_add(
        aie::accum<accfloat, 32>(aie::load_v<32>(x + 32 * g))
            .template to_vector<float>());
    const bfloat16 hi = (bfloat16)sum;
    sums[g] = hi;
    lo[g] = (bfloat16)(sum - (float)hi);
  }
}

}  // namespace

extern "C" {

// Once per token, first: with PROLOGUE_RESNORM, h = H + D in place over H.
// Also selects round-to-nearest-even for every float-to-bf16 narrowing that
// follows: a core powers up in floor mode, which truncates each block's dot
// products and biases the sum over the k blocks (about 1% relative error at
// dim 2048); the mode is core-global and sticky.
void gemv_residual(float *raw, int k_blocks, int prologue) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  if (prologue == PROLOGUE_RESNORM) {
    const int n = k_blocks * K_TILE;
    for (int i = 0; i < n; i += 16) {
      aie::accum<accfloat, 16> acc;
      acc.from_vector(aie::load_v<16>(raw + i));
      acc = aie::add(acc, aie::load_v<16>(raw + n + i));
      aie::store_v(raw + i, acc.template to_vector<float>());
    }
  }
  event1();
}

// On the residual column, between gemv_residual and gemv_prep: M_TILE values
// of h, piece `piece` of this core's quarter (`row` of 4).
void gemv_residual_out(float *y, const float *raw, int row, int piece) {
  event0();
  copy_vectorized<float, M_TILE>(
      y, raw + (row * (HIDDEN / 4) + piece * M_TILE));
  event1();
}

// Once per token, after gemv_residual: the rest of the input stage.
void gemv_prep(float *raw, bfloat16 *sums, int k_blocks, int prologue) {
  event0();
  prep(raw, sums, k_blocks, prologue);
  event1();
}

// Before each round's k loop.
void gemv_zero(float *y_acc) {
  event0();
  zero_vectorized<float, M_TILE, 1, false>(y_acc);
  event1();
}

// y_acc += block * x[k * K_TILE : (k + 1) * K_TILE].
void gemv_block(const q4nx_block_t *w, const bfloat16 *x, float *y_acc,
                const bfloat16 *sums, int k) {
  event0();
  accumulate<M_TILE, K_TILE>(w, x + k * K_TILE, y_acc, sums + k * (K_TILE / 32),
                             sums + MAX_DIM / 32 + k * (K_TILE / 32));
  event1();
}

// After each round's k loop. EPILOGUE_STORE writes the fp32 sums; with
// EPILOGUE_SWIGLU the core's rows are 16 gate rows then their 16 up rows,
// and it writes silu(gate) * up to the first 16 values (the rest are zero).
void gemv_store(float *y, const float *y_acc, int epilogue) {
  event0();
  if (epilogue == EPILOGUE_SWIGLU) {
    static_assert(M_TILE == 32, "SwiGLU pairs 16 gate rows with 16 up rows");
    aie::vector<float, 16> g = aie::load_v<16>(y_acc);
    aie::vector<float, 16> u = aie::load_v<16>(y_acc + 16);
    const aie::vector<float, 16> sig = sigmoid(g);
    aie::vector<float, 16> out =
        aie::mul(aie::mul(g, sig).template to_vector<float>(), u)
            .template to_vector<float>();
    aie::store_v(y, out);
    aie::store_v(y + 16, aie::zeros<float, 16>());
  } else {
    copy_vectorized<float, M_TILE>(y, y_acc);
  }
  event1();
}
}
