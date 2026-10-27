// q4nx matrix-vector product for decode: y = W x, one token.
//
// Each core owns Q4NX_M_TILE out-features per round and streams their q4nx
// blocks (Q4NX_M_TILE x Q4NX_K_TILE) past the token. The token buffer holds
// GEMV_MAX_DIM values; the number of k blocks in use arrives at run time, so
// one core program serves every shape.
//
// The block layout is IRON's q4nx.h. The accumulation is adapted from its
// q4nx_accumulate, keeping more precision: each block's dot product and each
// per-32 token sum are carried as two bf16 halves (hi + lo) instead of one,
// and the output stays fp32. Geometry arrives as -D flags.
#include "aie_kernels/aie_kernel_utils.h"
#include "aie_kernels/common/zero.h"
#include "aie_kernels/flm_gemma4/q4nx.h"
#include "aie_kernels/flm_gemma4/utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

#if !defined(Q4NX_M_TILE) || !defined(Q4NX_K_TILE) || !defined(Q4NX_GROUP) || \
    !defined(GEMV_MAX_DIM)
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

}  // namespace

extern "C" {

// Once per token: the per-32 sums of the first k_blocks * K_TILE values of x,
// which fold every block's minima into one MAC, as hi and lo bf16 halves
// (sums[0 : MAX_DIM / 32] and sums[MAX_DIM / 32 : MAX_DIM / 16]).
//
// Also selects round-to-nearest-even for every float-to-bf16 narrowing that
// follows. A core powers up in floor mode, which truncates each block's dot
// products and biases the sum over the k blocks (about 1% relative error at
// dim 2048); the mode is core-global and sticky.
void gemv_prep(bfloat16 *x, bfloat16 *sums, int k_blocks) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 *lo = sums + MAX_DIM / 32;
  for (int g = 0; g < k_blocks * (K_TILE / 32); g++) {
    const float sum = aie::reduce_add(
        aie::accum<accfloat, 32>(aie::load_v<32>(x + 32 * g))
            .template to_vector<float>());
    const bfloat16 hi = (bfloat16)sum;
    sums[g] = hi;
    lo[g] = (bfloat16)(sum - (float)hi);
  }
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

// After each round's k loop: the fp32 sums go out unrounded.
void gemv_store(float *y, const float *y_acc) {
  event0();
  copy_vectorized<float, M_TILE>(y, y_acc);
  event1();
}
}
