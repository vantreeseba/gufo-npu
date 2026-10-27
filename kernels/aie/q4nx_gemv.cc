// q4nx matrix-vector product for decode: y = W x, one token.
//
// Each core owns Q4NX_M_TILE out-features per round and streams their q4nx
// blocks (Q4NX_M_TILE x Q4NX_K_TILE) past the token. With GEMV_NORM=1 the
// token arrives followed by its RMS norm weight and is normalized once, so a
// projection that follows a norm needs no separate dispatch.
//
// Built on IRON's q4nx.h (the block layout and its accumulation). Geometry
// arrives as -D flags.
#include "aie_kernels/aie_kernel_utils.h"
#include "aie_kernels/common/zero.h"
#include "aie_kernels/flm_gemma4/q4nx.h"
#include "aie_kernels/flm_gemma4/rms_norm.h"
#include "aie_kernels/flm_gemma4/utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

#if !defined(Q4NX_M_TILE) || !defined(Q4NX_K_TILE) || !defined(Q4NX_GROUP) || \
    !defined(GEMV_DIM) || !defined(GEMV_NORM)
#error "q4nx_gemv.cc needs its geometry -D defined"
#endif

using q4nx_block_t = q4nx_block<Q4NX_M_TILE, Q4NX_K_TILE>;

namespace {

constexpr int M_TILE = Q4NX_M_TILE;
constexpr int K_TILE = Q4NX_K_TILE;
constexpr int DIM = GEMV_DIM;
constexpr int K_BLOCKS = DIM / K_TILE;

static_assert(Q4NX_GROUP == 32, "a column sum covers exactly one group");
static_assert(K_TILE % 32 == 0, "a k tile must hold whole groups");
static_assert(DIM % K_TILE == 0, "the token must cover whole k tiles");
static_assert(M_TILE % 16 == 0, "rows are processed 16 at a time");

}  // namespace

extern "C" {

// Once per token: optionally RMS-normalize x in place (its weight at x + DIM),
// then the per-32 sums of x that fold every block's minima into one MAC.
//
// Also selects round-to-nearest-even for every float-to-bf16 narrowing that
// follows. A core powers up in floor mode, which truncates each block's dot
// products and biases the sum over the k blocks (about 1% relative error at
// dim 2048); the mode is core-global and sticky.
void gemv_prep(bfloat16 *x, bfloat16 *sums) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
#if GEMV_NORM
  rms_norm<DIM>(x, x, x + DIM);
#endif
  for (int k = 0; k < K_BLOCKS; k++) {
    group_sums_32<K_TILE>(sums + k * (K_TILE / 32), x + k * K_TILE);
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
  q4nx_accumulate<M_TILE, K_TILE>(w, x + k * K_TILE, y_acc,
                                  sums + k * (K_TILE / 32));
  event1();
}

// After each round's k loop.
void gemv_store(bfloat16 *y, const float *y_acc) {
  event0();
  narrow_to_bf16<M_TILE>(y, y_acc);
  event1();
}
}
