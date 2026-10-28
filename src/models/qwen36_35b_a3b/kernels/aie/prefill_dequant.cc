// Qwen3.6-35B-A3B prefill: one compact q4nx block (src/core/q4nx.hpp: bf16 d
// and dmin per row, uint8 sc and int8 mn per row and 32 inputs, 4-bit codes)
// to bfp16ebs8, in the order iron.operators.flm.GEMM reads its B operand.
//
// Adapted from mlir-aie's aie_kernels/quant/q4nx_dequant.cc (Apache-2.0 WITH
// LLVM-exception), which reads FastFlowLM's 5-bit layout (a bf16 scale and
// minimum per row and 32 inputs). The codes are laid out the same way in both
// ([m / 16][k][m % 16], low nibble first), so only the tables differ: a
// preamble expands d * sc and dmin * mn into that layout's scale and minimum
// tables, then the same loop runs.
//
// Each weight dmin * mn + d * sc * code is rounded once, from fp32 straight to
// bfp16, as the host packs it (src/core/bfp16.hpp): d * sc and dmin * mn
// (8-bit integers times bf16, exact in fp32) are kept as bf16 hi + lo pairs,
// the codes are exact in bf16, and the operands are transposed into the
// GEMM's order before the MACs rather than after a bf16 rounding. Rounds to
// nearest even (a core starts in floor mode). Geometry arrives as -D flags.
#include "aie_kernels/aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

#if !defined(Q4NX_M_TILE) || !defined(Q4NX_K_TILE) || !defined(Q4NX_GROUP) || \
    !defined(Q4NX_CT_K) || !defined(Q4NX_S) || !defined(Q4NX_T)
#error "qwen36_q4nx_dequant.cc needs its geometry -D defined"
#endif

namespace {

constexpr int M_TILE = Q4NX_M_TILE;
constexpr int K_TILE = Q4NX_K_TILE;
constexpr int GROUP = Q4NX_GROUP;
constexpr int CT_K = Q4NX_CT_K;
constexpr int SS = Q4NX_S;
constexpr int TT = Q4NX_T;

constexpr int PR = 16;
constexpr int GROUPS = K_TILE / GROUP;
constexpr int SCALES = M_TILE * GROUPS;
constexpr int TILE_VALUES = SS * TT;
constexpr int STEPS = CT_K / SS;
constexpr int N8 = M_TILE / TT;

// Compact block: d [M], dmin [M], sc [GROUPS][M], mn [GROUPS][M], codes.
constexpr int OFF_DMIN = M_TILE * 2;
constexpr int OFF_SC = M_TILE * 4;
constexpr int OFF_MN = OFF_SC + SCALES;
constexpr int OFF_CODES = OFF_MN + SCALES;

static_assert(TT == 8 && SS == 8, "8x8 bfp16 tiles");
static_assert(PR * SS == 128, "one 512-bit load holds 128 nibbles");
static_assert(M_TILE == 2 * PR, "two 16-row strips per block");
static_assert(K_TILE % CT_K == 0 && CT_K % SS == 0 && GROUP % SS == 0, "");

constexpr int K_STEPS = K_TILE / SS;

struct GroupRow {
  uint16_t v[K_STEPS];
};

constexpr GroupRow group_row() {
  GroupRow t{};
  for (int s = 0; s < K_STEPS; s++)
    t.v[s] = (uint16_t)(s * SS / GROUP * M_TILE * sizeof(bfloat16));
  return t;
}

constexpr GroupRow GROW = group_row();

struct HalfMode {
  uint8_t v[N8];
};

constexpr HalfMode half_mode() {
  HalfMode t{};
  for (int n8 = 0; n8 < N8; n8++)
    t.v[n8] = (uint8_t)(T32_16x2_lo + (n8 & 1));
  return t;
}

constexpr HalfMode HMODE = half_mode();

// The expanded tables, [GROUPS][M] each: the hi halves of the scales and of
// the minima, then their lo halves.
alignas(64) bfloat16 tables[4 * SCALES];
constexpr int OFF_MIN_HI = SCALES;
constexpr int OFF_SCALE_LO = 2 * SCALES;
constexpr int OFF_MIN_LO = 3 * SCALES;

aie::vector<bfloat16, 32> rep8(const bfloat16 *p) {
  aie::vector<uint32_t, 16> v =
      aie::load_v<TT>(p).template grow<32>().template cast_to<uint32_t>();
  v = ::shuffle(v, v, T128_2x4_lo);
  v = ::shuffle(v, v, T128_2x4_lo);
  return v.template cast_to<bfloat16>();
}

// A row pattern of rep8 as a full 8 x 8 tile, in the GEMM's order.
aie::vector<bfloat16, TT * SS> tile(const aie::vector<bfloat16, 32> &v) {
  return aie::transpose(aie::concat(v, v), SS, TT);
}

// tables = d * sc and dmin * mn as hi + lo bf16 pairs (exact: an 8-bit
// integer times a bf16 has at most 16 significant bits).
void expand(const uint8_t *__restrict qw) {
  const bfloat16 *d = (const bfloat16 *)qw;
  const bfloat16 *dmin = (const bfloat16 *)(qw + OFF_DMIN);
  const uint8_t *sc = qw + OFF_SC;
  const int8_t *mn = (const int8_t *)(qw + OFF_MN);
  for (int h = 0; h < M_TILE / PR; h++) {
    const aie::vector<bfloat16, PR> dv = aie::load_v<PR>(d + h * PR);
    const aie::vector<bfloat16, PR> mv = aie::load_v<PR>(dmin + h * PR);
    AIE_LOOP_RANGE(GROUPS, GROUPS)
    for (int g = 0; g < GROUPS; g++) {
      aie::accum<accfloat, PR> s;
      s.from_vector(aie::to_float(aie::load_v<PR>(sc + g * M_TILE + h * PR), 0));
      aie::accum<accfloat, PR> m;
      m.from_vector(aie::to_float(aie::load_v<PR>(mn + g * M_TILE + h * PR), 0));
      // Integers below 256 are exact in bf16, and so are the products in
      // fp32; each splits into hi + lo without loss.
      aie::accum<accfloat, PR> ps =
          aie::mul(s.template to_vector<bfloat16>(), dv);
      aie::accum<accfloat, PR> pm =
          aie::mul(m.template to_vector<bfloat16>(), mv);
      const int at = g * M_TILE + h * PR;
      const aie::vector<bfloat16, PR> s_hi = ps.template to_vector<bfloat16>();
      const aie::vector<bfloat16, PR> m_hi = pm.template to_vector<bfloat16>();
      aie::store_v(tables + at, s_hi);
      aie::store_v(tables + OFF_MIN_HI + at, m_hi);
      aie::store_v(tables + OFF_SCALE_LO + at,
                   aie::sub(ps, s_hi).template to_vector<bfloat16>());
      aie::store_v(tables + OFF_MIN_LO + at,
                   aie::sub(pm, m_hi).template to_vector<bfloat16>());
    }
  }
}

}  // namespace

extern "C" {

void qwen36_q4nx_dequant(const uint8_t *__restrict qw,
                         bfp16ebs8 *__restrict out) {
  event0();
  aie::rounding_mode saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);

  expand(qw);
  const bfloat16 *scales = tables;
  const uint8_t *qs = qw + OFF_CODES;

  aie::block_vector_output_buffer_stream<bfp16ebs8, TILE_VALUES> s_out(out);

  for (int ks = 0; ks < K_TILE / CT_K; ks++) {
    const uint8_t *q_it = qs + ks * CT_K * PR / 2;
    const uint8_t *s_it = (const uint8_t *)scales;
    const uint16_t *row_it = GROW.v + ks * STEPS;
    const uint8_t *mode_it = HMODE.v;
    dims_3d_t q_dims =
        dims_3d_from_steps(STEPS, SS * PR / 2, 2, 0, K_TILE * PR / 2);
    dims_2d_t s_dims = dims_2d_from_steps(STEPS, 0, TT * sizeof(bfloat16));
    dims_2d_t row_dims = dims_2d_from_steps(STEPS, sizeof(uint16_t), 0);
    dims_2d_t mode_dims = dims_2d_from_steps(STEPS, 0, 1);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(N8 * STEPS, N8 * STEPS)
    for (int j = 0; j < N8 * STEPS; j++) {
      aie::vector<uint32_t, 16> w = aie::load_v<16>((const uint32_t *)q_it);
      aie::vector<uint32_t, 16> half = ::shuffle(w, w, *mode_it);
      aie::vector<uint4, TT * SS> q =
          half.template extract<8>(0).template cast_to<uint4>();
      aie::accum<accfloat, TT * SS> qf;
      qf.from_vector(aie::to_float(q, 0));
      aie::vector<bfloat16, TT * SS> qb = qf.template to_vector<bfloat16>();

      const bfloat16 *sp = (const bfloat16 *)(s_it + *row_it);
      // The tile in the GEMM's order: every operand transposed (the weight
      // is elementwise in them), then built in fp32 and rounded once.
      const aie::vector<bfloat16, TT * SS> qt = aie::transpose(qb, SS, TT);
      aie::accum<accfloat, TT * SS> acc(tile(rep8(sp + OFF_MIN_HI)));
      acc = aie::add(acc, tile(rep8(sp + OFF_MIN_LO)));
      acc = aie::mac(acc, tile(rep8(sp)), qt);
      acc = aie::mac(acc, tile(rep8(sp + OFF_SCALE_LO)), qt);
      s_out << acc.template to_vector<bfp16ebs8>();

      q_it = add_3d_byte(q_it, q_dims);
      s_it = add_2d_byte(s_it, s_dims);
      row_it = add_2d_byte(row_it, row_dims);
      mode_it = add_2d_byte(mode_it, mode_dims);
    }
  }

  aie::set_rounding(saved_rounding);
  event1();
}

}  // extern "C"
