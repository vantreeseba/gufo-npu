// Qwen3-VL-Embedding-2B's ViT attention core program (vit_attention.py):
// bidirectional flash attention for one head of dimension 64, a 64-row query
// block per core over the keys in blocks of 32, with fp32 softmax state and
// output. Both products run on the bfp16 mmul (8x8x8, fp32 accumulators) as
// 3-term splits, x = hi + lo with hi = bfp16(x) and lo = bfp16(x - hi):
// A B ~ Ah Bh + Al Bh + Ah Bl. The host splits Q, K and V (src/core/bfp16.hpp);
// the core splits P.
//
// Objects (bytes; a split tile is an 8x8 tile as bfp16ebs8 block vectors, hi
// then lo, 72 bytes each, one block per tile row):
//   q   the query block: split tiles [qt][dt] (8 queries x 8 dims), pre-scaled
//       by log2(e) / sqrt(64) so that the softmax is in base 2.
//   kv  one key block: split k tiles [dt][kt] (8 keys x 8 dims), BK fp32
//       biases (0, or -inf for padding keys), then split v^T tiles (8 dims x
//       8 keys) in groups of 4 dim tiles, [g][kt][j] for dim tile 4g + j.
//   o   a quarter of the output (query tiles 2c and 2c + 1), bf16 tiles
//       [qt][dt].
//
// Per key block, three passes over the 8 query tiles: A computes S = Q K^T
// (plus the bias), the running row max m (an integer, so the rescale 2^(m_old
// - m) is exact) and S - m; B turns S - m into P = 2^(S - m), splits it in
// place, and adds its row sums into l; C rescales O where m moved and adds
// P V. The output is O / l, rounded to bf16.
#include <aie_api/aie.hpp>
#include <stdint.h>

#define BQ 64
#define BK 32
#define HD 64
#define UNROLL _Pragma("clang loop unroll(full)")

namespace {

using bv = aie::block_vector<bfp16ebs8, 64>;
using AccT = aie::accum<accfloat, 64>;
using Acc32T = aie::accum<accfloat, 32>;
using in_bv = aie::block_vector_input_buffer_stream<bfp16ebs8, 64>;
using out_bv = aie::block_vector_output_buffer_stream<bfp16ebs8, 64>;
using vbf64 = aie::vector<bfloat16, 64>;
using vbf32 = aie::vector<bfloat16, 32>;
using vf8 = aie::vector<float, 8>;
using vf16 = aie::vector<float, 16>;
using vf32 = aie::vector<float, 32>;
using vf64 = aie::vector<float, 64>;
using vi8 = aie::vector<int32_t, 8>;

constexpr int kQT = BQ / 8, kKT = BK / 8, kDT = HD / 8;
constexpr int kBV = 72;
constexpr int kQBytes = kQT * kDT * 2 * kBV;
constexpr int kKSplit = kDT * kKT * 2 * kBV;
constexpr int kKBytes = kKSplit + BK * 4;
constexpr int kVBytes = kDT * kKT * 2 * kBV;
// Scratch: m (int32 per row); its change d (int32 per row); whether m moved,
// per query tile; l (row pattern per query tile); the bias pattern per key
// tile; S - m and then P (fp32 [qt][kt] tiles, 256 bytes), which pass B
// overwrites with the split P ([qt][kt][hi, lo], 144 bytes a tile, never
// reaching a tile it has not read).
constexpr int kOffM = 0;
constexpr int kOffD = kOffM + BQ * 4;
constexpr int kOffC = kOffD + BQ * 4;
constexpr int kOffL = kOffC + 64;
constexpr int kOffB = kOffL + kQT * 64 * 4;
constexpr int kOffP = kOffB + kKT * 64 * 4;
constexpr int kScratch = kOffP + kQT * kKT * 64 * 4;
static_assert(kScratch == VIT_ATTN_SCRATCH, "VIT_ATTN_SCRATCH must match the layout");
static_assert(kQBytes == VIT_ATTN_Q_BYTES, "VIT_ATTN_Q_BYTES must match the layout");
static_assert(kKBytes + kVBytes == VIT_ATTN_KV_BYTES, "VIT_ATTN_KV_BYTES must match");
static_assert(kOffP % 64 == 0, "P must be 64-byte aligned");

inline bfp16ebs8* bvp(uint8_t* p) { return reinterpret_cast<bfp16ebs8*>(p); }
inline int32_t* MRow(uint8_t* s) { return reinterpret_cast<int32_t*>(s + kOffM); }
inline int32_t* DRow(uint8_t* s) { return reinterpret_cast<int32_t*>(s + kOffD); }
inline int32_t* Changed(uint8_t* s) { return reinterpret_cast<int32_t*>(s + kOffC); }
inline float* LPat(uint8_t* s) { return reinterpret_cast<float*>(s + kOffL); }
inline float* BPat(uint8_t* s) { return reinterpret_cast<float*>(s + kOffB); }
inline float* PBuf(uint8_t* s) { return reinterpret_cast<float*>(s + kOffP); }

inline AccT LoadAcc(const float* p) {
  AccT a;
  a.from_vector(aie::load_v<64>(p));
  return a;
}

// Lane (r, c) = v[r] for rows 0-3 (h0) and 4-7 (h1): three rounds of
// zipping a vector with itself, each doubling every value.
inline void RowPat(vf8 v, vf32& h0, vf32& h1) {
  const vf16 a = aie::concat(v, v);
  const vf16 x = aie::interleave_zip(a, a, 1).first;  // v0 v0 v1 v1 .. v7 v7
  const auto y = aie::interleave_zip(x, x, 1);        // v0 x4 .. v3 x4 | v4 x4 ..
  const auto z0 = aie::interleave_zip(y.first, y.first, 1);
  const auto z1 = aie::interleave_zip(y.second, y.second, 1);
  h0 = aie::concat(z0.first, z0.second);
  h1 = aie::concat(z1.first, z1.second);
}

inline vf64 RowPat64(vf8 v) {
  vf32 h0, h1;
  RowPat(v, h0, h1);
  return aie::concat(h0, h1);
}

// 2^x for two sets of 32 lanes in lockstep (x <= 1): x = k + f with k an
// integer and |f| <= 1/2, 2^f by a degree-3 polynomial (relative minimax on
// [-1/2, 1/2], 7.6e-5) with bf16 limbs where they matter: 2.2e-4 max and
// 6.8e-5 rms relative error against float64 (EXPERIMENTS.md). x is clamped
// at -64: 2^-64 is nothing next to a row's largest weight (>= 1/2).
inline void Exp2x2(vf32* xs) {
  constexpr int N = 2;
  const vf32 magic = aie::broadcast<float, 32>(12582912.0f);  // 1.5 * 2^23
  const vbf32 one = aie::broadcast<bfloat16, 32>(bfloat16(1.0f));
  const vbf32 c3 = aie::broadcast<bfloat16, 32>(bfloat16(0.0552040201f));
  auto acc = [](float c) { return Acc32T(aie::broadcast<float, 32>(c)); };
  vf32 xm[N];
  vbf32 f_hi[N], f_lo[N], a_hi[N], b_hi[N], b_lo[N];
  Acc32T p[N];
  // max(x, -64) as an unsigned min of the bits: positive values have the
  // smaller patterns, negative ones order by magnitude (-inf the largest).
  UNROLL for (int i = 0; i < N; ++i) xs[i] =
      aie::min(xs[i].cast_to<uint32_t>(), aie::broadcast<uint32_t, 32>(0xc2800000u))
          .cast_to<float>();
  UNROLL for (int i = 0; i < N; ++i) xm[i] = aie::add(xs[i], magic);  // k in the low bits
  // f = x - k, k = xm - magic, both exact in fp32.
  UNROLL for (int i = 0; i < N; ++i) p[i] = Acc32T(aie::sub(xs[i], aie::sub(xm[i], magic)));
  UNROLL for (int i = 0; i < N; ++i) f_hi[i] = p[i].to_vector<bfloat16>();
  UNROLL for (int i = 0; i < N; ++i) f_lo[i] = aie::msc(p[i], f_hi[i], one).to_vector<bfloat16>();
  UNROLL for (int i = 0; i < N; ++i)
      a_hi[i] = aie::mac(acc(0.2426138645f), c3, f_hi[i]).to_vector<bfloat16>();
  UNROLL for (int i = 0; i < N; ++i)
      p[i] = aie::mac(aie::mac(acc(0.693255037f), a_hi[i], f_hi[i]), a_hi[i], f_lo[i]);
  UNROLL for (int i = 0; i < N; ++i) b_hi[i] = p[i].to_vector<bfloat16>();
  UNROLL for (int i = 0; i < N; ++i) b_lo[i] = aie::msc(p[i], b_hi[i], one).to_vector<bfloat16>();
  UNROLL for (int i = 0; i < N; ++i) p[i] = aie::mac(acc(0.9999277373f), b_hi[i], f_hi[i]);
  UNROLL for (int i = 0; i < N; ++i) p[i] = aie::mac(p[i], b_hi[i], f_lo[i]);
  UNROLL for (int i = 0; i < N; ++i) p[i] = aie::mac(p[i], b_lo[i], f_hi[i]);
  // 2^k * 2^f: k into the exponent field (xm's bits << 23 is k << 23).
  UNROLL for (int i = 0; i < N; ++i)
      xs[i] = aie::add(p[i].to_vector<float>().cast_to<int32_t>(),
                       aie::upshift(xm[i].cast_to<int32_t>(), 23))
                  .cast_to<float>();
}

// fp32 tile x a row factor exact in bf16 (a power of two): two limbs.
inline AccT ScaleRows(AccT y, vbf64 f) {
  const vbf64 ones = aie::broadcast<bfloat16, 64>(1.0f);
  const vbf64 hi = y.to_vector<bfloat16>();
  const vbf64 lo = aie::msc(y, hi, ones).to_vector<bfloat16>();
  return aie::mac(aie::mul(hi, f), lo, f);
}

// The row factor 2^d (d <= 0 per row, -inf rows to 0) as a bf16 row pattern.
inline vbf64 Alpha(const int32_t* d) {
  const vi8 e = aie::max(aie::add(aie::load_v<8>(d), aie::broadcast<int32_t, 8>(127)),
                         aie::zeros<int32_t, 8>());
  AccT fa;
  fa.from_vector(RowPat64(aie::upshift(e, 23).cast_to<float>()));
  return fa.to_vector<bfloat16>();
}

inline bv Ones() {
  AccT o;
  o.from_vector(aie::broadcast<float, 64>(1.0f));
  return o.to_vector<bfp16ebs8>();
}

inline bv NegEye() {
  struct Eye {
    float v[64];
    constexpr Eye() : v{} {
      for (int i = 0; i < 64; ++i) v[i] = (i / 8 == i % 8) ? -1.0f : 0.0f;
    }
  };
  alignas(64) static constexpr Eye kEye;
  AccT a;
  a.from_vector(aie::load_v<64>(kEye.v));
  return a.to_vector<bfp16ebs8>();
}

}  // namespace

extern "C" {

// A new query block: O = 0, m = -2^20, l = 0.
void attn_begin(float* acc, uint8_t* scratch) {
  for (int i = 0; i < BQ * HD; i += 64) aie::store_v(acc + i, aie::zeros<float, 64>());
  for (int i = 0; i < BQ; i += 8)
    aie::store_v(MRow(scratch) + i, aie::broadcast<int32_t, 8>(-(1 << 20)));
  for (int i = 0; i < kQT * 64; i += 64) aie::store_v(LPat(scratch) + i, aie::zeros<float, 64>());
}

// One key block.
void attn_block(uint8_t* q, uint8_t* kv, float* acc, uint8_t* scratch) {
  uint8_t* kobj = kv;
  uint8_t* vobj = kv + kKBytes;
  ::aie::rounding_mode saved = ::aie::swap_rounding(aie::rounding_mode::conv_even);
  const float* bias = reinterpret_cast<const float*>(kobj + kKSplit);
  // The bias as a column pattern per key tile (lane (r, c) = bias[c]).
  for (int kt = 0; kt < kKT; ++kt) {
    const vf8 b = aie::load_v<8>(bias + kt * 8);
    vf64 pat;
    for (int r = 0; r < 8; ++r) pat.insert(r, b);
    aie::store_v(BPat(scratch) + kt * 64, pat);
  }
  int32_t* m_row = MRow(scratch);
  int32_t* d_row = DRow(scratch);
  int32_t* changed = Changed(scratch);
  float* l_pat = LPat(scratch);
  float* p_buf = PBuf(scratch);

  // Pass A, per query tile: S = Q K^T, the running max m, and S - m.
  for (int qt = 0; qt < kQT; ++qt) {
    AccT s[kKT];
    UNROLL for (int kt = 0; kt < kKT; ++kt) s[kt] = LoadAcc(BPat(scratch) + kt * 64);
    {
      in_bv qi(bvp(q + qt * kDT * 2 * kBV));
      in_bv ki(bvp(kobj));
      // Four key tiles: four accumulator chains, all three terms on each.
      UNROLL for (int dt = 0; dt < kDT; ++dt) {
        const bv qh = qi.pop(), ql = qi.pop();
        UNROLL for (int kt = 0; kt < kKT; ++kt) {
          const bv kh = ki.pop(), kl = ki.pop();
          s[kt] = ::mac_8x8_8x8T(qh, kh, s[kt]);
          s[kt] = ::mac_8x8_8x8T(ql, kh, s[kt]);
          s[kt] = ::mac_8x8_8x8T(qh, kl, s[kt]);
        }
      }
    }
    // Row max bound: max over the key tiles in bf16, an 8x8 transpose, then
    // the max over the 8 column groups.
    vbf64 mx = s[0].to_vector<bfloat16>();
    UNROLL for (int kt = 1; kt < kKT; ++kt) mx = aie::max(mx, s[kt].to_vector<bfloat16>());
    const vbf64 mt = aie::transpose(mx, 8, 8);
    const aie::vector<bfloat16, 8> rmax = aie::max(
        aie::max(aie::max(mt.extract<8>(0), mt.extract<8>(1)),
                 aie::max(mt.extract<8>(2), mt.extract<8>(3))),
        aie::max(aie::max(mt.extract<8>(4), mt.extract<8>(5)),
                 aie::max(mt.extract<8>(6), mt.extract<8>(7))));
    // m = max(m_old, round(rowmax + 1)), an integer, so alpha = 2^(m_old - m)
    // is exact.
    aie::accum<accfloat, 8> ra;
    ra.from_vector(rmax);
    const vf8 magic = aie::broadcast<float, 8>(12582912.0f);
    const vi8 magic_bits = aie::broadcast<int32_t, 8>(0x4b400000);
    const vf8 rounded =
        aie::add(aie::add(ra.to_vector<float>(), aie::broadcast<float, 8>(1.0f)), magic);
    const vi8 cand = aie::sub(rounded.cast_to<int32_t>(), magic_bits);
    const vi8 m_old = aie::load_v<8>(m_row + qt * 8);
    const vi8 m_i = aie::max(cand, m_old);
    const vi8 d = aie::sub(m_old, m_i);
    changed[qt] = aie::reduce_add(d);
    aie::store_v(m_row + qt * 8, m_i);
    aie::store_v(d_row + qt * 8, d);
    const vf64 m_pat = RowPat64(aie::sub(aie::add(m_i, magic_bits).cast_to<float>(), magic));
    UNROLL for (int kt = 0; kt < kKT; ++kt)
        aie::store_v(p_buf + (qt * kKT + kt) * 64, aie::sub(s[kt], m_pat).to_vector<float>());
  }

  // Pass B, per query tile: P = 2^(S - m) a tile at a time, split in place,
  // and l = alpha l + its row sums (on the mmul: lane (r, c) = sum of row r).
  {
    const bv ones = Ones();
    const bv neg_eye = NegEye();
    out_bv po(bvp(reinterpret_cast<uint8_t*>(p_buf)));
    for (int qt = 0; qt < kQT; ++qt) {
      AccT sums = aie::zeros<accfloat, 64>();
      for (int kt = 0; kt < kKT; ++kt) {
        const float* src = p_buf + (qt * kKT + kt) * 64;
        vf32 xs[2] = {aie::load_v<32>(src), aie::load_v<32>(src + 32)};
        Exp2x2(xs);
        AccT p;
        p.from_vector(aie::concat(xs[0], xs[1]));
        const bv ph = p.to_vector<bfp16ebs8>();
        const bv pl = AccT(::mac_8x8_8x8T(ph, neg_eye, p)).to_vector<bfp16ebs8>();
        po.push(ph);
        po.push(pl);
        sums = ::mac_8x8_8x8T(ph, ones, sums);
        sums = ::mac_8x8_8x8T(pl, ones, sums);
      }
      AccT l = LoadAcc(l_pat + qt * 64);
      if (changed[qt] != 0) l = ScaleRows(l, Alpha(d_row + qt * 8));
      aie::store_v(l_pat + qt * 64, aie::add(l, sums.to_vector<float>()).to_vector<float>());
    }
  }

  // Pass C, per query tile: O = alpha O where m moved, then O += P V, four
  // output tiles at a time.
  uint8_t* p_split = reinterpret_cast<uint8_t*>(p_buf);
  for (int qt = 0; qt < kQT; ++qt) {
    float* o_qt = acc + qt * kDT * 64;
    if (changed[qt] != 0) {
      const vbf64 f = Alpha(d_row + qt * 8);
      _Pragma("clang loop unroll(disable)")
      for (int dt = 0; dt < kDT; ++dt)
        aie::store_v(o_qt + dt * 64, ScaleRows(LoadAcc(o_qt + dt * 64), f).to_vector<float>());
    }
    in_bv vi(bvp(vobj));
    UNROLL for (int g = 0; g < kDT; g += 4) {
      AccT o[4];
      UNROLL for (int j = 0; j < 4; ++j) o[j] = LoadAcc(o_qt + (g + j) * 64);
      in_bv pi(bvp(p_split + qt * kKT * 2 * kBV));
      UNROLL for (int kt = 0; kt < kKT; ++kt) {
        const bv ph = pi.pop(), pl = pi.pop();
        UNROLL for (int j = 0; j < 4; ++j) {
          const bv vh = vi.pop(), vl = vi.pop();
          o[j] = ::mac_8x8_8x8T(ph, vh, o[j]);
          o[j] = ::mac_8x8_8x8T(pl, vh, o[j]);
          o[j] = ::mac_8x8_8x8T(ph, vl, o[j]);
        }
      }
      UNROLL for (int j = 0; j < 4; ++j) aie::store_v(o_qt + (g + j) * 64, o[j].to_vector<float>());
    }
  }
  ::aie::set_rounding(saved);
}

// Output quarter c: query tiles 2c and 2c + 1 as O / l, rounded to bf16.
void attn_out(bfloat16* oobj, float* acc, uint8_t* scratch, int32_t chunk) {
  ::aie::rounding_mode saved = ::aie::swap_rounding(aie::rounding_mode::conv_even);
  const float* l_pat = LPat(scratch);
  for (int j = 0; j < 2; ++j) {
    const int qt = chunk * 2 + j;
    alignas(32) float inv[8];
    for (int r = 0; r < 8; ++r) inv[r] = 1.0f / l_pat[qt * 64 + r * 8];
    vf32 i0, i1;
    RowPat(aie::load_v<8>(inv), i0, i1);
    for (int dt = 0; dt < kDT; ++dt) {
      const float* o = acc + (qt * kDT + dt) * 64;
      AccT r;
      r.from_vector(aie::concat(aie::mul(aie::load_v<32>(o), i0).to_vector<float>(),
                                aie::mul(aie::load_v<32>(o + 32), i1).to_vector<float>()));
      aie::store_v(oobj + (j * kDT + dt) * 64, r.to_vector<bfloat16>());
    }
  }
  ::aie::set_rounding(saved);
}

}  // extern "C"
