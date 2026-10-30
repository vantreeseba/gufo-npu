#include "models/qwen3vl_embed_2b/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <utility>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "core/simd.hpp"
#include "core/vec.hpp"

namespace gufo_npu::qwen3vl_embed_2b {
namespace {

using vec::Dot;
using vec::RmsNorm;

// Attention work items: query rows per item, keys per online-softmax block
// (one transposed key tile), query vectors per register tile.
constexpr int kQueryBlock = 16;
constexpr int kKeyBlock = 64;
constexpr int kTile = 4;

// y += s * x over n values.
void AddScaled(float s, const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] += s * x[i];
  }
}

// Tiles start on a cache line, so full-width loads never split one.
template<typename T>
struct CacheAligned {
  using value_type = T;
  static constexpr std::align_val_t kAlign{64};
  CacheAligned() = default;
  template<typename U>
  explicit CacheAligned(const CacheAligned<U>&) {}
  T* allocate(std::size_t n) {
    return static_cast<T*>(::operator new(n * sizeof(T), kAlign));
  }
  void deallocate(T* p, std::size_t) { ::operator delete(p, kAlign); }
  friend bool operator==(const CacheAligned&, const CacheAligned&) {
    return true;
  }
};
using Floats = std::vector<float, CacheAligned<float>>;

struct Item {
  int segment = 0;
  int kv_head = 0;
  int t0 = 0;  // query rows [t0, t1) of the segment, local
  int t1 = 0;
};

// The plain kernel, for CPUs without AVX-512 (hosted CI): per query vector,
// a dot product per key and a scaled add per value, softmax online over
// blocks of kKeyBlock keys.
void AttendPlain(const Config& cfg, const Item& item, const float* base,
                 float* out_base) {
  const int hd = cfg.head_dim;
  const int group = cfg.heads / cfg.kv_heads;
  const int q_dim = cfg.QDim();
  const int kv_dim = cfg.KvDim();
  const auto stride = static_cast<std::size_t>(cfg.InRows());
  const int n = (item.t1 - item.t0) * group;  // query vectors
  thread_local std::vector<float> o, m, sum, p;
  o.assign(static_cast<std::size_t>(n) * hd, 0.0f);
  m.assign(static_cast<std::size_t>(n), -INFINITY);
  sum.assign(static_cast<std::size_t>(n), 0.0f);
  p.resize(kKeyBlock);
  for (int jb = 0; jb < item.t1; jb += kKeyBlock) {
    for (int t = item.t0; t < item.t1; ++t) {
      const int limit = std::min(t + 1, jb + kKeyBlock);  // causal
      if (limit <= jb) {
        continue;
      }
      const int count = limit - jb;
      for (int r = 0; r < group; ++r) {
        const int qh = item.kv_head * group + r;
        const int v = (t - item.t0) * group + r;
        const float* q = base + static_cast<std::size_t>(t) * stride +
                         static_cast<std::size_t>(qh) * hd;
        float mx = m[v];
        for (int j = 0; j < count; ++j) {
          const float* key = base + static_cast<std::size_t>(jb + j) * stride +
                             q_dim + item.kv_head * hd;
          p[j] = Dot(q, key, hd);
          mx = std::max(mx, p[j]);
        }
        const float correction = std::exp(m[v] - mx);
        float* ov = o.data() + static_cast<std::size_t>(v) * hd;
        if (correction != 1.0f) {
          for (int e = 0; e < hd; ++e) {
            ov[e] *= correction;
          }
        }
        float total = 0;
        for (int j = 0; j < count; ++j) {
          const float w = std::exp(p[j] - mx);
          total += w;
          const float* value = base +
                               static_cast<std::size_t>(jb + j) * stride +
                               q_dim + kv_dim + item.kv_head * hd;
          AddScaled(w, value, ov, hd);
        }
        sum[v] = sum[v] * correction + total;
        m[v] = mx;
      }
    }
  }
  for (int t = item.t0; t < item.t1; ++t) {
    for (int r = 0; r < group; ++r) {
      const int qh = item.kv_head * group + r;
      const int v = (t - item.t0) * group + r;
      const float inv = 1.0f / sum[v];
      const float* ov = o.data() + static_cast<std::size_t>(v) * hd;
      float* dst = out_base + static_cast<std::size_t>(t) * q_dim +
                   static_cast<std::size_t>(qh) * hd;
      for (int e = 0; e < hd; ++e) {
        dst[e] = ov[e] * inv;
      }
    }
  }
}

#if defined(__x86_64__) && defined(__GNUC__)
// Zen 5 runs AVX-512 at full width; the build targets AVX2, so these
// kernels are compiled for AVX-512F alone and picked at run time.
#define GUFO_QWEN3VL_AVX512 1

// s[i * kKeyBlock + j] = q_i . key_j for query vectors i in [0, n) (rows hd
// apart, n a multiple of kTile) and a block's keys, kt holding them
// transposed (kt[d * kKeyBlock + j], 64-byte aligned): keys run along the
// lanes, so no dot product needs a horizontal sum.
__attribute__((target("avx512f"))) void ScoresAvx512(const float* q, int hd,
                                                     const float* kt, int n,
                                                     float* s) {
  static_assert(kKeyBlock == 64, "four 16-lane vectors per key block");
  for (int i = 0; i < n; i += kTile) {
    const float* qi = q + static_cast<std::size_t>(i) * hd;
    float* si = s + static_cast<std::size_t>(i) * kKeyBlock;
    __m512 acc[kTile][4];
    for (auto& a : acc) {
      for (auto& x : a) {
        x = _mm512_setzero_ps();
      }
    }
    for (int d = 0; d < hd; ++d) {
      const float* row = kt + static_cast<std::size_t>(d) * kKeyBlock;
      const __m512 k0 = _mm512_load_ps(row);
      const __m512 k1 = _mm512_load_ps(row + 16);
      const __m512 k2 = _mm512_load_ps(row + 32);
      const __m512 k3 = _mm512_load_ps(row + 48);
      for (int r = 0; r < kTile; ++r) {
        const __m512 x = _mm512_set1_ps(qi[r * hd + d]);
        acc[r][0] = _mm512_fmadd_ps(x, k0, acc[r][0]);
        acc[r][1] = _mm512_fmadd_ps(x, k1, acc[r][1]);
        acc[r][2] = _mm512_fmadd_ps(x, k2, acc[r][2]);
        acc[r][3] = _mm512_fmadd_ps(x, k3, acc[r][3]);
      }
    }
    for (int r = 0; r < kTile; ++r) {
      for (int w = 0; w < 4; ++w) {
        _mm512_storeu_ps(si + r * kKeyBlock + 16 * w, acc[r][w]);
      }
    }
  }
}

// Columns [e, e + 16 W) of one tile of ValuesAvx512.
template<int W>
__attribute__((target("avx512f"))) void ValuesTileAvx512(
    const float* p, const float* corr, const float* v, std::size_t stride,
    int count, int hd, float* o, int e) {
  __m512 acc[kTile][W];
  for (int r = 0; r < kTile; ++r) {
    const __m512 c = _mm512_set1_ps(corr[r]);
    for (int w = 0; w < W; ++w) {
      acc[r][w] = _mm512_mul_ps(_mm512_loadu_ps(o + r * hd + e + 16 * w), c);
    }
  }
  for (int j = 0; j < count; ++j) {
    const float* vj = v + static_cast<std::size_t>(j) * stride + e;
    __m512 x[W];
    for (int w = 0; w < W; ++w) {
      x[w] = _mm512_loadu_ps(vj + 16 * w);
    }
    for (int r = 0; r < kTile; ++r) {
      const __m512 pr = _mm512_set1_ps(p[r * kKeyBlock + j]);
      for (int w = 0; w < W; ++w) {
        acc[r][w] = _mm512_fmadd_ps(pr, x[w], acc[r][w]);
      }
    }
  }
  for (int r = 0; r < kTile; ++r) {
    for (int w = 0; w < W; ++w) {
      _mm512_storeu_ps(o + r * hd + e + 16 * w, acc[r][w]);
    }
  }
}

// o_i = o_i * corr[i] + sum_{j < count} p[i * kKeyBlock + j] value_j for
// query vectors i in [0, n) (n a multiple of kTile, o rows hd apart, values
// `stride` floats apart).
__attribute__((target("avx512f"))) void ValuesAvx512(
    const float* p, const float* corr, const float* v, std::size_t stride,
    int count, int hd, int n, float* o) {
  for (int i = 0; i < n; i += kTile) {
    const float* pi = p + static_cast<std::size_t>(i) * kKeyBlock;
    const float* ci = corr + i;
    float* oi = o + static_cast<std::size_t>(i) * hd;
    int e = 0;
    for (; e + 64 <= hd; e += 64) {
      ValuesTileAvx512<4>(pi, ci, v, stride, count, hd, oi, e);
    }
    for (; e + 16 <= hd; e += 16) {
      ValuesTileAvx512<1>(pi, ci, v, stride, count, hd, oi, e);
    }
    for (; e < hd; ++e) {
      for (int r = 0; r < kTile; ++r) {
        float acc = oi[r * hd + e] * ci[r];
        for (int j = 0; j < count; ++j) {
          acc += pi[r * kKeyBlock + j] *
                 v[static_cast<std::size_t>(j) * stride + e];
        }
        oi[r * hd + e] = acc;
      }
    }
  }
}

// The fast kernel: per key block, scores as a register-blocked product
// against the transposed keys (`tiles` holds the item's kv head's tiles,
// one per key block, `tile_step` floats apart), the online-softmax update
// per query vector, then the weighted values.
void AttendAvx512(const Config& cfg, const Item& item, const float* base,
                  const float* tiles, std::size_t tile_step, float* out_base) {
  const int hd = cfg.head_dim;
  const int group = cfg.heads / cfg.kv_heads;
  const int q_dim = cfg.QDim();
  const auto stride = static_cast<std::size_t>(cfg.InRows());
  // Query vectors i = (t - t0) * group + r, padded to whole tiles with
  // zero queries that are never written out.
  const int n = (item.t1 - item.t0) * group;
  const int padded = (n + kTile - 1) / kTile * kTile;
  thread_local Floats q, s, o;
  thread_local std::vector<float> m, sum, corr;
  q.assign(static_cast<std::size_t>(padded) * hd, 0.0f);
  s.resize(static_cast<std::size_t>(padded) * kKeyBlock);
  o.assign(static_cast<std::size_t>(padded) * hd, 0.0f);
  m.assign(static_cast<std::size_t>(padded), -INFINITY);
  sum.assign(static_cast<std::size_t>(padded), 0.0f);
  corr.resize(static_cast<std::size_t>(padded));
  for (int t = item.t0; t < item.t1; ++t) {
    for (int r = 0; r < group; ++r) {
      const int qh = item.kv_head * group + r;
      const int i = (t - item.t0) * group + r;
      std::copy_n(base + static_cast<std::size_t>(t) * stride +
                      static_cast<std::size_t>(qh) * hd,
                  hd, q.data() + static_cast<std::size_t>(i) * hd);
    }
  }
  const float* values = base + q_dim + cfg.KvDim() + item.kv_head * hd;
  for (int jb = 0, b = 0; jb < item.t1; jb += kKeyBlock, ++b) {
    ScoresAvx512(q.data(), hd, tiles + b * tile_step, padded, s.data());
    for (int i = 0; i < padded; ++i) {
      // Causal: a row's keys in the block are a prefix of it.
      const int t = i < n ? item.t0 + i / group : item.t1 - 1;
      const int count = std::min(kKeyBlock, t - jb + 1);
      float* p = s.data() + static_cast<std::size_t>(i) * kKeyBlock;
      if (count <= 0) {
        std::fill_n(p, kKeyBlock, 0.0f);
        corr[i] = 1.0f;
        continue;
      }
      float mx = m[i];
      for (int j = 0; j < count; ++j) {
        mx = std::max(mx, p[j]);
      }
      corr[i] = std::exp(m[i] - mx);
      const float total = simd::ExpShiftSum(p, mx, p, count);
      std::fill(p + count, p + kKeyBlock, 0.0f);
      sum[i] = sum[i] * corr[i] + total;
      m[i] = mx;
    }
    ValuesAvx512(s.data(), corr.data(),
                 values + static_cast<std::size_t>(jb) * stride, stride,
                 std::min(kKeyBlock, item.t1 - jb), hd, padded, o.data());
  }
  for (int t = item.t0; t < item.t1; ++t) {
    for (int r = 0; r < group; ++r) {
      const int qh = item.kv_head * group + r;
      const int i = (t - item.t0) * group + r;
      const float inv = 1.0f / sum[i];
      const float* ov = o.data() + static_cast<std::size_t>(i) * hd;
      float* dst = out_base + static_cast<std::size_t>(t) * q_dim +
                   static_cast<std::size_t>(qh) * hd;
      for (int e = 0; e < hd; ++e) {
        dst[e] = ov[e] * inv;
      }
    }
  }
}
#endif

}  // namespace

std::vector<Segment> Pack(const std::vector<std::vector<int>>& inputs,
                          int vocab) {
  std::vector<Segment> segments;
  segments.reserve(inputs.size());
  int rows = 0;
  for (const auto& ids : inputs) {
    if (ids.empty()) {
      throw std::invalid_argument("qwen3vl: empty input");
    }
    for (const int id : ids) {
      if (id < 0 || id >= vocab) {
        throw std::invalid_argument("qwen3vl: token id out of range");
      }
    }
    segments.push_back({rows, static_cast<int>(ids.size())});
    rows += static_cast<int>(ids.size());
  }
  return segments;
}

RopeTable::RopeTable(int head_dim, float theta) : half_(head_dim / 2) {
  inv_freq_.resize(half_);
  for (int i = 0; i < half_; ++i) {
    // HF: 1.0 / (base ** (arange(0, dim, 2, dtype=int64).float() / dim)).
    inv_freq_[i] = static_cast<float>(
        1.0 / std::pow(static_cast<double>(theta),
                       static_cast<double>(2 * i) / (2.0 * half_)));
  }
}

void RopeTable::Reserve(int count) {
  if (count <= count_) {
    return;
  }
  cos_.resize(static_cast<std::size_t>(count) * half_);
  sin_.resize(cos_.size());
  for (int p = count_; p < count; ++p) {
    for (int i = 0; i < half_; ++i) {
      // fp32 angle, as HF's position_ids.float() @ inv_freq.
      const float angle = static_cast<float>(p) * inv_freq_[i];
      cos_[static_cast<std::size_t>(p) * half_ + i] = std::cos(angle);
      sin_[static_cast<std::size_t>(p) * half_ + i] = std::sin(angle);
    }
  }
  count_ = count;
}

void RopeTable::Apply(float* x, int position) const {
  const float* cs = cos_.data() + static_cast<std::size_t>(position) * half_;
  const float* sn = sin_.data() + static_cast<std::size_t>(position) * half_;
  for (int i = 0; i < half_; ++i) {
    const float a = x[i];
    const float b = x[i + half_];
    x[i] = a * cs[i] - b * sn[i];
    x[i + half_] = b * cs[i] + a * sn[i];
  }
}

void Attention(const Config& cfg, const RopeTable& rope, const float* q_norm,
               const float* k_norm, const std::vector<Segment>& segments,
               float* in, float* out, const ParallelRun& run) {
  detail::Attention(cfg, rope, q_norm, k_norm, segments, in, out, run,
                    detail::HasAvx512());
}

namespace detail {

bool HasAvx512() {
#ifdef GUFO_QWEN3VL_AVX512
  static const bool has = __builtin_cpu_supports("avx512f");
  return has;
#else
  return false;
#endif
}

void Attention(const Config& cfg, const RopeTable& rope, const float* q_norm,
               const float* k_norm, const std::vector<Segment>& segments,
               float* in, float* out, const ParallelRun& run, bool avx512) {
  if (avx512 && !HasAvx512()) {
    throw std::invalid_argument("qwen3vl: the CPU has no AVX-512");
  }
  const int hd = cfg.head_dim;
  const int heads = cfg.heads;
  const int kv_heads = cfg.kv_heads;
  const int q_dim = cfg.QDim();
  const auto stride = static_cast<std::size_t>(cfg.InRows());
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  // Per row: normalize, rotate (and scale q) in place.
  std::vector<int> row_position;
  int rows = 0;
  for (const Segment& s : segments) {
    rows = std::max(rows, s.begin + s.length);
  }
  row_position.assign(static_cast<std::size_t>(rows), 0);
  for (const Segment& s : segments) {
    for (int t = 0; t < s.length; ++t) {
      row_position[static_cast<std::size_t>(s.begin + t)] = t;
    }
  }
  run(static_cast<std::size_t>(rows), [&](std::size_t r) {
    float* row = in + r * stride;
    const int position = row_position[r];
    for (int h = 0; h < heads; ++h) {
      float* q = row + h * hd;
      RmsNorm(q, q_norm, hd, cfg.rms_eps, q);
      rope.Apply(q, position);
      for (int e = 0; e < hd; ++e) {
        q[e] *= scale;
      }
    }
    for (int h = 0; h < kv_heads; ++h) {
      float* k = row + q_dim + h * hd;
      RmsNorm(k, k_norm, hd, cfg.rms_eps, k);
      rope.Apply(k, position);
    }
  });

  // Work items: (segment, kv head, block of query rows), each with that kv
  // head's query heads, softmax online over key blocks; the longest first,
  // so a pool that hands out items in order finishes evenly.
  std::vector<Item> items;
  for (int s = 0; s < static_cast<int>(segments.size()); ++s) {
    for (int t0 = 0; t0 < segments[s].length; t0 += kQueryBlock) {
      for (int g = 0; g < kv_heads; ++g) {
        items.push_back(
            {s, g, t0, std::min(segments[s].length, t0 + kQueryBlock)});
      }
    }
  }
  std::stable_sort(items.begin(), items.end(),
                   [](const Item& a, const Item& b) { return a.t1 > b.t1; });

  if (!avx512) {
    run(items.size(), [&](std::size_t index) {
      const Item& item = items[index];
      const Segment& seg = segments[static_cast<std::size_t>(item.segment)];
      AttendPlain(cfg, item, in + static_cast<std::size_t>(seg.begin) * stride,
                  out + static_cast<std::size_t>(seg.begin) * q_dim);
    });
    return;
  }

#ifdef GUFO_QWEN3VL_AVX512
  // Keys transposed: one tile (hd x kKeyBlock, zero past the segment's
  // end) per key block and kv head, tile (s, b, g) at
  // ((first[s] + b) * kv_heads + g) * tile. Kept by the calling thread
  // between layers; workers reach it through `tiles`.
  const std::size_t tile = static_cast<std::size_t>(hd) * kKeyBlock;
  std::vector<std::size_t> first(segments.size());
  std::vector<std::pair<int, int>> blocks;  // (segment, key block)
  for (std::size_t s = 0; s < segments.size(); ++s) {
    first[s] = blocks.size();
    for (int b = 0; b * kKeyBlock < segments[s].length; ++b) {
      blocks.emplace_back(static_cast<int>(s), b);
    }
  }
  thread_local Floats key_tiles;
  if (key_tiles.size() < blocks.size() * kv_heads * tile) {
    key_tiles.resize(blocks.size() * kv_heads * tile);
  }
  float* const tiles = key_tiles.data();
  run(blocks.size(), [&](std::size_t index) {
    const Segment& seg =
        segments[static_cast<std::size_t>(blocks[index].first)];
    const int jb = blocks[index].second * kKeyBlock;
    for (int g = 0; g < kv_heads; ++g) {
      float* dst = tiles + (index * kv_heads + g) * tile;
      for (int j = 0; j < kKeyBlock; ++j) {
        const float* key =
            jb + j < seg.length
                ? in + static_cast<std::size_t>(seg.begin + jb + j) * stride +
                      q_dim + g * hd
                : nullptr;
        for (int d = 0; d < hd; ++d) {
          dst[static_cast<std::size_t>(d) * kKeyBlock + j] =
              key != nullptr ? key[d] : 0.0f;
        }
      }
    }
  });
  run(items.size(), [&](std::size_t index) {
    const Item& item = items[index];
    const auto s = static_cast<std::size_t>(item.segment);
    const Segment& seg = segments[s];
    AttendAvx512(cfg, item, in + static_cast<std::size_t>(seg.begin) * stride,
                 tiles + (first[s] * kv_heads + item.kv_head) * tile,
                 kv_heads * tile,
                 out + static_cast<std::size_t>(seg.begin) * q_dim);
  });
#endif
}

}  // namespace detail

std::vector<std::vector<float>> Pool(const Config& cfg,
                                     const float* output_norm,
                                     const std::vector<Segment>& segments,
                                     const float* h) {
  const int d = cfg.hidden;
  std::vector<std::vector<float>> out;
  out.reserve(segments.size());
  for (const Segment& s : segments) {
    if (s.length <= 0) {
      throw std::invalid_argument("qwen3vl: empty input");
    }
    std::vector<float> e(static_cast<std::size_t>(d));
    const float* last =
        h + static_cast<std::size_t>(s.begin + s.length - 1) * d;
    RmsNorm(last, output_norm, d, cfg.rms_eps, e.data());
    double norm = 0;
    for (const float x : e) {
      norm += static_cast<double>(x) * x;
    }
    const auto inv = static_cast<float>(1.0 / std::max(std::sqrt(norm), 1e-12));
    for (float& x : e) {
      x *= inv;
    }
    out.push_back(std::move(e));
  }
  return out;
}

void Truncate(std::vector<float>& embedding, int dimensions) {
  if (dimensions <= 0 ||
      static_cast<std::size_t>(dimensions) >= embedding.size()) {
    return;
  }
  embedding.resize(static_cast<std::size_t>(dimensions));
  double norm = 0;
  for (const float x : embedding) {
    norm += static_cast<double>(x) * x;
  }
  const auto inv = static_cast<float>(1.0 / std::max(std::sqrt(norm), 1e-12));
  for (float& x : embedding) {
    x *= inv;
  }
}

}  // namespace gufo_npu::qwen3vl_embed_2b
