#include "models/qwen3vl_embed_2b/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "core/vec.hpp"

namespace gufo_npu::qwen3vl_embed_2b {
namespace {

using vec::Dot;
using vec::RmsNorm;

// Attention work items: query rows per item, keys per online-softmax block.
constexpr int kQueryBlock = 16;
constexpr int kKeyBlock = 64;

// y += s * x over n values.
void AddScaled(float s, const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] += s * x[i];
  }
}

struct Item {
  int segment = 0;
  int kv_head = 0;
  int t0 = 0;  // query rows [t0, t1) of the segment, local
  int t1 = 0;
};

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
  const int hd = cfg.head_dim;
  const int heads = cfg.heads;
  const int group = heads / cfg.kv_heads;
  const int q_dim = cfg.QDim();
  const int kv_dim = cfg.KvDim();
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
    for (int h = 0; h < cfg.kv_heads; ++h) {
      float* k = row + q_dim + h * hd;
      RmsNorm(k, k_norm, hd, cfg.rms_eps, k);
      rope.Apply(k, position);
    }
  });

  // Work items: (segment, kv head, block of query rows), each with that kv
  // head's query heads, softmax online over blocks of keys.
  std::vector<Item> items;
  for (int s = 0; s < static_cast<int>(segments.size()); ++s) {
    for (int t0 = 0; t0 < segments[s].length; t0 += kQueryBlock) {
      for (int g = 0; g < cfg.kv_heads; ++g) {
        items.push_back(
            {s, g, t0, std::min(segments[s].length, t0 + kQueryBlock)});
      }
    }
  }
  run(items.size(), [&](std::size_t index) {
    const Item& item = items[index];
    const Segment& seg = segments[static_cast<std::size_t>(item.segment)];
    const int n = (item.t1 - item.t0) * group;  // query vectors
    thread_local std::vector<float> o, m, sum, p;
    o.assign(static_cast<std::size_t>(n) * hd, 0.0f);
    m.assign(static_cast<std::size_t>(n), -INFINITY);
    sum.assign(static_cast<std::size_t>(n), 0.0f);
    p.resize(kKeyBlock);
    const float* base = in + static_cast<std::size_t>(seg.begin) * stride;
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
            const float* key = base +
                               static_cast<std::size_t>(jb + j) * stride +
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
        float* dst = out + static_cast<std::size_t>(seg.begin + t) * q_dim +
                     static_cast<std::size_t>(qh) * hd;
        for (int e = 0; e < hd; ++e) {
          dst[e] = ov[e] * inv;
        }
      }
    }
  });
}

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
