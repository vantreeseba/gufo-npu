// What converting Qwen3.6-35B-A3B's weights to q4nx costs, on held-out text,
// and which way of choosing the 4-bit values costs least (same format, same
// kernel). The fp32 CPU decoder runs teacher forced over the evaluation texts
// (tools/reference/fetch_text.py, >= 1,000 positions) and the oracle prompts
// with three weight sets, each against (a), the GGUF as stored:
//   (d) every non-expert matrix (Q8_0 in the GGUF) through q4nx, LM head as
//       stored;
//   (b) (d) plus the LM head;
//   (c) (b) plus the routed experts: Q4_K keeps its codes (d and dmin round
//       to bf16), Q5_K/Q6_K requantize.
// Reported per set: KL(a || set) mean and top-1 agreement with (a) on the
// evaluation text, and on the oracle prompts top-1 against (a) and against
// llama.cpp. Weights in q4nx are kept as codes plus each group's step and
// minimum and multiplied in fp32.
//
// Methods (--methods, comma separated; default `library`):
//   library         q4nx::PackRows, what the NPU decoder loads; the only
//                   method with pass thresholds
//   minmax          per 32 weights a min/max fit (the packer before
//                   2026-10-27)
//   search          llama.cpp's Q4_K fit (weights av + |w|), the library's
//                   algorithm (it differs only in rounding ties and sum order)
//   search-uniform  the same fit with equal weights
//   search-refine   search, then refit d and dmin by least squares
//   imatrix         search weighted by the mean x^2 of each input from a
//                   calibration pass over <text>/calib (llama.cpp's
//                   qw * sqrt(sigma2 + w^2))
//   imatrix-plain   weighted by the mean x^2 alone
//   imatrix-file    imatrix, with the mean x^2 from a llama.cpp imatrix GGUF
//                   (--imatrix); the LM head, which it lacks, from the
//                   calibration pass
//   awq-search, awq-imatrix
//                   per-input scales s = mean|x|^alpha (powers of two, alpha
//                   chosen per matrix on calibration samples) folded into the
//                   weights and divided out of the input, on the matrices
//                   whose input scale the NPU path can absorb: the mixers'
//                   input and output projections, the shared expert's down
//                   projection and the LM head
// --sets (default d,b,c) picks the weight sets; --cache <dir> keeps the (a)
// logits and the calibration statistics between runs.
//   GUFO_NPU_QWEN36_MODEL   the Qwen3.6-35B-A3B GGUF
//   GUFO_NPU_QWEN36_ORACLE  oracle output directory
//   GUFO_NPU_QWEN36_TEXT    fetch_text.py output (eval/ and calib/)
// Exits 77 (skip) when any is unset.
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "core/cpus.hpp"
#include "core/gguf.hpp"
#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "core/thread_pool.hpp"
#include "core/tokenizer.hpp"
#include "core/vec.hpp"
#include "models/qwen35/config.hpp"
#include "models/qwen35/decoder.hpp"
#include "models/qwen35/gguf_projections.hpp"

namespace {

namespace fs = std::filesystem;
namespace q35 = gufo_npu::qwen35;
namespace quant = gufo_npu::quant;
namespace vec = gufo_npu::vec;
using gufo_npu::gguf::GgmlType;
using gufo_npu::gguf::TensorInfo;
using q35::Stage;

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Workers sleep as soon as a run ends, so this pool's ~250 runs per token
// don't spin on CPUs the decoder's own (spinning) pools need.
class BlockingPool {
public:
  explicit BlockingPool(std::size_t workers) {
    for (std::size_t i = 0; i < workers; ++i) {
      threads_.emplace_back([this] { Loop(); });
    }
  }
  ~BlockingPool() {
    {
      std::lock_guard<std::mutex> hold(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) {
      t.join();
    }
  }
  void Run(std::size_t n, const std::function<void(std::size_t)>& fn) {
    {
      std::lock_guard<std::mutex> hold(mutex_);
      fn_ = &fn;
      n_ = n;
      next_.store(0);
      active_ = threads_.size();
      ++generation_;
    }
    wake_.notify_all();
    Drain();
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] { return active_ == 0; });
  }

private:
  void Drain() {
    for (std::size_t i = next_.fetch_add(1); i < n_; i = next_.fetch_add(1)) {
      (*fn_)(i);
    }
  }
  void Loop() {
    std::size_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) {
        return;
      }
      seen = generation_;
      lock.unlock();
      Drain();
      lock.lock();
      if (--active_ == 0) {
        done_.notify_one();
      }
    }
  }
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable wake_, done_;
  const std::function<void(std::size_t)>* fn_ = nullptr;
  std::size_t n_ = 0;
  std::atomic<std::size_t> next_{0};
  std::size_t active_ = 0;
  std::size_t generation_ = 0;
  bool stop_ = false;
};

BlockingPool& Pool() {
  static BlockingPool pool(gufo_npu::HalfCpuWorkers());
  return pool;
}
std::mutex pool_mutex;

void ParallelRows(std::size_t n, std::size_t chunk,
                  const std::function<void(std::size_t, std::size_t)>& fn) {
  std::lock_guard<std::mutex> hold(pool_mutex);
  Pool().Run((n + chunk - 1) / chunk, [&](std::size_t job) {
    fn(job * chunk, std::min(n, (job + 1) * chunk));
  });
}

// ---------------------------------------------------------------- quantizers

constexpr int kGroups = 8;
struct QBlock {
  std::uint16_t d = 0, dmin = 0;
  std::uint8_t sc[kGroups] = {};
  std::int8_t mn[kGroups] = {};
  std::uint8_t q[256] = {};
};

float Bf(std::uint16_t bits) {
  return quant::Bf16ToFloat(bits);
}

std::uint16_t Bf16AtLeast(float x) {
  std::uint16_t bits = quant::FloatToBf16(x);
  if (quant::Bf16ToFloat(bits) < x) {
    ++bits;
  }
  return bits;
}

enum class Weighting { kUniform, kLlama, kImatrix, kImatrixPlain };
struct Method {
  std::string name;
  bool minmax = false;
  Weighting weighting = Weighting::kLlama;
  bool refine = false;
  bool awq = false;
  bool library = false;    // q4nx::PackRows itself
  bool from_file = false;  // importance from --imatrix
};

// The packer before 2026-10-27 (a min/max fit), reproduced.
void QuantMinMax(const float* w, QBlock& b) {
  float lo[kGroups];
  float hi[kGroups];
  float max_abs_min = 0;
  for (int g = 0; g < kGroups; ++g) {
    const auto [lo_it, hi_it] =
        std::minmax_element(w + g * 32, w + g * 32 + 32);
    lo[g] = *lo_it;
    hi[g] = *hi_it;
    max_abs_min = std::max(max_abs_min, std::abs(lo[g]));
  }
  b.dmin = Bf16AtLeast(max_abs_min / 127.0f);
  const float dmin = Bf(b.dmin);
  float need[kGroups];
  float max_need = 0;
  for (int g = 0; g < kGroups; ++g) {
    const float q = dmin > 0 ? std::floor(lo[g] / dmin) : 0.0f;
    b.mn[g] = static_cast<std::int8_t>(std::clamp(q, -127.0f, 127.0f));
    need[g] = (hi[g] - dmin * static_cast<float>(b.mn[g])) / 15.0f;
    max_need = std::max(max_need, need[g]);
  }
  b.d = Bf16AtLeast(max_need / 255.0f);
  const float d = Bf(b.d);
  for (int g = 0; g < kGroups; ++g) {
    const float sc = d > 0 ? std::ceil(need[g] / d) : 0.0f;
    b.sc[g] = static_cast<std::uint8_t>(std::clamp(sc, 0.0f, 255.0f));
    const float step = d * static_cast<float>(b.sc[g]);
    const float min = dmin * static_cast<float>(b.mn[g]);
    for (int l = 0; l < 32; ++l) {
      float code = step > 0 ? std::nearbyint((w[g * 32 + l] - min) / step) : 0;
      b.q[g * 32 + l] =
          static_cast<std::uint8_t>(std::clamp(code, 0.0f, 15.0f));
    }
  }
}

// Nearest codes for one group under (step, min); returns the weighted SSE.
float GroupCodes(const float* x, const float* w, float step, float min,
                 std::uint8_t* q) {
  float sse = 0;
  const float inv = step > 0 ? 1.0f / step : 0.0f;
  for (int l = 0; l < 32; ++l) {
    float c = step > 0 ? std::nearbyint((x[l] - min) * inv) : 0.0f;
    c = std::clamp(c, 0.0f, 15.0f);
    q[l] = static_cast<std::uint8_t>(c);
    const float e = min + step * c - x[l];
    sse += w[l] * e * e;
  }
  return sse;
}

// llama.cpp's make_qkx3_quants (rmin -0.9, rdelta 0.05, 36 steps) without
// its min <= 0 clamp (q4nx's sub-minimum is signed): the weighted
// least-squares (scale, min) over candidate code assignments.
void FitGroup(const float* x, const float* w, float* scale_out,
              float* min_out) {
  float lo = x[0], hi = x[0], sum_w = 0, sum_x = 0;
  for (int i = 0; i < 32; ++i) {
    lo = std::min(lo, x[i]);
    hi = std::max(hi, x[i]);
    sum_w += w[i];
    sum_x += w[i] * x[i];
  }
  if (hi <= lo) {
    *scale_out = 0;
    *min_out = lo;
    return;
  }
  float best_scale = (hi - lo) / 15.0f;
  float best_min = lo;
  float best = 0;
  for (int i = 0; i < 32; ++i) {
    const float l =
        std::clamp(std::nearbyint((x[i] - lo) / best_scale), 0.0f, 15.0f);
    const float e = best_scale * l + lo - x[i];
    best += w[i] * e * e;
  }
  float laux[32];
  for (int is = 0; is <= 36; ++is) {
    const float iscale = (-0.9f + 0.05f * is + 15.0f) / (hi - lo);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < 32; ++i) {
      const float l =
          std::clamp(std::nearbyint(iscale * (x[i] - lo)), 0.0f, 15.0f);
      laux[i] = l;
      sum_l += w[i] * l;
      sum_l2 += w[i] * l * l;
      sum_xl += w[i] * l * x[i];
    }
    const float det = sum_w * sum_l2 - sum_l * sum_l;
    if (det <= 0) {
      continue;
    }
    const float scale = (sum_w * sum_xl - sum_x * sum_l) / det;
    const float min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
    if (scale <= 0) {
      continue;
    }
    float sse = 0;
    for (int i = 0; i < 32; ++i) {
      const float e = scale * laux[i] + min - x[i];
      sse += w[i] * e * e;
    }
    if (sse < best) {
      best = sse;
      best_scale = scale;
      best_min = min;
    }
  }
  *scale_out = best_scale;
  *min_out = best_min;
}

// Per group, the integer sub-scale and sub-minimum near (scale, min) under
// (d, dmin) with the lowest weighted SSE; returns the block's SSE.
float ChooseSubs(const float* x, const float* w, float d, float dmin,
                 const float* scale, const float* mins, QBlock& b) {
  float total = 0;
  std::uint8_t q[32];
  for (int g = 0; g < kGroups; ++g) {
    const float sc0 = d > 0 ? std::nearbyint(scale[g] / d) : 0.0f;
    const float mn0 = dmin > 0 ? std::nearbyint(mins[g] / dmin) : 0.0f;
    float best = INFINITY;
    for (int dsc = -1; dsc <= 1; ++dsc) {
      const float sc = std::clamp(sc0 + dsc, 0.0f, 255.0f);
      if (d == 0 && sc != 0) {
        continue;
      }
      for (int dmn = -1; dmn <= 1; ++dmn) {
        const float mn = std::clamp(mn0 + dmn, -127.0f, 127.0f);
        if (dmin == 0 && mn != 0) {
          continue;
        }
        const float step = d * sc;
        const float min = dmin * mn;
        const float sse = GroupCodes(x + g * 32, w + g * 32, step, min, q);
        if (sse < best) {
          best = sse;
          b.sc[g] = static_cast<std::uint8_t>(sc);
          b.mn[g] = static_cast<std::int8_t>(mn);
          std::memcpy(b.q + g * 32, q, 32);
        }
      }
    }
    total += best;
  }
  return total;
}

// `imp`: per-input importance (mean x^2) for these 256 inputs, or null.
void QuantSearch(const float* x, const float* imp, const Method& m, QBlock& b) {
  float w[256];
  float sum_x2 = 0;
  for (int k = 0; k < 256; ++k) {
    sum_x2 += x[k] * x[k];
  }
  const float sigma2 = 2 * sum_x2 / 256;
  const float av_x = std::sqrt(sigma2);
  for (int k = 0; k < 256; ++k) {
    switch (m.weighting) {
      case Weighting::kUniform:
        w[k] = 1;
        break;
      case Weighting::kLlama:
        w[k] = av_x + std::abs(x[k]);
        break;
      case Weighting::kImatrix:
        w[k] = imp != nullptr ? imp[k] * std::sqrt(sigma2 + x[k] * x[k])
                              : av_x + std::abs(x[k]);
        break;
      case Weighting::kImatrixPlain:
        w[k] = imp != nullptr ? imp[k] : 1.0f;
        break;
    }
    w[k] = std::max(w[k], 1e-30f);
  }
  float scale[kGroups];
  float mins[kGroups];
  float max_scale = 0;
  float max_abs_min = 0;
  for (int g = 0; g < kGroups; ++g) {
    FitGroup(x + g * 32, w + g * 32, &scale[g], &mins[g]);
    max_scale = std::max(max_scale, scale[g]);
    max_abs_min = std::max(max_abs_min, std::abs(mins[g]));
  }
  b.d = Bf16AtLeast(max_scale / 255.0f);
  b.dmin = Bf16AtLeast(max_abs_min / 127.0f);
  float sse = ChooseSubs(x, w, Bf(b.d), Bf(b.dmin), scale, mins, b);
  if (!m.refine) {
    return;
  }
  for (int iter = 0; iter < 2; ++iter) {
    // (d, dmin) by weighted least squares with the integers fixed.
    double a2 = 0, ac = 0, c2 = 0, ax = 0, cx = 0;
    for (int g = 0; g < kGroups; ++g) {
      for (int l = 0; l < 32; ++l) {
        const int k = g * 32 + l;
        const double a = static_cast<double>(b.sc[g]) * b.q[k];
        const double c = b.mn[g];
        a2 += w[k] * a * a;
        ac += w[k] * a * c;
        c2 += w[k] * c * c;
        ax += w[k] * a * x[k];
        cx += w[k] * c * x[k];
      }
    }
    const double det = a2 * c2 - ac * ac;
    if (det <= 0) {
      break;
    }
    const double d = (ax * c2 - cx * ac) / det;
    const double dmin = (a2 * cx - ac * ax) / det;
    if (d <= 0 || dmin < 0) {
      break;
    }
    // Per group, refit (scale, min) to the current codes, then choose the
    // integers again under the new super values.
    QBlock t = b;
    t.d = quant::FloatToBf16(static_cast<float>(d));
    t.dmin = quant::FloatToBf16(static_cast<float>(dmin));
    for (int g = 0; g < kGroups; ++g) {
      float sw = 0, sl = 0, sl2 = 0, sx = 0, sxl = 0;
      for (int l = 0; l < 32; ++l) {
        const int k = g * 32 + l;
        const float q = b.q[k];
        sw += w[k];
        sl += w[k] * q;
        sl2 += w[k] * q * q;
        sx += w[k] * x[k];
        sxl += w[k] * q * x[k];
      }
      const float dt = sw * sl2 - sl * sl;
      if (dt > 0) {
        scale[g] = std::max(0.0f, (sw * sxl - sx * sl) / dt);
        mins[g] = (sl2 * sx - sl * sxl) / dt;
      }
    }
    const float tsse = ChooseSubs(x, w, Bf(t.d), Bf(t.dmin), scale, mins, t);
    if (tsse < sse) {
      sse = tsse;
      b = t;
    } else {
      break;
    }
  }
}

void Quantize(const float* x, const float* imp, const Method& m, QBlock& b) {
  if (m.minmax) {
    QuantMinMax(x, b);
  } else {
    QuantSearch(x, imp, m, b);
  }
}

// ------------------------------------------------------ compact q4nx matrix

// The weights q4nx stores, kept as codes plus each group's step and minimum
// (exactly min + step * code), for a fast fp32 product on the CPU.
struct QMat {
  std::size_t rows = 0, dim = 0;
  std::vector<std::uint8_t> q;
  std::vector<float> step, min;
  std::vector<float> inv_s;  // AWQ: the input is multiplied by this first
  std::size_t Bytes() const { return q.size() / 2 + step.size() + min.size(); }
};

struct Part {
  const TensorInfo* t;
  std::size_t first, rows;
};

// A source of fp32 weight rows.
struct RowSource {
  std::size_t rows = 0, dim = 0;
  std::function<void(std::size_t, float*)> row;
  std::vector<Part> parts;
};

RowSource FromParts(std::vector<Part> parts, std::size_t dim) {
  RowSource src;
  src.dim = dim;
  src.parts = parts;
  for (const auto& p : parts) {
    src.rows += p.rows;
  }
  src.row = [parts, dim](std::size_t r, float* out) {
    for (const auto& p : parts) {
      if (r < p.rows) {
        quant::Dequantize(
            p.t->type,
            p.t->data + (p.first + r) * quant::RowBytes(p.t->type, dim), out,
            dim);
        return;
      }
      r -= p.rows;
    }
  };
  return src;
}

// Packs `src` with `m`; `imp` (dim values, mean x^2) and `s` (AWQ scales:
// W diag(s), input x / s) are optional.
QMat Build(const RowSource& src, const Method& m, const std::vector<float>* imp,
           const std::vector<float>* s) {
  QMat out;
  out.rows = src.rows;
  out.dim = src.dim;
  out.q.resize(src.rows * src.dim);
  out.step.resize(src.rows * src.dim / 32);
  out.min.resize(out.step.size());
  std::vector<float> imp_scaled;
  if (s != nullptr) {
    out.inv_s.resize(src.dim);
    for (std::size_t k = 0; k < src.dim; ++k) {
      out.inv_s[k] = 1.0f / (*s)[k];
    }
    if (imp != nullptr) {
      imp_scaled.resize(src.dim);
      for (std::size_t k = 0; k < src.dim; ++k) {
        imp_scaled[k] = (*imp)[k] / ((*s)[k] * (*s)[k]);
      }
    }
  }
  const float* importance =
      imp == nullptr ? nullptr
                     : (s != nullptr ? imp_scaled.data() : imp->data());
  ParallelRows(src.rows, 16, [&](std::size_t begin, std::size_t end) {
    std::vector<float> row(src.dim);
    QBlock b;
    for (std::size_t r = begin; r < end; ++r) {
      src.row(r, row.data());
      if (s != nullptr) {
        for (std::size_t k = 0; k < src.dim; ++k) {
          row[k] *= (*s)[k];
        }
      }
      for (std::size_t kb = 0; kb < src.dim / 256; ++kb) {
        Quantize(row.data() + kb * 256,
                 importance == nullptr ? nullptr : importance + kb * 256, m, b);
        const float d = Bf(b.d);
        const float dmin = Bf(b.dmin);
        for (int g = 0; g < kGroups; ++g) {
          const std::size_t gi = (r * src.dim + kb * 256) / 32 + g;
          out.step[gi] = d * static_cast<float>(b.sc[g]);
          out.min[gi] = dmin * static_cast<float>(b.mn[g]);
        }
        std::memcpy(&out.q[r * src.dim + kb * 256], b.q, 256);
      }
    }
  });
  return out;
}

// The library's own packer (q4nx::PackRows with the model's fit,
// qwen35::Q4nxFit), read back from its blocks.
QMat BuildLibrary(const std::vector<Part>& parts, std::size_t dim,
                  gufo_npu::q4nx::Fit fit) {
  namespace q4nx = gufo_npu::q4nx;
  std::vector<q4nx::Segment> segments;
  for (const auto& p : parts) {
    segments.push_back({p.t->type,
                        p.t->data + p.first * quant::RowBytes(p.t->type, dim),
                        p.rows});
  }
  const q4nx::Matrix packed = q4nx::PackRows(segments, dim, fit);
  QMat out;
  out.rows = packed.rows;
  out.dim = dim;
  out.q.resize(out.rows * dim);
  out.step.resize(out.rows * dim / 32);
  out.min.resize(out.step.size());
  const std::size_t k_blocks = dim / q4nx::kKTile;
  ParallelRows(out.rows, 32, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = begin; r < end; ++r) {
      const std::size_t tile = r / 32, m = r % 32;
      for (std::size_t kb = 0; kb < k_blocks; ++kb) {
        const std::size_t index =
            ((tile / q4nx::kRows) * k_blocks + kb) * q4nx::kRows +
            tile % q4nx::kRows;
        const std::uint8_t* block =
            packed.bytes.data() + index * q4nx::kBlockBytes;
        std::uint16_t bits;
        std::memcpy(&bits, block + m * 2, 2);
        const float d = Bf(bits);
        std::memcpy(&bits, block + 64 + m * 2, 2);
        const float dmin = Bf(bits);
        for (std::size_t g = 0; g < 8; ++g) {
          const std::size_t gi = (r * dim + kb * 256) / 32 + g;
          out.step[gi] = d * static_cast<float>(block[128 + g * 32 + m]);
          out.min[gi] = dmin * static_cast<float>(static_cast<std::int8_t>(
                                   block[384 + g * 32 + m]));
          for (std::size_t l = 0; l < 32; ++l) {
            const std::size_t i = (((m / 16) * 8 + g) * 32 + l) * 16 + m % 16;
            const std::uint8_t byte = block[640 + i / 2];
            out.q[r * dim + kb * 256 + g * 32 + l] =
                (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
          }
        }
      }
    }
  });
  return out;
}

float RowDot(const QMat& m, std::size_t r, const float* x, const float* gsum) {
  const std::uint8_t* q = m.q.data() + r * m.dim;
  const float* step = m.step.data() + r * m.dim / 32;
  const float* min = m.min.data() + r * m.dim / 32;
  __m256 acc = _mm256_setzero_ps();
  float mins = 0;
  for (std::size_t g = 0; g < m.dim / 32; ++g) {
    __m256 a = _mm256_setzero_ps();
    for (int j = 0; j < 4; ++j) {
      const __m128i b =
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + g * 32 + j * 8));
      const __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b));
      a = _mm256_fmadd_ps(v, _mm256_loadu_ps(x + g * 32 + j * 8), a);
    }
    acc = _mm256_fmadd_ps(_mm256_set1_ps(step[g]), a, acc);
    mins += min[g] * gsum[g];
  }
  const __m128 s4 =
      _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  return _mm_cvtss_f32(_mm_add_ss(s2, _mm_movehdup_ps(s2))) + mins;
}

void Gemv(const QMat& m, const float* x_in, float* y) {
  std::vector<float> xs(x_in, x_in + m.dim);
  if (!m.inv_s.empty()) {
    for (std::size_t k = 0; k < m.dim; ++k) {
      xs[k] *= m.inv_s[k];
    }
  }
  std::vector<float> gsum(m.dim / 32);
  for (std::size_t g = 0; g < gsum.size(); ++g) {
    float s = 0;
    for (int l = 0; l < 32; ++l) {
      s += xs[g * 32 + l];
    }
    gsum[g] = s;
  }
  ParallelRows(m.rows, 64, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = begin; r < end; ++r) {
      y[r] = RowDot(m, r, xs.data(), gsum.data());
    }
  });
}

// A Q4_K row as q4nx keeps it (d, dmin rounded to bf16).
void UnpackQ4KAsQ4nx(const std::uint8_t* src, float* dst, std::size_t n) {
  for (std::size_t blk = 0; blk < n / 256; ++blk, src += 144, dst += 256) {
    std::uint16_t bits;
    std::memcpy(&bits, src, 2);
    const float d = Bf(quant::FloatToBf16(quant::HalfToFloat(bits)));
    std::memcpy(&bits, src + 2, 2);
    const float dmin = Bf(quant::FloatToBf16(quant::HalfToFloat(bits)));
    for (std::size_t g = 0; g < 8; ++g) {
      std::uint8_t sc;
      std::uint8_t m;
      quant::ScaleMinK4(g, src + 4, &sc, &m);
      const float step = d * static_cast<float>(sc);
      const float min = dmin * static_cast<float>(static_cast<std::int8_t>(-m));
      const std::uint8_t* q = src + 16 + (g / 2) * 32;
      const int shift = static_cast<int>(g % 2) * 4;
      for (std::size_t l = 0; l < 32; ++l) {
        dst[g * 32 + l] =
            min + step * static_cast<float>((q[l] >> shift) & 0xF);
      }
    }
  }
}

// --------------------------------------------------------- calibration data

struct Key {
  int stage, layer, expert;
  bool operator<(const Key& o) const {
    return std::tie(stage, layer, expert) <
           std::tie(o.stage, o.layer, o.expert);
  }
};
struct Stat {
  std::vector<double> sum2, sumabs;
  std::size_t n = 0;
  std::vector<float> samples;  // ns x dim
  std::size_t ns = 0;
  std::size_t calls = 0;
};
constexpr std::size_t kSamples = 64;
constexpr std::size_t kSampleStride = 53;

class Recorder : public q35::Projections {
public:
  // Routed experts are recorded only in `expert_layers` (the layers whose
  // experts requantize).
  Recorder(const q35::Projections& stored, const q35::Config& c,
           std::vector<bool> expert_layers)
      : stored_(stored), c_(c), expert_layers_(std::move(expert_layers)) {}
  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override {
    Record({static_cast<int>(stage), layer, -1},
           q35::StageInputs(c_, stage, layer), x, tokens, true);
    stored_.Apply(stage, layer, x, tokens, y);
  }
  void ApplyExpert(Stage stage, int layer, int expert, const float* x,
                   int tokens, float* y) const override {
    if (expert_layers_[layer]) {
      Record({static_cast<int>(stage), layer, expert},
             q35::StageInputs(c_, stage, layer), x, tokens, false);
    }
    stored_.ApplyExpert(stage, layer, expert, x, tokens, y);
  }
  std::map<Key, Stat> stats;

private:
  void Record(Key key, std::size_t dim, const float* x, int tokens,
              bool samples) const {
    std::lock_guard<std::mutex> hold(mutex_);
    auto& s = const_cast<Recorder*>(this)->stats[key];
    if (s.sum2.empty()) {
      s.sum2.assign(dim, 0);
      s.sumabs.assign(dim, 0);
    }
    for (int t = 0; t < tokens; ++t) {
      const float* xt = x + t * dim;
      for (std::size_t k = 0; k < dim; ++k) {
        s.sum2[k] += static_cast<double>(xt[k]) * xt[k];
        s.sumabs[k] += std::abs(xt[k]);
      }
      ++s.n;
      if (samples && s.ns < kSamples && s.calls % kSampleStride == 7) {
        s.samples.insert(s.samples.end(), xt, xt + dim);
        ++s.ns;
      }
      ++s.calls;
    }
  }
  const q35::Projections& stored_;
  const q35::Config& c_;
  std::vector<bool> expert_layers_;
  mutable std::mutex mutex_;
};

void SaveStats(const std::map<Key, Stat>& stats, const fs::path& path) {
  std::ofstream out(path, std::ios::binary);
  const std::size_t n = stats.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  for (const auto& [k, s] : stats) {
    out.write(reinterpret_cast<const char*>(&k), sizeof(k));
    const std::size_t dim = s.sum2.size();
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&s.n), sizeof(s.n));
    out.write(reinterpret_cast<const char*>(&s.ns), sizeof(s.ns));
    out.write(reinterpret_cast<const char*>(s.sum2.data()), dim * 8);
    out.write(reinterpret_cast<const char*>(s.sumabs.data()), dim * 8);
    out.write(reinterpret_cast<const char*>(s.samples.data()),
              s.samples.size() * 4);
  }
}

bool LoadStats(std::map<Key, Stat>& stats, const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::size_t n = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(n));
  for (std::size_t i = 0; i < n; ++i) {
    Key k;
    Stat s;
    std::size_t dim = 0;
    in.read(reinterpret_cast<char*>(&k), sizeof(k));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&s.n), sizeof(s.n));
    in.read(reinterpret_cast<char*>(&s.ns), sizeof(s.ns));
    s.sum2.resize(dim);
    s.sumabs.resize(dim);
    s.samples.resize(s.ns * dim);
    in.read(reinterpret_cast<char*>(s.sum2.data()), dim * 8);
    in.read(reinterpret_cast<char*>(s.sumabs.data()), dim * 8);
    in.read(reinterpret_cast<char*>(s.samples.data()), s.samples.size() * 4);
    stats[k] = std::move(s);
  }
  return static_cast<bool>(in);
}

// Mean x^2 per input, or null when the key saw too few tokens.
const std::vector<float>* Importance(const std::map<Key, Stat>& stats, Key key,
                                     std::map<Key, std::vector<float>>& cache) {
  auto it = cache.find(key);
  if (it != cache.end()) {
    return &it->second;
  }
  auto s = stats.find(key);
  if (s == stats.end() || s->second.n < 8) {
    return nullptr;
  }
  std::vector<float> imp(s->second.sum2.size());
  for (std::size_t k = 0; k < imp.size(); ++k) {
    imp[k] = static_cast<float>(s->second.sum2[k] / s->second.n);
  }
  return &(cache[key] = std::move(imp));
}

// AWQ: s = mean|x|^alpha, normalized, rounded to powers of two (exact to
// fold into bf16 norm weights); alpha chosen per matrix on the samples.
std::vector<float> AwqScales(const RowSource& src, const Stat& st,
                             const Method& m, const std::vector<float>* imp,
                             double* best_alpha, double* gain) {
  const std::size_t dim = src.dim;
  const std::size_t ns = st.ns;
  std::vector<float> mean_abs(dim);
  for (std::size_t k = 0; k < dim; ++k) {
    mean_abs[k] = static_cast<float>(st.sumabs[k] / st.n) + 1e-8f;
  }
  const double alphas[] = {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75};
  constexpr int kAlphas = 7;
  std::vector<std::vector<float>> scales(kAlphas, std::vector<float>(dim));
  std::vector<std::vector<float>> imps(kAlphas);
  for (int a = 0; a < kAlphas; ++a) {
    float lo = INFINITY, hi = 0;
    for (std::size_t k = 0; k < dim; ++k) {
      scales[a][k] = std::pow(mean_abs[k], static_cast<float>(alphas[a]));
      lo = std::min(lo, scales[a][k]);
      hi = std::max(hi, scales[a][k]);
    }
    const float norm = std::sqrt(lo * hi);
    for (std::size_t k = 0; k < dim; ++k) {
      scales[a][k] = std::exp2(std::nearbyint(std::log2(scales[a][k] / norm)));
    }
    if (imp != nullptr) {
      imps[a].resize(dim);
      for (std::size_t k = 0; k < dim; ++k) {
        imps[a][k] = (*imp)[k] / (scales[a][k] * scales[a][k]);
      }
    }
  }
  // Squared output error per alpha, summed over rows and samples.
  std::vector<double> err(kAlphas, 0.0);
  std::mutex err_mutex;
  ParallelRows(src.rows, 64, [&](std::size_t begin, std::size_t end) {
    std::vector<float> row(dim), scaled(dim), deq(dim);
    std::vector<double> local(kAlphas, 0.0);
    QBlock b;
    for (std::size_t r = begin; r < end; ++r) {
      src.row(r, row.data());
      std::vector<float> ref(ns);
      for (std::size_t i = 0; i < ns; ++i) {
        ref[i] =
            vec::Dot(row.data(), &st.samples[i * dim], static_cast<int>(dim));
      }
      for (int a = 0; a < kAlphas; ++a) {
        for (std::size_t k = 0; k < dim; ++k) {
          scaled[k] = row[k] * scales[a][k];
        }
        for (std::size_t kb = 0; kb < dim / 256; ++kb) {
          Quantize(scaled.data() + kb * 256,
                   imp != nullptr ? imps[a].data() + kb * 256 : nullptr, m, b);
          const float d = Bf(b.d);
          const float dmin = Bf(b.dmin);
          for (int g = 0; g < kGroups; ++g) {
            const float step = d * static_cast<float>(b.sc[g]);
            const float min = dmin * static_cast<float>(b.mn[g]);
            for (int l = 0; l < 32; ++l) {
              const std::size_t k = kb * 256 + g * 32 + l;
              // W' x' = (W s)(x / s): fold 1/s into the weight for the check.
              deq[k] = (min + step * static_cast<float>(b.q[g * 32 + l])) /
                       scales[a][k];
            }
          }
        }
        for (std::size_t i = 0; i < ns; ++i) {
          const double e = vec::Dot(deq.data(), &st.samples[i * dim],
                                    static_cast<int>(dim)) -
                           ref[i];
          local[a] += e * e;
        }
      }
    }
    std::lock_guard<std::mutex> hold(err_mutex);
    for (int a = 0; a < kAlphas; ++a) {
      err[a] += local[a];
    }
  });
  int best = 0;
  for (int a = 1; a < kAlphas; ++a) {
    if (err[a] < err[best]) {
      best = a;
    }
  }
  *best_alpha = alphas[best];
  *gain = err[0] > 0 ? err[best] / err[0] : 1.0;
  return scales[best];
}

// ------------------------------------------------------------ weight sets

struct Store {
  struct Layer {
    QMat in, out, ffn_in, ffn_down;
    std::vector<QMat> exp_in, exp_down;  // requantized experts only
    const TensorInfo* gate_exps = nullptr;
    const TensorInfo* up_exps = nullptr;
    const TensorInfo* down_exps = nullptr;
    const TensorInfo* alpha = nullptr;
    const TensorInfo* beta = nullptr;
  };
  std::vector<Layer> layers;
  QMat lm;
  double pack_seconds = 0;
  double dense_seconds = 0;
  std::string awq_log;
};

// Mean x^2 per input from a llama.cpp imatrix GGUF (<weight>.in_sum2 /
// <weight>.counts, per expert for the stacked expert tensors), or null.
const std::vector<float>* ImportanceFromFile(
    const gufo_npu::gguf::File& imatrix, const q35::Config& c, Key key,
    std::map<Key, std::vector<float>>& cache) {
  auto it = cache.find(key);
  if (it != cache.end()) {
    return &it->second;
  }
  const std::string p = "blk." + std::to_string(key.layer) + ".";
  std::string name;
  switch (static_cast<Stage>(key.stage)) {
    case Stage::kLinearIn:
      name = p + "attn_qkv.weight";
      break;
    case Stage::kAttentionIn:
      name = p + "attn_q.weight";
      break;
    case Stage::kMixerOut:
      name = p + (c.IsFullAttention(key.layer) ? "attn_output.weight"
                                               : "ssm_out.weight");
      break;
    case Stage::kFfnIn:
      name = p + "ffn_gate_shexp.weight";
      break;
    case Stage::kFfnDown:
      name = p + "ffn_down_shexp.weight";
      break;
    case Stage::kExpertIn:
      name = p + "ffn_gate_exps.weight";
      break;
    case Stage::kExpertDown:
      name = p + "ffn_down_exps.weight";
      break;
    default:
      return nullptr;  // no LM head in llama.cpp's imatrix
  }
  const TensorInfo* sums = imatrix.FindTensor(name + ".in_sum2");
  const TensorInfo* counts = imatrix.FindTensor(name + ".counts");
  if (sums == nullptr || counts == nullptr) {
    return nullptr;
  }
  const auto dim = static_cast<std::size_t>(sums->shape[0]);
  const std::size_t e =
      key.expert >= 0 ? static_cast<std::size_t>(key.expert) : 0;
  const auto* s = reinterpret_cast<const float*>(sums->data) + e * dim;
  const float n = reinterpret_cast<const float*>(counts->data)[e];
  if (n < 1) {
    return nullptr;
  }
  std::vector<float> imp(dim);
  for (std::size_t k = 0; k < dim; ++k) {
    imp[k] = s[k] / n;
  }
  return &(cache[key] = std::move(imp));
}

struct BuildContext {
  const gufo_npu::gguf::File& file;
  const q35::Config& c;
  const Method& m;
  const std::map<Key, Stat>* stats;
  std::map<Key, std::vector<float>> imp_cache;
  const gufo_npu::gguf::File* imatrix = nullptr;
};

QMat BuildOne(BuildContext& bc, const RowSource& src, Key key, bool awq_ok,
              std::string* log) {
  const bool wants_imp = bc.m.weighting == Weighting::kImatrix ||
                         bc.m.weighting == Weighting::kImatrixPlain;
  const std::vector<float>* imp = nullptr;
  if (wants_imp && bc.m.from_file && bc.imatrix != nullptr) {
    imp = ImportanceFromFile(*bc.imatrix, bc.c, key, bc.imp_cache);
  }
  if (wants_imp && imp == nullptr && bc.stats != nullptr) {
    // The calibration pass: everything for "imatrix", the LM head (which
    // llama.cpp's imatrix leaves out) for "imatrix-file".
    imp = Importance(*bc.stats, key, bc.imp_cache);
  }
  if (bc.m.awq && awq_ok && bc.stats != nullptr) {
    auto st = bc.stats->find(key);
    if (st != bc.stats->end() && st->second.ns > 0) {
      double alpha = 0, gain = 1;
      const auto s = AwqScales(src, st->second, bc.m, imp, &alpha, &gain);
      char line[160];
      std::snprintf(line, sizeof(line),
                    "  awq stage %d layer %d: alpha %.3f, output error x%.3f\n",
                    key.stage, key.layer, alpha, gain);
      *log += line;
      if (alpha > 0) {
        return Build(src, bc.m, imp, &s);
      }
    }
  }
  if (bc.m.library) {
    return BuildLibrary(src.parts, src.dim, q35::Q4nxFit(bc.c));
  }
  return Build(src, bc.m, imp, nullptr);
}

void PackSet(BuildContext& bc, Store& store, bool lm_head, bool experts) {
  const auto& c = bc.c;
  const auto& file = bc.file;
  const auto hidden = static_cast<std::size_t>(c.hidden);
  const double start = Now();
  if (store.layers.empty()) {
    store.layers.resize(c.layers);
    for (int il = 0; il < c.layers; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      auto t = [&](const char* name) { return &file.Tensor(p + name); };
      auto all = [](const TensorInfo* x) {
        return Part{x, 0, static_cast<std::size_t>(x->Rows())};
      };
      auto& l = store.layers[il];
      if (c.IsFullAttention(il)) {
        l.in = BuildOne(
            bc,
            FromParts({all(t("attn_q.weight")), all(t("attn_k.weight")),
                       all(t("attn_v.weight"))},
                      hidden),
            {static_cast<int>(Stage::kAttentionIn), il, -1}, true,
            &store.awq_log);
        l.out = BuildOne(
            bc, FromParts({all(t("attn_output.weight"))}, c.heads * c.head_dim),
            {static_cast<int>(Stage::kMixerOut), il, -1}, true, &store.awq_log);
      } else {
        l.in = BuildOne(
            bc,
            FromParts({all(t("attn_qkv.weight")), all(t("attn_gate.weight"))},
                      hidden),
            {static_cast<int>(Stage::kLinearIn), il, -1}, true, &store.awq_log);
        l.out = BuildOne(
            bc, FromParts({all(t("ssm_out.weight"))}, c.LinearValueDim()),
            {static_cast<int>(Stage::kMixerOut), il, -1}, true, &store.awq_log);
        l.alpha = t("ssm_alpha.weight");
        l.beta = t("ssm_beta.weight");
      }
      // The shared expert's gate/up shares its NPU call (and input scaling)
      // with the routed Q4_K experts: no AWQ there.
      l.ffn_in = BuildOne(
          bc,
          FromParts(
              {all(t("ffn_gate_shexp.weight")), all(t("ffn_up_shexp.weight"))},
              hidden),
          {static_cast<int>(Stage::kFfnIn), il, -1}, false, &store.awq_log);
      l.ffn_down = BuildOne(
          bc, FromParts({all(t("ffn_down_shexp.weight"))}, c.FfnWidth()),
          {static_cast<int>(Stage::kFfnDown), il, -1}, true, &store.awq_log);
      l.gate_exps = t("ffn_gate_exps.weight");
      l.up_exps = t("ffn_up_exps.weight");
      l.down_exps = t("ffn_down_exps.weight");
    }
    store.dense_seconds = Now() - start;
  }
  if (lm_head && store.lm.rows == 0) {
    const auto* head = &file.Tensor("output.weight");
    store.lm = BuildOne(
        bc,
        FromParts({{head, 0, static_cast<std::size_t>(head->Rows())}}, hidden),
        {static_cast<int>(Stage::kLmHead), 0, -1}, true, &store.awq_log);
  }
  if (experts) {
    const auto ef = static_cast<std::size_t>(c.expert_ffn);
    for (int il = 0; il < c.layers; ++il) {
      auto& l = store.layers[il];
      if (!l.exp_in.empty() || !l.exp_down.empty()) {
        continue;
      }
      for (int e = 0; e < c.experts; ++e) {
        if (l.gate_exps->type != GgmlType::kQ4_K ||
            l.up_exps->type != GgmlType::kQ4_K) {
          l.exp_in.push_back(BuildOne(
              bc,
              FromParts({{l.gate_exps, e * ef, ef}, {l.up_exps, e * ef, ef}},
                        hidden),
              {static_cast<int>(Stage::kExpertIn), il, e}, false,
              &store.awq_log));
        }
        if (l.down_exps->type != GgmlType::kQ4_K) {
          l.exp_down.push_back(
              BuildOne(bc, FromParts({{l.down_exps, e * hidden, hidden}}, ef),
                       {static_cast<int>(Stage::kExpertDown), il, e}, false,
                       &store.awq_log));
        }
      }
    }
  }
  store.pack_seconds += Now() - start;
}

class EvalSet : public q35::Projections {
public:
  EvalSet(const q35::Config& c, const q35::Projections& stored,
          const Store& store, bool dense, bool lm_head, bool experts)
      : c_(c),
        stored_(stored),
        store_(store),
        dense_(dense),
        lm_head_(lm_head),
        experts_(experts) {}

  void Apply(Stage stage, int layer, const float* x, int tokens,
             float* y) const override {
    const QMat* m = nullptr;
    if (stage == Stage::kLmHead) {
      m = lm_head_ ? &store_.lm : nullptr;
    } else if (dense_) {
      const auto& l = store_.layers[layer];
      switch (stage) {
        case Stage::kLinearIn:
        case Stage::kAttentionIn:
          m = &l.in;
          break;
        case Stage::kMixerOut:
          m = &l.out;
          break;
        case Stage::kFfnIn:
          m = &l.ffn_in;
          break;
        case Stage::kFfnDown:
          m = &l.ffn_down;
          break;
        default:
          break;
      }
    }
    if (m == nullptr) {
      return stored_.Apply(stage, layer, x, tokens, y);
    }
    const std::size_t in = q35::StageInputs(c_, stage, layer);
    const std::size_t out = q35::StageOutputs(c_, stage, layer);
    for (int t = 0; t < tokens; ++t) {
      Gemv(*m, x + t * in, y + t * out);
      if (stage == Stage::kLinearIn) {
        // alpha and beta: F32, as stored.
        const auto& l = store_.layers[layer];
        float* ab = y + t * out + m->rows;
        std::vector<float> row(in);
        for (const TensorInfo* w : {l.alpha, l.beta}) {
          for (std::uint64_t r = 0; r < w->Rows(); ++r) {
            quant::Dequantize(w->type,
                              w->data + r * quant::RowBytes(w->type, in),
                              row.data(), in);
            *ab++ = vec::Dot(row.data(), x + t * in, static_cast<int>(in));
          }
        }
      }
    }
  }

  void ApplyExpert(Stage stage, int layer, int expert, const float* x,
                   int tokens, float* y) const override {
    if (!experts_) {
      return stored_.ApplyExpert(stage, layer, expert, x, tokens, y);
    }
    const auto& l = store_.layers[layer];
    const auto e = static_cast<std::size_t>(expert);
    const auto ef = static_cast<std::size_t>(c_.expert_ffn);
    const auto hidden = static_cast<std::size_t>(c_.hidden);
    const std::size_t in = q35::StageInputs(c_, stage, layer);
    const std::size_t out = q35::StageOutputs(c_, stage, layer);
    const QMat* m = nullptr;
    if (stage == Stage::kExpertIn && !l.exp_in.empty()) {
      m = &l.exp_in[e];
    } else if (stage == Stage::kExpertDown && !l.exp_down.empty()) {
      m = &l.exp_down[e];
    }
    for (int t = 0; t < tokens; ++t) {
      if (m != nullptr) {
        Gemv(*m, x + t * in, y + t * out);
        continue;
      }
      std::vector<Part> parts;
      if (stage == Stage::kExpertIn) {
        parts = {{l.gate_exps, e * ef, ef}, {l.up_exps, e * ef, ef}};
      } else {
        parts = {{l.down_exps, e * hidden, hidden}};
      }
      std::size_t offset = 0;
      for (const auto& p : parts) {
        const auto row_bytes = quant::RowBytes(p.t->type, in);
        ParallelRows(p.rows, 32, [&](std::size_t begin, std::size_t end) {
          std::vector<float> row(in);
          for (std::size_t r = begin; r < end; ++r) {
            UnpackQ4KAsQ4nx(p.t->data + (p.first + r) * row_bytes, row.data(),
                            in);
            y[t * out + offset + r] =
                vec::Dot(row.data(), x + t * in, static_cast<int>(in));
          }
        });
        offset += p.rows;
      }
    }
  }

private:
  const q35::Config& c_;
  const q35::Projections& stored_;
  const Store& store_;
  bool dense_, lm_head_, experts_;
};

// ------------------------------------------------------------- evaluation

std::string ReadText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

std::vector<int> ReadIds(const fs::path& path) {
  std::string text = ReadText(path);
  std::replace_if(
      text.begin(), text.end(),
      [](char ch) { return ch == '[' || ch == ']' || ch == ','; }, ' ');
  std::istringstream stream(text);
  std::vector<int> ids;
  for (int id; stream >> id;) {
    ids.push_back(id);
  }
  return ids;
}

std::vector<float> ReadFloats(const fs::path& path) {
  const std::string bytes = ReadText(path);
  std::vector<float> out(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
  return out;
}

// The first ~2400 bytes, cut at a UTF-8 character boundary: plenty for 256
// tokens, and it keeps tokenizing cheap.
std::string Head(std::string text) {
  std::size_t n = std::min<std::size_t>(text.size(), 2400);
  while (n < text.size() &&
         (static_cast<unsigned char>(text[n]) & 0xC0) == 0x80) {
    --n;
  }
  text.resize(n);
  return text;
}

void WriteFloats(const fs::path& path, const std::vector<float>& v) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()), v.size() * 4);
}

std::size_t Argmax(const float* x, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(x, x + n) - x);
}

double Kl(const float* p, const float* q, std::size_t n) {
  auto lse = [n](const float* x) {
    const double max = *std::max_element(x, x + n);
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sum += std::exp(x[i] - max);
    }
    return max + std::log(sum);
  };
  const double lp = lse(p);
  const double lq = lse(q);
  double kl = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double logp = p[i] - lp;
    kl += std::exp(logp) * (logp - (q[i] - lq));
  }
  return kl;
}

struct Sequence {
  std::string name;
  std::vector<int> ids;
  std::vector<float> llama;  // oracle prompts only
  std::vector<float> stored;
};

std::vector<float> Decode(q35::Decoder& decoder, const std::vector<int>& ids) {
  const auto vocab = static_cast<std::size_t>(decoder.config().vocab);
  std::vector<float> out(ids.size() * vocab);
  auto state = decoder.NewState();
  for (std::size_t t = 0; t < ids.size(); ++t) {
    const auto& logits = decoder.Step(ids[t], state);
    std::copy(logits.begin(), logits.end(), out.begin() + t * vocab);
  }
  return out;
}

std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream in(s);
  for (std::string item; std::getline(in, item, ',');) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

Method ParseMethod(const std::string& name) {
  Method m;
  m.name = name;
  if (name == "library") {
    m.library = true;
  } else if (name == "minmax") {
    m.minmax = true;
  } else if (name == "search") {
    m.weighting = Weighting::kLlama;
  } else if (name == "search-uniform") {
    m.weighting = Weighting::kUniform;
  } else if (name == "search-refine") {
    m.refine = true;
  } else if (name == "imatrix") {
    m.weighting = Weighting::kImatrix;
  } else if (name == "imatrix-plain") {
    m.weighting = Weighting::kImatrixPlain;
  } else if (name == "imatrix-file") {
    m.weighting = Weighting::kImatrix;
    m.from_file = true;
  } else if (name == "awq-search") {
    m.awq = true;
  } else if (name == "awq-imatrix") {
    m.awq = true;
    m.weighting = Weighting::kImatrix;
  } else {
    std::fprintf(stderr, "unknown method %s\n", name.c_str());
    std::exit(2);
  }
  return m;
}

// Pass thresholds for the library packer, per weight set: KL(a || set) mean
// and top-1 agreement with (a) on the evaluation text, ~20% and ~1 point
// from the values measured 2026-10-27 (KL 0.0230 / 0.0268 / 0.0301, top-1
// 94.65% / 93.80% / 93.07%; docs/models/qwen3.6-35b-a3b/QUALITY.md). The
// min/max fit it replaced (KL 0.0316 / 0.0359 / 0.0394) fails all three.
struct Threshold {
  const char* set;
  double max_kl;
  double min_top1;
};
constexpr Threshold kThresholds[] = {
    {"d", 0.028, 0.935}, {"b", 0.032, 0.925}, {"c", 0.036, 0.92}};

// (a)'s logits for `s`, from the cache when it has them.
void StoredLogits(q35::Decoder& decoder, const fs::path& cache,
                  const std::string& key, Sequence& s, std::size_t vocab) {
  const fs::path path = cache.empty() ? fs::path() : cache / ("a-" + key);
  if (!cache.empty() && fs::exists(path) &&
      fs::file_size(path) == s.ids.size() * vocab * sizeof(float)) {
    s.stored = ReadFloats(path);
    return;
  }
  s.stored = Decode(decoder, s.ids);
  if (!cache.empty()) {
    WriteFloats(path, s.stored);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::vector<std::string> methods = {"library"};
  std::vector<std::string> sets = {"d", "b", "c"};
  fs::path cache;
  const char* imatrix_path = nullptr;
  for (int i = 1; i < argc; i += 2) {
    const std::string option = argv[i];
    if (i + 1 >= argc) {
      std::fprintf(stderr, "%s needs a value\n", option.c_str());
      return 2;
    }
    if (option == "--methods") {
      methods = Split(argv[i + 1]);
    } else if (option == "--sets") {
      sets = Split(argv[i + 1]);
    } else if (option == "--cache") {
      cache = argv[i + 1];
    } else if (option == "--imatrix") {
      imatrix_path = argv[i + 1];
    } else {
      std::fprintf(stderr,
                   "usage: qwen36_q4nx_quality_test [--methods m,...] "
                   "[--sets d,b,c] [--cache dir] [--imatrix file]\n");
      return 2;
    }
  }
  const char* model = std::getenv("GUFO_NPU_QWEN36_MODEL");
  const char* oracle = std::getenv("GUFO_NPU_QWEN36_ORACLE");
  const char* text = std::getenv("GUFO_NPU_QWEN36_TEXT");
  if (model == nullptr || oracle == nullptr || text == nullptr ||
      *model == '\0' || *oracle == '\0' || *text == '\0') {
    std::puts(
        "SKIP: set GUFO_NPU_QWEN36_MODEL, GUFO_NPU_QWEN36_ORACLE and "
        "GUFO_NPU_QWEN36_TEXT");
    return 77;
  }
  const fs::path text_dir = text;
  constexpr std::size_t kMaxTokens = 256;
  const auto file = gufo_npu::gguf::File::Open(model);
  const gufo_npu::Tokenizer tokenizer(*file);
  const q35::Config c = q35::ReadConfig(*file);
  const auto vocab = static_cast<std::size_t>(c.vocab);
  const q35::GgufProjections stored(*file,
                                    q35::GgufProjections::Precision::kFloat);
  if (!cache.empty()) {
    fs::create_directories(cache);
  }
  // Every text in `dir`, up to kMaxTokens tokens each.
  auto load = [&](const fs::path& dir) {
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
      paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    std::vector<Sequence> out;
    for (const auto& path : paths) {
      Sequence s;
      s.name = path.stem().string();
      s.ids = tokenizer.Encode(Head(ReadText(path)));
      if (s.ids.size() > kMaxTokens) {
        s.ids.resize(kMaxTokens);
      }
      out.push_back(std::move(s));
    }
    return out;
  };
  std::vector<Sequence> eval = load(text_dir / "eval");
  std::size_t eval_positions = 0;
  std::printf("evaluation text:");
  for (const auto& s : eval) {
    std::printf(" %s %zu", s.name.c_str(), s.ids.size());
    eval_positions += s.ids.size();
  }
  std::printf(" (%zu positions)\n", eval_positions);
  std::vector<Sequence> prompts;
  for (const char* name : {"chat", "code", "prose", "long"}) {
    Sequence s;
    s.name = name;
    s.ids = ReadIds(fs::path(oracle) / (s.name + ".ids.json"));
    s.llama = ReadFloats(fs::path(oracle) / (s.name + ".logits.f32"));
    if (s.ids.empty() || s.llama.size() != s.ids.size() * vocab) {
      std::fprintf(stderr, "FAIL: oracle files for %s\n", name);
      return 1;
    }
    prompts.push_back(std::move(s));
  }
  {
    const double t0 = Now();
    q35::Decoder decoder(*file, stored);
    for (auto& s : eval) {
      StoredLogits(decoder, cache, s.name + ".eval", s, vocab);
    }
    for (auto& s : prompts) {
      StoredLogits(decoder, cache, s.name + ".oracle", s, vocab);
    }
    std::printf("(a) logits ready (%.0f s)\n", Now() - t0);
  }

  // Calibration statistics over <text>/calib with the stored weights.
  std::map<Key, Stat> stats;
  bool need_stats = false;
  for (const auto& name : methods) {
    const Method m = ParseMethod(name);
    need_stats = need_stats || m.awq || m.weighting == Weighting::kImatrix ||
                 m.weighting == Weighting::kImatrixPlain;
    if (m.from_file && imatrix_path == nullptr) {
      std::fprintf(stderr, "imatrix-file needs --imatrix\n");
      return 2;
    }
  }
  if (need_stats &&
      (cache.empty() || !LoadStats(stats, cache / "calib.stats"))) {
    const double t0 = Now();
    std::vector<bool> expert_layers(c.layers);
    for (int il = 0; il < c.layers; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      for (const char* name : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                               "ffn_down_exps.weight"}) {
        expert_layers[il] =
            expert_layers[il] || file->Tensor(p + name).type != GgmlType::kQ4_K;
      }
    }
    Recorder recorder(stored, c, expert_layers);
    q35::Decoder decoder(*file, recorder);
    const auto calib = load(text_dir / "calib");
    std::size_t tokens = 0;
    for (const auto& s : calib) {
      auto state = decoder.NewState();
      for (const int id : s.ids) {
        decoder.Step(id, state);
      }
      tokens += s.ids.size();
    }
    stats = std::move(recorder.stats);
    if (!cache.empty()) {
      SaveStats(stats, cache / "calib.stats");
    }
    std::printf("calibration: %zu texts, %zu tokens (%.0f s)\n", calib.size(),
                tokens, Now() - t0);
  }

  int failures = 0;
  for (const auto& name : methods) {
    const Method m = ParseMethod(name);
    Store store;
    BuildContext bc{*file, c, m, need_stats ? &stats : nullptr, {}};
    std::unique_ptr<gufo_npu::gguf::File> imatrix;
    if (m.from_file) {
      imatrix = gufo_npu::gguf::File::Open(imatrix_path);
      bc.imatrix = imatrix.get();
    }
    for (const auto& set : sets) {
      const bool lm_head = set != "d";
      const bool experts = set == "c";
      PackSet(bc, store, lm_head, experts);
      if (!store.awq_log.empty()) {
        std::printf("%s", store.awq_log.c_str());
        store.awq_log.clear();
      }
      const EvalSet projections(c, stored, store, true, lm_head, experts);
      const double t0 = Now();
      q35::Decoder decoder(*file, projections);
      double kl_sum = 0;
      std::size_t n = 0;
      std::size_t top1 = 0;
      std::string per;
      for (auto& s : eval) {
        const auto logits = Decode(decoder, s.ids);
        double kl = 0;
        std::size_t same = 0;
        for (std::size_t t = 0; t < s.ids.size(); ++t) {
          kl += Kl(&s.stored[t * vocab], &logits[t * vocab], vocab);
          same += Argmax(&logits[t * vocab], vocab) ==
                          Argmax(&s.stored[t * vocab], vocab)
                      ? 1
                      : 0;
        }
        kl_sum += kl;
        top1 += same;
        n += s.ids.size();
        char buf[96];
        std::snprintf(buf, sizeof(buf), " %s %.4f/%.1f%%", s.name.c_str(),
                      kl / s.ids.size(), 100.0 * same / s.ids.size());
        per += buf;
      }
      double oracle_kl = 0;
      std::size_t on = 0;
      std::size_t o_llama = 0;
      std::size_t o_a = 0;
      for (auto& s : prompts) {
        const auto logits = Decode(decoder, s.ids);
        for (std::size_t t = 0; t < s.ids.size(); ++t) {
          const auto top = Argmax(&logits[t * vocab], vocab);
          o_llama += top == Argmax(&s.llama[t * vocab], vocab) ? 1 : 0;
          o_a += top == Argmax(&s.stored[t * vocab], vocab) ? 1 : 0;
          oracle_kl += Kl(&s.stored[t * vocab], &logits[t * vocab], vocab);
        }
        on += s.ids.size();
      }
      const double kl = kl_sum / n;
      const double agree = static_cast<double>(top1) / n;
      const char* verdict = "    ";
      if (m.library) {
        for (const auto& th : kThresholds) {
          if (set == th.set) {
            const bool ok = kl <= th.max_kl && agree >= th.min_top1;
            verdict = ok ? "ok  " : "FAIL";
            failures += ok ? 0 : 1;
          }
        }
      }
      std::printf(
          "%s %-14s (%s) | text %zu: KL %.4f, top-1 vs (a) %.2f%% | oracle "
          "%zu: KL %.4f, top-1 vs (a) %zu (%.1f%%), vs llama.cpp %zu (%.1f%%) "
          "| packed in %.1f s | %.0f s\n",
          verdict, m.name.c_str(), set.c_str(), n, kl, 100.0 * agree, on,
          oracle_kl / on, o_a, 100.0 * o_a / on, o_llama, 100.0 * o_llama / on,
          store.pack_seconds, Now() - t0);
      std::printf("  per text (KL/top-1):%s\n", per.c_str());
    }
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
