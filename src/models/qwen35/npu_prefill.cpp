#include "models/qwen35/npu_prefill.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "core/burst_pool.hpp"
#include "core/q4nx.hpp"
#include "core/quant.hpp"
#include "core/simd.hpp"
#include "core/vec.hpp"
#include "npu/gemm.hpp"

namespace gufo_npu::qwen35 {
namespace {

using npu::Bfp16Gemm;
using npu::GemmBuffer;
using vec::Dot;
using vec::RmsNorm;

constexpr int kMaxChunk = static_cast<int>(npu::kGemmMaxChunk);

// The model the chunk plan is chosen with, measured on the 2B (BENCHMARKS.md,
// "Prefill GEMM" and "Where the time goes"): the GEMMs' throughput at each
// chunk size of npu::kGemmChunks (a 256-row call runs at ~11 TFLOPS, a
// 1,024-row one at ~17), and the CPU's time per token between them (~0.14 ms
// at 1k tokens: mixers, norms, conversions).
constexpr double kGemmTflops[] = {11.0, 14.5, 17.0};
static_assert(std::size(kGemmTflops) == std::size(npu::kGemmChunks));
constexpr double kCpuSecondsPerToken = 0.14e-3;

// One pass of a prefill: `lanes` chunks of `m` GEMM rows over `tokens`
// prompt tokens (the last chunk padded).
struct Pass {
  int m = 0;
  int lanes = 0;
  int tokens = 0;
};

// The passes over `count` tokens with the least estimated time, for GEMMs of
// `flops` per token and `cpu` seconds of CPU work per token. One lane takes
// its GEMMs' time and then its CPU time; two lanes overlap one chunk's GEMMs
// with the other's CPU work, about max(GEMMs of both, CPU of both). So 1,000
// tokens run as 512 + 488 (GEMMs ~0.2 s) rather than four chunks of 256
// (~0.25 s) or one of 1,024 without overlap.
std::vector<Pass> PlanPasses(int count, int max_lanes, double flops,
                             double cpu) {
  const auto npu = [&](std::size_t chunk) {
    return flops * static_cast<double>(npu::kGemmChunks[chunk]) /
           (kGemmTflops[chunk] * 1e12);
  };
  // best[r]: least time for the last r tokens, and the pass that starts it.
  std::vector<double> best(static_cast<std::size_t>(count) + 1, 0.0);
  std::vector<Pass> first(static_cast<std::size_t>(count) + 1);
  for (int r = 1; r <= count; ++r) {
    best[r] = INFINITY;
    for (std::size_t c = 0; c < std::size(npu::kGemmChunks); ++c) {
      const int m = static_cast<int>(npu::kGemmChunks[c]);
      for (int lanes = 1; lanes <= max_lanes; ++lanes) {
        if (lanes > 1 && r <= (lanes - 1) * m) {
          continue;  // a lane would be empty
        }
        const int tokens = std::min(r, lanes * m);
        const double t = lanes == 1 ? npu(c) + cpu * tokens
                                    : std::max(lanes * npu(c), cpu * tokens);
        if (t + best[r - tokens] < best[r]) {
          best[r] = t + best[r - tokens];
          first[r] = {m, lanes, tokens};
        }
      }
    }
  }
  std::vector<Pass> plan;
  for (int r = count; r > 0; r -= first[r].tokens) {
    plan.push_back(first[r]);
  }
  return plan;
}

// Attention: query tokens per work item, keys per online-softmax block.
constexpr int kQueryBlock = 16;
constexpr int kKeyBlock = 64;

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<float> ToFloats(const gguf::TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

q4nx::Segment SegmentOf(const gguf::TensorInfo& t) {
  return {t.type, t.data, static_cast<std::size_t>(t.Rows())};
}

// The weights NpuDecoder multiplies (the same q4nx packing), row-major fp32.
std::vector<float> Q4nxWeights(const std::vector<q4nx::Segment>& segments,
                               std::size_t dim) {
  const auto packed = q4nx::PackRows(segments, dim);
  auto w = q4nx::Unpack(packed);
  w.resize(packed.rows * dim);
  return w;
}

float Sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}
float Softplus(float x) {
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

// The delta rule for one token on kDeltaColumns value columns j of a head's
// state S (dk rows of the columns, `stride` apart), with the decoder's
// arithmetic: u = (decay S)^T k, delta = beta (v - u), S' = decay S +
// k delta^T, y = S'^T q.
constexpr int kDeltaColumns = 32;
constexpr int kMaxHeadDim = 128;

void DeltaColumns(float* s, int dk, int stride, const float* k, const float* q,
                  const float* v, float decay, float beta, float* y) {
#if defined(__AVX2__) && defined(__FMA__)
  __m256 acc[4];
  for (auto& a : acc) {
    a = _mm256_setzero_ps();
  }
  for (int i = 0; i < dk; ++i) {
    const float* row = s + static_cast<std::size_t>(i) * stride;
    const __m256 ki = _mm256_set1_ps(k[i] * decay);
    for (int c = 0; c < 4; ++c) {
      acc[c] = _mm256_fmadd_ps(_mm256_loadu_ps(row + 8 * c), ki, acc[c]);
    }
  }
  __m256 delta[4];
  __m256 out[4];
  const __m256 b = _mm256_set1_ps(beta);
  for (int c = 0; c < 4; ++c) {
    delta[c] =
        _mm256_mul_ps(b, _mm256_sub_ps(_mm256_loadu_ps(v + 8 * c), acc[c]));
    out[c] = _mm256_setzero_ps();
  }
  const __m256 dc = _mm256_set1_ps(decay);
  for (int i = 0; i < dk; ++i) {
    float* row = s + static_cast<std::size_t>(i) * stride;
    const __m256 ki = _mm256_set1_ps(k[i]);
    const __m256 qi = _mm256_set1_ps(q[i]);
    for (int c = 0; c < 4; ++c) {
      const __m256 updated = _mm256_fmadd_ps(
          ki, delta[c], _mm256_mul_ps(_mm256_loadu_ps(row + 8 * c), dc));
      _mm256_storeu_ps(row + 8 * c, updated);
      out[c] = _mm256_fmadd_ps(updated, qi, out[c]);
    }
  }
  for (int c = 0; c < 4; ++c) {
    _mm256_storeu_ps(y + 8 * c, out[c]);
  }
#else
  float delta[kDeltaColumns] = {};
  for (int i = 0; i < dk; ++i) {
    const float* row = s + static_cast<std::size_t>(i) * stride;
    const float ki = k[i] * decay;
    for (int j = 0; j < kDeltaColumns; ++j) {
      delta[j] += row[j] * ki;
    }
  }
  for (int j = 0; j < kDeltaColumns; ++j) {
    delta[j] = beta * (v[j] - delta[j]);
    y[j] = 0;
  }
  for (int i = 0; i < dk; ++i) {
    float* row = s + static_cast<std::size_t>(i) * stride;
    for (int j = 0; j < kDeltaColumns; ++j) {
      row[j] = row[j] * decay + k[i] * delta[j];
      y[j] += row[j] * q[i];
    }
  }
#endif
}

// Lane scratch starts on a cache line, so parallel jobs that write
// different 128-byte column blocks of a row never share a line.
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

// Attention kernels over one token's G = 4 query heads of a kv head (fp32
// rows hd apart) against the fp16 K/V cache (vec::ToHalf bits, rows `stride`
// halves apart), widening the cache in registers (F16C or AVX-512).
using Half = std::uint16_t;

#if defined(__AVX2__) && defined(__FMA__)
float HorizontalSum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

__m256 LoadHalf8(const Half* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}
#endif

#if defined(__x86_64__) && defined(__GNUC__)
// Zen 5 runs AVX-512 at full width: the attention kernels have 512-bit
// versions, picked at run time (the build targets AVX2).
#define GUFO_PREFILL_AVX512 1

bool HasAvx512() {
  static const bool has = __builtin_cpu_supports("avx512f");
  return has;
}

__attribute__((target("avx512f"))) __m512 LoadHalf16(const Half* p) {
  return _mm512_cvtph_ps(
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}

__attribute__((target("avx512f"))) void Scores4Avx512(const float* q, int hd,
                                                      const Half* keys,
                                                      std::size_t stride,
                                                      int count, float* scores,
                                                      int ld) {
  int j = 0;
  for (; j + 2 <= count; j += 2) {
    const Half* k0 = keys + static_cast<std::size_t>(j) * stride;
    const Half* k1 = k0 + stride;
    __m512 acc[4][2];
    for (auto& a : acc) {
      a[0] = _mm512_setzero_ps();
      a[1] = _mm512_setzero_ps();
    }
    for (int e = 0; e < hd; e += 16) {
      const __m512 x0 = LoadHalf16(k0 + e);
      const __m512 x1 = LoadHalf16(k1 + e);
      for (int g = 0; g < 4; ++g) {
        const __m512 qv = _mm512_loadu_ps(q + g * hd + e);
        acc[g][0] = _mm512_fmadd_ps(qv, x0, acc[g][0]);
        acc[g][1] = _mm512_fmadd_ps(qv, x1, acc[g][1]);
      }
    }
    for (int g = 0; g < 4; ++g) {
      scores[g * ld + j] = _mm512_reduce_add_ps(acc[g][0]);
      scores[g * ld + j + 1] = _mm512_reduce_add_ps(acc[g][1]);
    }
  }
  for (; j < count; ++j) {
    const Half* key = keys + static_cast<std::size_t>(j) * stride;
    for (int g = 0; g < 4; ++g) {
      scores[g * ld + j] = vec::DotHalf(q + g * hd, key, hd);
    }
  }
}

__attribute__((target("avx512f"))) void AccumulateValues4Avx512(
    const float* p, int ld, const Half* values, std::size_t stride, int count,
    int hd, float* out) {
  for (int e = 0; e < hd; e += 32) {
    __m512 acc[4][2];
    for (int g = 0; g < 4; ++g) {
      acc[g][0] = _mm512_loadu_ps(out + g * hd + e);
      acc[g][1] = _mm512_loadu_ps(out + g * hd + e + 16);
    }
    for (int j = 0; j < count; ++j) {
      const Half* v = values + static_cast<std::size_t>(j) * stride + e;
      const __m512 v0 = LoadHalf16(v);
      const __m512 v1 = LoadHalf16(v + 16);
      for (int g = 0; g < 4; ++g) {
        const __m512 pg = _mm512_set1_ps(p[g * ld + j]);
        acc[g][0] = _mm512_fmadd_ps(pg, v0, acc[g][0]);
        acc[g][1] = _mm512_fmadd_ps(pg, v1, acc[g][1]);
      }
    }
    for (int g = 0; g < 4; ++g) {
      _mm512_storeu_ps(out + g * hd + e, acc[g][0]);
      _mm512_storeu_ps(out + g * hd + e + 16, acc[g][1]);
    }
  }
}
#endif

// scores[g * ld + j] = q_g . key_j for G = 4 query heads and keys j in
// [0, count); hd a multiple of 32.
void Scores4(const float* q, int hd, const Half* keys, std::size_t stride,
             int count, float* scores, int ld) {
#ifdef GUFO_PREFILL_AVX512
  if (HasAvx512()) {
    Scores4Avx512(q, hd, keys, stride, count, scores, ld);
    return;
  }
#endif
  int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
  for (; j + 2 <= count; j += 2) {
    const Half* k0 = keys + static_cast<std::size_t>(j) * stride;
    const Half* k1 = k0 + stride;
    __m256 acc[4][2];
    for (auto& a : acc) {
      a[0] = _mm256_setzero_ps();
      a[1] = _mm256_setzero_ps();
    }
    for (int e = 0; e < hd; e += 8) {
      const __m256 x0 = LoadHalf8(k0 + e);
      const __m256 x1 = LoadHalf8(k1 + e);
      for (int g = 0; g < 4; ++g) {
        const __m256 qv = _mm256_loadu_ps(q + g * hd + e);
        acc[g][0] = _mm256_fmadd_ps(qv, x0, acc[g][0]);
        acc[g][1] = _mm256_fmadd_ps(qv, x1, acc[g][1]);
      }
    }
    for (int g = 0; g < 4; ++g) {
      scores[g * ld + j] = HorizontalSum(acc[g][0]);
      scores[g * ld + j + 1] = HorizontalSum(acc[g][1]);
    }
  }
#endif
  for (; j < count; ++j) {
    const Half* key = keys + static_cast<std::size_t>(j) * stride;
    for (int g = 0; g < 4; ++g) {
      scores[g * ld + j] = vec::DotHalf(q + g * hd, key, hd);
    }
  }
}

// out_g += sum_j p[g * ld + j] value_j for G = 4 rows of hd (a multiple of
// 32).
void AccumulateValues4(const float* p, int ld, const Half* values,
                       std::size_t stride, int count, int hd, float* out) {
#ifdef GUFO_PREFILL_AVX512
  if (HasAvx512()) {
    AccumulateValues4Avx512(p, ld, values, stride, count, hd, out);
    return;
  }
#endif
#if defined(__AVX2__) && defined(__FMA__)
  for (int e = 0; e < hd; e += 16) {
    __m256 acc[4][2];
    for (int g = 0; g < 4; ++g) {
      acc[g][0] = _mm256_loadu_ps(out + g * hd + e);
      acc[g][1] = _mm256_loadu_ps(out + g * hd + e + 8);
    }
    for (int j = 0; j < count; ++j) {
      const Half* v = values + static_cast<std::size_t>(j) * stride + e;
      const __m256 v0 = LoadHalf8(v);
      const __m256 v1 = LoadHalf8(v + 8);
      for (int g = 0; g < 4; ++g) {
        const __m256 pg = _mm256_broadcast_ss(p + g * ld + j);
        acc[g][0] = _mm256_fmadd_ps(pg, v0, acc[g][0]);
        acc[g][1] = _mm256_fmadd_ps(pg, v1, acc[g][1]);
      }
    }
    for (int g = 0; g < 4; ++g) {
      _mm256_storeu_ps(out + g * hd + e, acc[g][0]);
      _mm256_storeu_ps(out + g * hd + e + 8, acc[g][1]);
    }
  }
#else
  for (int j = 0; j < count; ++j) {
    const Half* v = values + static_cast<std::size_t>(j) * stride;
    for (int g = 0; g < 4; ++g) {
      vec::AddScaledHalf(p[g * ld + j], v, out + g * hd, hd);
    }
  }
#endif
}

}  // namespace

struct NpuPrefill::Impl {
  struct Layer {
    bool attention = false;
    std::unique_ptr<Bfp16Gemm> in, out, ffn, down;
    std::vector<float> attn_norm, post_norm;
    // Gated DeltaNet.
    std::vector<float> alpha, beta;  // [value heads][hidden]
    std::vector<float> conv;         // [kernel][channels] (transposed)
    std::vector<float> ssm_a, dt_bias, ssm_norm;
    // Gated attention.
    std::vector<float> q_norm, k_norm;
  };

  Config config;
  npu::GemmRuntime runtime;
  const gguf::TensorInfo* embed = nullptr;
  std::vector<Layer> layers;
  std::vector<float> rope_inv_freq;

  // MTP: eh_proj, then the block's input projection (its keys and values).
  bool has_mtp = false;
  Layer mtp;
  std::unique_ptr<Bfp16Gemm> eh_proj;
  std::vector<float> enorm, hnorm, output_norm;

  // One chunk of prompt tokens in flight: its GEMM operands (A: rows x dim
  // bf16, C: rows x out bf16) and its CPU scratch. Two lanes run two
  // consecutive chunks through the layers together, so one lane's GEMMs run
  // on the NPU while the CPU works on the other's.
  struct Lane {
    int n = 0;         // tokens
    int m = 0;         // GEMM rows (n padded to a chunk size)
    int position = 0;  // of the first
    std::unique_ptr<GemmBuffer> a, c;
    npu::GemmCall call;
    std::size_t c_sync = 0;     // bytes of C to sync once the call is done
    Floats h;                   // residual stream
    Floats normed;              // RMS(h) * attn_norm, for the DeltaNet gates
    Floats in;                  // input projection, fp32
    Floats mixed;               // mixer output
    Floats gates;               // decay | beta per token and value head
    Floats u;                   // conv output: q | k | v per token
    Floats queries;             // normed, rotated and scaled
    Floats rope_cos, rope_sin;  // per token
  };
  Lane lanes[2];
  // CPU workers, spinning for the length of a Run or FillMtp: one per
  // physical core with the caller (half the hardware threads, for SMT).
  // Measured on the 16-core box at 1k tokens: 15 workers 0.61 s, 7 workers
  // 0.75 s, 23 workers 0.78 s, 31 workers 1.4 s (spinners on every hardware
  // thread delay the NPU calls' completions).
  BurstPool pool{
      std::max<std::size_t>(std::thread::hardware_concurrency() / 2, 1) - 1};

  Profile profile;
  // GEMM work per token of a prefill pass and of the MTP cache fill.
  double flops_per_token = 0;
  double mtp_flops_per_token = 0;

  Layer MakeLayer(const gguf::File& file, int il, bool full) {
    const Config& cfg = config;
    const auto hidden = static_cast<std::size_t>(cfg.hidden);
    const std::string p = "blk." + std::to_string(il) + ".";
    auto t = [&](const char* name) -> const gguf::TensorInfo& {
      return file.Tensor(p + name);
    };
    auto gemm = [&](const std::vector<q4nx::Segment>& segments,
                    std::size_t dim) {
      const auto w = Q4nxWeights(segments, dim);
      std::size_t rows = 0;
      for (const auto& s : segments) {
        rows += s.rows;
      }
      return std::make_unique<Bfp16Gemm>(runtime, w.data(), rows, dim);
    };
    Layer l;
    l.attention = cfg.IsFullAttention(il);
    l.attn_norm = ToFloats(t("attn_norm.weight"));
    if (l.attention) {
      l.in = gemm({SegmentOf(t("attn_q.weight")), SegmentOf(t("attn_k.weight")),
                   SegmentOf(t("attn_v.weight"))},
                  hidden);
      l.q_norm = ToFloats(t("attn_q_norm.weight"));
      l.k_norm = ToFloats(t("attn_k_norm.weight"));
    } else {
      l.in = gemm(
          {SegmentOf(t("attn_qkv.weight")), SegmentOf(t("attn_gate.weight"))},
          hidden);
      l.alpha = ToFloats(t("ssm_alpha.weight"));
      l.beta = ToFloats(t("ssm_beta.weight"));
      const auto conv = ToFloats(t("ssm_conv1d.weight"));  // [ch][kernel]
      const int kernel = cfg.conv_kernel;
      const int channels = cfg.ConvChannels();
      l.conv.resize(conv.size());
      for (int ch = 0; ch < channels; ++ch) {
        for (int s = 0; s < kernel; ++s) {
          l.conv[static_cast<std::size_t>(s) * channels + ch] =
              conv[static_cast<std::size_t>(ch) * kernel + s];
        }
      }
      l.ssm_a = ToFloats(t("ssm_a"));
      l.dt_bias = ToFloats(t("ssm_dt.bias"));
      l.ssm_norm = ToFloats(t("ssm_norm.weight"));
    }
    if (!full) {
      return l;  // the MTP block: only its input projection is needed
    }
    const auto& out = t(l.attention ? "attn_output.weight" : "ssm_out.weight");
    l.out = gemm({SegmentOf(out)}, static_cast<std::size_t>(out.RowElements()));
    l.ffn =
        gemm({SegmentOf(t("ffn_gate.weight")), SegmentOf(t("ffn_up.weight"))},
             hidden);
    const auto& down = t("ffn_down.weight");
    l.down =
        gemm({SegmentOf(down)}, static_cast<std::size_t>(down.RowElements()));
    l.post_norm = ToFloats(t("post_attention_norm.weight"));
    return l;
  }

  Impl(const gguf::File& file, const std::filesystem::path& kernel_dir)
      : config(ReadConfig(file)), runtime(kernel_dir) {
    const Config& cfg = config;
    if (cfg.experts != 0 || !cfg.tied_embeddings) {
      throw std::runtime_error(
          "npu prefill: dense, tied-embedding models only");
    }
    if (cfg.heads != 4 * cfg.kv_heads || cfg.head_dim % 32 != 0) {
      throw std::runtime_error(
          "npu prefill: 4 query heads per kv head and a head size that is a "
          "multiple of 32 only");
    }
    if (cfg.linear_head_dim > kMaxHeadDim ||
        cfg.linear_head_dim % kDeltaColumns != 0) {
      throw std::runtime_error("npu prefill: unsupported DeltaNet head size");
    }
    embed = &file.Tensor("token_embd.weight");
    for (int il = 0; il < cfg.layers; ++il) {
      layers.push_back(MakeLayer(file, il, true));
    }
    if (cfg.mtp_layers > 0) {
      has_mtp = true;
      const std::string p = "blk." + std::to_string(cfg.layers) + ".";
      mtp = MakeLayer(file, cfg.layers, false);
      const auto& eh = file.Tensor(p + "nextn.eh_proj.weight");
      const auto w = Q4nxWeights({SegmentOf(eh)},
                                 static_cast<std::size_t>(eh.RowElements()));
      eh_proj = std::make_unique<Bfp16Gemm>(
          runtime, w.data(), static_cast<std::size_t>(eh.Rows()),
          static_cast<std::size_t>(eh.RowElements()));
      enorm = ToFloats(file.Tensor(p + "nextn.enorm.weight"));
      hnorm = ToFloats(file.Tensor(p + "nextn.hnorm.weight"));
      output_norm = ToFloats(file.Tensor("output_norm.weight"));
    }
    auto flops = [](const std::unique_ptr<Bfp16Gemm>& g) {
      return g ? 2.0 * static_cast<double>(g->rows() * g->dim()) : 0.0;
    };
    for (const auto& l : layers) {
      flops_per_token +=
          flops(l.in) + flops(l.out) + flops(l.ffn) + flops(l.down);
    }
    mtp_flops_per_token = flops(eh_proj) + flops(mtp.in);
    rope_inv_freq.resize(cfg.rope_dim / 2);
    for (int i = 0; i < cfg.rope_dim / 2; ++i) {
      rope_inv_freq[i] = static_cast<float>(
          1.0 / std::pow(static_cast<double>(cfg.rope_theta),
                         2.0 * i / static_cast<double>(cfg.rope_dim)));
    }

    std::size_t a_bytes = 0;
    std::size_t c_bytes = 0;
    auto grow = [&](const std::unique_ptr<Bfp16Gemm>& g) {
      if (g) {
        a_bytes = std::max(a_bytes, g->a_bytes(npu::kGemmMaxChunk));
        c_bytes = std::max(c_bytes, g->c_bytes(npu::kGemmMaxChunk));
      }
    };
    std::size_t in_rows = 0;
    for (const auto& l : layers) {
      for (const auto* g : {&l.in, &l.out, &l.ffn, &l.down}) {
        grow(*g);
      }
      in_rows = std::max(in_rows, l.in->rows());
    }
    grow(mtp.in);
    grow(eh_proj);
    if (mtp.in) {
      in_rows = std::max(in_rows, mtp.in->rows());  // FillMtpChunk's `in`
    }
    const auto rows = static_cast<std::size_t>(kMaxChunk);
    const auto d = static_cast<std::size_t>(cfg.hidden);
    for (Lane& lane : lanes) {
      lane.a = std::make_unique<GemmBuffer>(runtime, a_bytes);
      lane.c = std::make_unique<GemmBuffer>(runtime, c_bytes);
      lane.h.resize(rows * d);
      lane.normed.resize(rows * d);
      lane.in.resize(rows * in_rows);
      lane.mixed.resize(
          rows * std::max(cfg.LinearValueDim(), cfg.heads * cfg.head_dim));
      lane.gates.resize(rows * 2 * cfg.linear_value_heads);
      lane.u.resize(rows * cfg.ConvChannels());
      lane.queries.resize(rows * cfg.heads * cfg.head_dim);
      lane.rope_cos.resize(rows * cfg.rope_dim / 2);
      lane.rope_sin.resize(lane.rope_cos.size());
    }
  }

  // Starts `g` over the lane's first n rows of A into C.
  void Start(Bfp16Gemm& g, Lane& lane) {
    const auto start = Clock::now();
    lane.a->ToDevice(static_cast<std::size_t>(lane.n) * g.dim() *
                     sizeof(std::uint16_t));
    profile.sync_seconds += SecondsSince(start);
    lane.call = g.Start(*lane.a, *lane.c, static_cast<std::size_t>(lane.m));
    lane.c_sync =
        static_cast<std::size_t>(lane.n) * g.rows() * sizeof(std::uint16_t);
    ++profile.gemms;
  }

  // Waits for the lane's call and makes its C visible.
  void Finish(Lane& lane) {
    auto start = Clock::now();
    lane.call.Wait();
    profile.gemm_seconds += SecondsSince(start);
    start = Clock::now();
    lane.c->FromDevice(lane.c_sync);
    profile.sync_seconds += SecondsSince(start);
  }

  // A row t = bf16(RMS(x_t) * w), x rows `stride` apart; with `keep`, also
  // the fp32 result.
  void NormToA(Lane& lane, const float* x, std::size_t stride,
               const std::vector<float>& w, float* keep) {
    const int d = config.hidden;
    auto* bits = lane.a->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      const float* row = x + t * stride;
      const float inv =
          1.0f /
          std::sqrt(Dot(row, row, d) / static_cast<float>(d) + config.rms_eps);
      simd::ScaleToBf16Row(row, inv, w.data(), bits + t * d, d);
      if (keep != nullptr) {
        RmsNorm(row, w.data(), d, config.rms_eps, keep + t * d);
      }
    });
  }

  // A row t = bf16(x_t), `width` values.
  void ToA(Lane& lane, const float* x, int width) {
    auto* bits = lane.a->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      simd::ToBf16Row(x + t * width, bits + t * width, width);
    });
  }

  // C rows (`width` bf16 each) to fp32 `out`; with `add`, onto it.
  void FromC(Lane& lane, int width, float* out, bool add) {
    const auto* bits = lane.c->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      simd::FromBf16Row(bits + t * width, out + t * width, width, add);
    });
  }

  // C row = [gate | up] (2 ffn bf16) -> A row = silu(gate) * up.
  void SwiGlu(Lane& lane, int ffn) {
    const auto* cb = lane.c->as<std::uint16_t>();
    auto* bits = lane.a->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      const std::uint16_t* gate = cb + t * 2 * ffn;
      simd::SwiGluBf16Row(gate, gate + ffn, bits + t * ffn, ffn);
    });
  }

  // cos/sin of each RoPE frequency at the lane's positions, as the decoder
  // computes them.
  void RopeTable(Lane& lane) {
    const int half = config.rope_dim / 2;
    for (int t = 0; t < lane.n; ++t) {
      for (int i = 0; i < half; ++i) {
        const double angle =
            static_cast<double>(lane.position + t) * rope_inv_freq[i];
        lane.rope_cos[t * half + i] = static_cast<float>(std::cos(angle));
        lane.rope_sin[t * half + i] = static_cast<float>(std::sin(angle));
      }
    }
  }

  void Rope(const Lane& lane, float* x, int t) const {
    const int half = config.rope_dim / 2;
    const float* cs = lane.rope_cos.data() + t * half;
    const float* sn = lane.rope_sin.data() + t * half;
    for (int i = 0; i < half; ++i) {
      const float a0 = x[i];
      const float b0 = x[i + half];
      x[i] = a0 * cs[i] - b0 * sn[i];
      x[i + half] = a0 * sn[i] + b0 * cs[i];
    }
  }

  // Throws before any work (as the decoder does) when an id is not in the
  // vocabulary; the jobs index the embedding table with them.
  void CheckTokens(const int* tokens, int count) const {
    for (int t = 0; t < count; ++t) {
      if (tokens[t] < 0 || tokens[t] >= config.vocab) {
        throw std::out_of_range("qwen35: token id out of range");
      }
    }
  }

  void Embed(Lane& lane, const int* tokens) {
    const int d = config.hidden;
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      quant::Dequantize(embed->type,
                        embed->data + static_cast<std::size_t>(tokens[t]) *
                                          quant::RowBytes(embed->type, d),
                        lane.h.data() + t * d, d);
    });
    RopeTable(lane);
  }

  // Normalizes and rotates the lane's keys (rows of `in`, kv after the q
  // and gate values) and appends them and the values to `st`'s cache, as
  // fp16 (the decoder's cache format).
  void AppendKv(Lane& lane, const Layer& l, std::size_t in_stride,
                DecodeState::Layer& st) {
    const Config& cfg = config;
    const int hd = cfg.head_dim;
    const std::size_t kv = static_cast<std::size_t>(cfg.kv_heads) * hd;
    const std::size_t base = st.keys.size();
    st.keys.resize(base + lane.n * kv);
    st.values.resize(base + lane.n * kv);
    pool.Run(static_cast<std::size_t>(lane.n), [&](std::size_t t) {
      const float* k = lane.in.data() + t * in_stride + 2 * cfg.heads * hd;
      const float* v = k + kv;
      thread_local std::vector<float> key;
      key.resize(hd);
      Half* kd = st.keys.data() + base + t * kv;
      for (int kh = 0; kh < cfg.kv_heads; ++kh) {
        RmsNorm(k + kh * hd, l.k_norm.data(), hd, cfg.rms_eps, key.data());
        Rope(lane, key.data(), static_cast<int>(t));
        vec::ToHalf(key.data(), kd + kh * hd, hd);
      }
      vec::ToHalf(v, st.values.data() + base + t * kv, static_cast<int>(kv));
    });
  }

  // Gated attention of the lane's tokens into `mixed`: causal over the cache
  // and the lane.
  void Attention(Lane& lane, const Layer& l, DecodeState::Layer& st) {
    const Config& cfg = config;
    const int n = lane.n;
    const int hd = cfg.head_dim;
    const int heads = cfg.heads;
    const int group = heads / cfg.kv_heads;
    const std::size_t in_stride = l.in->rows();
    const std::size_t kv = static_cast<std::size_t>(cfg.kv_heads) * hd;
    // Keys cached before the lane's tokens.
    const int past = static_cast<int>(st.keys.size() / kv);
    AppendKv(lane, l, in_stride, st);
    // Queries: q_norm, RoPE, and 1/sqrt(hd) (a power of two: exact).
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    pool.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
      for (int qh = 0; qh < heads; ++qh) {
        const float* raw = lane.in.data() + t * in_stride + qh * 2 * hd;
        float* q = lane.queries.data() + (t * heads + qh) * hd;
        RmsNorm(raw, l.q_norm.data(), hd, cfg.rms_eps, q);
        Rope(lane, q, static_cast<int>(t));
        for (int e = 0; e < hd; ++e) {
          q[e] *= scale;
        }
      }
    });
    // Work items: (kv head, block of query tokens), each with that kv head's
    // query heads, softmax online over key blocks.
    const int blocks = (n + kQueryBlock - 1) / kQueryBlock;
    const int out_stride = heads * hd;
    pool.Run(
        static_cast<std::size_t>(blocks * cfg.kv_heads), [&](std::size_t item) {
          const int g = static_cast<int>(item) % cfg.kv_heads;
          const int t0 = static_cast<int>(item) / cfg.kv_heads * kQueryBlock;
          const int t1 = std::min(n, t0 + kQueryBlock);
          const int rows = (t1 - t0) * group;
          thread_local std::vector<float> o, m, sum, p;
          o.assign(static_cast<std::size_t>(rows) * hd, 0.0f);
          m.assign(rows, -INFINITY);
          sum.assign(rows, 0.0f);
          p.resize(static_cast<std::size_t>(rows) * kKeyBlock);
          const Half* keys = st.keys.data() + g * hd;
          const Half* values = st.values.data() + g * hd;
          const int last = past + t1;  // keys any row of the item sees
          for (int jb = 0; jb < last; jb += kKeyBlock) {
            for (int t = t0; t < t1; ++t) {
              const int limit = std::min(past + t + 1, jb + kKeyBlock);
              if (limit <= jb) {
                continue;
              }
              const int count = limit - jb;
              const int r0 = (t - t0) * group;
              const float* q =
                  lane.queries.data() +
                  (static_cast<std::size_t>(t) * heads + g * group) * hd;
              float* pr = p.data() + static_cast<std::size_t>(r0) * kKeyBlock;
              Scores4(q, hd, keys + static_cast<std::size_t>(jb) * kv, kv,
                      count, pr, kKeyBlock);
              for (int r = 0; r < group; ++r) {
                float* s = pr + r * kKeyBlock;
                float mx = m[r0 + r];
                for (int j = 0; j < count; ++j) {
                  mx = std::max(mx, s[j]);
                }
                const float correction = std::exp(m[r0 + r] - mx);
                const float total = simd::ExpShiftSum(s, mx, s, count);
                if (correction != 1.0f) {
                  float* orow =
                      o.data() + static_cast<std::size_t>(r0 + r) * hd;
                  for (int e = 0; e < hd; ++e) {
                    orow[e] *= correction;
                  }
                }
                sum[r0 + r] = sum[r0 + r] * correction + total;
                m[r0 + r] = mx;
              }
              AccumulateValues4(
                  pr, kKeyBlock, values + static_cast<std::size_t>(jb) * kv, kv,
                  count, hd, o.data() + static_cast<std::size_t>(r0) * hd);
            }
          }
          for (int t = t0; t < t1; ++t) {
            for (int r = 0; r < group; ++r) {
              const int qh = g * group + r;
              const float* orow =
                  o.data() +
                  static_cast<std::size_t>((t - t0) * group + r) * hd;
              const float inv = 1.0f / sum[(t - t0) * group + r];
              const float* gate =
                  lane.in.data() + t * in_stride + qh * 2 * hd + hd;
              float* out = lane.mixed.data() +
                           static_cast<std::size_t>(t) * out_stride + qh * hd;
              for (int e = 0; e < hd; ++e) {
                out[e] = orow[e] * inv * Sigmoid(gate[e]);
              }
            }
          }
        });
  }

  // Gated DeltaNet of the lane's tokens into `mixed`; `normed` holds the
  // layer's normalized input.
  void DeltaNet(Lane& lane, const Layer& l, DecodeState::Layer& st) {
    const Config& cfg = config;
    const int n = lane.n;
    const int d = cfg.hidden;
    const int hk = cfg.linear_key_heads;
    const int hv = cfg.linear_value_heads;
    const int dk = cfg.linear_head_dim;
    const int key_dim = cfg.LinearKeyDim();
    const int value_dim = cfg.LinearValueDim();
    const int channels = cfg.ConvChannels();
    const int window = cfg.conv_kernel - 1;
    const std::size_t in_stride = l.in->rows();  // qkv | z
    const float* in = lane.in.data();
    float* u = lane.u.data();
    float* gates = lane.gates.data();
    float* mixed = lane.mixed.data();

    // Per token and value head: the decay exp(g) and the write strength
    // beta, from the alpha and beta rows (fp32 on the CPU, as in decode).
    pool.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
      const float* x = lane.normed.data() + t * d;
      float* out = gates + t * 2 * hv;
      for (int head = 0; head < hv; ++head) {
        const float a =
            Dot(l.alpha.data() + static_cast<std::size_t>(head) * d, x, d);
        const float b =
            Dot(l.beta.data() + static_cast<std::size_t>(head) * d, x, d);
        out[head] = std::exp(l.ssm_a[head] * Softplus(a + l.dt_bias[head]));
        out[hv + head] = Sigmoid(b);
      }
    });

    // Causal depthwise conv over [3 previous inputs, current], then SiLU:
    // blocks of channels, vectorized across them, along the tokens.
    constexpr int kConvBlock = 128;
    pool.Run(
        static_cast<std::size_t>((channels + kConvBlock - 1) / kConvBlock),
        [&](std::size_t block) {
          const int c0 = static_cast<int>(block) * kConvBlock;
          const int width = std::min(kConvBlock, channels - c0);
          const float* x[8];
          for (int t = 0; t < n; ++t) {
            for (int s = 0; s <= window; ++s) {
              const int src = t - window + s;
              x[s] = src >= 0 ? in + src * in_stride + c0
                              : st.conv.data() +
                                    static_cast<std::size_t>(src + window) *
                                        channels +
                                    c0;
            }
            float* out = u + static_cast<std::size_t>(t) * channels + c0;
            const float* kw = l.conv.data() + c0;
            for (int i = 0; i < width; ++i) {
              float acc = kw[static_cast<std::size_t>(window) * channels + i] *
                          x[window][i];
              for (int s = 0; s < window; ++s) {
                acc += kw[static_cast<std::size_t>(s) * channels + i] * x[s][i];
              }
              out[i] = acc;
            }
            simd::SiluRow(out, out, width);
          }
        });
    // The window after these tokens: their last `window` inputs. Ascending
    // s reads older entries (src + window = n + s > s) before they are
    // overwritten.
    for (int s = 0; s < window; ++s) {
      const int src = n - window + s;
      float* dst = st.conv.data() + static_cast<std::size_t>(s) * channels;
      if (src >= 0) {
        std::copy_n(in + src * in_stride, channels, dst);
      } else {
        std::copy_n(
            st.conv.data() + static_cast<std::size_t>(src + window) * channels,
            channels, dst);
      }
    }

    // L2-normalize q and k per key head; scale q by 1/sqrt(dk).
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(dk));
    pool.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
      float* q = u + t * channels;
      float* k = q + key_dim;
      for (int head = 0; head < hk; ++head) {
        for (float* x : {q + head * dk, k + head * dk}) {
          const float inv = 1.0f / std::sqrt(Dot(x, x, dk) + 1e-6f);
          const float sc = x == q + head * dk ? inv * q_scale : inv;
          for (int e = 0; e < dk; ++e) {
            x[e] *= sc;
          }
        }
      }
    });

    // The delta rule along the tokens. Each value column of a head's state
    // evolves on its own (given k, q, v and the gates), so the jobs are
    // (value head, block of kDeltaColumns columns), four times as many as
    // heads to balance. A job copies its 16 KB slice of the state into an
    // aligned local block (no cache line shared with another job, all in
    // L1), runs every token, and copies it back.
    const int column_blocks = dk / kDeltaColumns;
    pool.Run(
        static_cast<std::size_t>(hv * column_blocks), [&](std::size_t job) {
          const int head = static_cast<int>(job) / column_blocks;
          const int j0 = static_cast<int>(job) % column_blocks * kDeltaColumns;
          const int kh = head % hk;  // GGUF order: value heads tile key heads
          float* sm =
              st.ssm.data() + static_cast<std::size_t>(head) * dk * dk + j0;
          alignas(64) float local[kMaxHeadDim * kDeltaColumns];
          for (int i = 0; i < dk; ++i) {
            std::copy_n(sm + static_cast<std::size_t>(i) * dk, kDeltaColumns,
                        local + i * kDeltaColumns);
          }
          for (int t = 0; t < n; ++t) {
            const float* ut = u + static_cast<std::size_t>(t) * channels;
            const float* g = gates + static_cast<std::size_t>(t) * 2 * hv;
            DeltaColumns(local, dk, kDeltaColumns, ut + key_dim + kh * dk,
                         ut + kh * dk, ut + 2 * key_dim + head * dk + j0,
                         g[head], g[hv + head],
                         mixed + static_cast<std::size_t>(t) * value_dim +
                             head * dk + j0);
          }
          for (int i = 0; i < dk; ++i) {
            std::copy_n(local + i * kDeltaColumns, kDeltaColumns,
                        sm + static_cast<std::size_t>(i) * dk);
          }
        });

    // Gated RMSNorm per head: RMS(o) * w * silu(z).
    pool.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
      for (int head = 0; head < hv; ++head) {
        float* yh = mixed + t * value_dim + head * dk;
        RmsNorm(yh, l.ssm_norm.data(), dk, cfg.rms_eps, yh);
        simd::MulSiluRow(yh, in + t * in_stride + channels + head * dk, dk);
      }
    });
  }

  // Seconds profiled so far, to charge the rest of a call to `other`.
  double Profiled() const {
    return profile.gemm_seconds + profile.sync_seconds +
           profile.deltanet_seconds + profile.attention_seconds;
  }

  // Runs the first `count` lanes (consecutive chunks, embedded) through
  // every layer. Each lane has at most one GEMM in flight; calls run in
  // start order, so while the CPU finishes one lane's step the NPU runs the
  // other lane's GEMM.
  void RunLanes(int count, DecodeState& state) {
    const Config& cfg = config;
    const int d = cfg.hidden;
    for (int il = 0; il < cfg.layers; ++il) {
      Layer& l = layers[il];
      auto& st = state.layers[il];
      for (int i = 0; i < count; ++i) {
        Lane& lane = lanes[i];
        if (il > 0) {
          Finish(lane);  // the previous layer's down projection
          FromC(lane, d, lane.h.data(), true);
        }
        NormToA(lane, lane.h.data(), d, l.attn_norm,
                l.attention ? nullptr : lane.normed.data());
        Start(*l.in, lane);
      }
      for (int i = 0; i < count; ++i) {
        Lane& lane = lanes[i];
        Finish(lane);
        FromC(lane, static_cast<int>(l.in->rows()), lane.in.data(), false);
        const auto mixer_start = Clock::now();
        if (l.attention) {
          Attention(lane, l, st);
          profile.attention_seconds += SecondsSince(mixer_start);
        } else {
          DeltaNet(lane, l, st);
          profile.deltanet_seconds += SecondsSince(mixer_start);
        }
        ToA(lane, lane.mixed.data(), static_cast<int>(l.out->dim()));
        Start(*l.out, lane);
      }
      for (int i = 0; i < count; ++i) {
        Lane& lane = lanes[i];
        Finish(lane);
        FromC(lane, d, lane.h.data(), true);
        NormToA(lane, lane.h.data(), d, l.post_norm, nullptr);
        Start(*l.ffn, lane);
      }
      for (int i = 0; i < count; ++i) {
        Lane& lane = lanes[i];
        Finish(lane);
        SwiGlu(lane, static_cast<int>(l.ffn->rows() / 2));
        Start(*l.down, lane);
      }
    }
    for (int i = 0; i < count; ++i) {
      Finish(lanes[i]);
      FromC(lanes[i], d, lanes[i].h.data(), true);
    }
  }

  // Runs `count` tokens, up to two chunks at a time, in the passes
  // PlanPasses picks.
  void Run(const int* tokens, int count, DecodeState& state, float* hidden) {
    const int d = config.hidden;
    CheckTokens(tokens, count);
    int done = 0;
    for (const Pass& pass :
         PlanPasses(count, 2, flops_per_token, kCpuSecondsPerToken)) {
      const auto start = Clock::now();
      const double before = Profiled();
      int used = 0;
      int taken = 0;
      for (int i = 0; i < pass.lanes; ++i) {
        Lane& lane = lanes[i];
        lane.m = pass.m;
        lane.n = std::min(pass.m, pass.tokens - taken);
        lane.position = state.position + taken;
        Embed(lane, tokens + done + taken);
        taken += lane.n;
        ++used;
      }
      RunLanes(used, state);
      state.position += taken;
      if (hidden != nullptr) {
        int row = done;
        for (int i = 0; i < used; ++i) {
          std::copy_n(lanes[i].h.data(),
                      static_cast<std::size_t>(lanes[i].n) * d,
                      hidden + static_cast<std::size_t>(row) * d);
          row += lanes[i].n;
        }
      }
      done += taken;
      profile.other_seconds += SecondsSince(start) - (Profiled() - before);
    }
  }

  void FillMtpChunk(const float* hidden, const int* tokens, const Pass& pass,
                    int position, DecodeState& state) {
    const Config& cfg = config;
    const int d = cfg.hidden;
    const int n = pass.tokens;
    const auto start = Clock::now();
    const double before = Profiled();
    Lane& lane = lanes[0];
    lane.n = n;
    lane.m = pass.m;
    lane.position = position;
    // [RMS(embed(token)) * enorm ; RMS(RMS(h) * output_norm) * hnorm].
    auto* bits = lane.a->as<std::uint16_t>();
    pool.Run(static_cast<std::size_t>(n), [&](std::size_t t) {
      thread_local std::vector<float> x, y;
      x.resize(d);
      y.resize(d);
      quant::Dequantize(embed->type,
                        embed->data + static_cast<std::size_t>(tokens[t]) *
                                          quant::RowBytes(embed->type, d),
                        x.data(), d);
      RmsNorm(x.data(), enorm.data(), d, cfg.rms_eps, y.data());
      std::uint16_t* row = bits + t * 2 * d;
      simd::ToBf16Row(y.data(), row, d);
      RmsNorm(hidden + t * d, output_norm.data(), d, cfg.rms_eps, x.data());
      RmsNorm(x.data(), hnorm.data(), d, cfg.rms_eps, y.data());
      simd::ToBf16Row(y.data(), row + d, d);
    });
    Start(*eh_proj, lane);
    Finish(lane);
    FromC(lane, d, lane.h.data(), false);
    NormToA(lane, lane.h.data(), d, mtp.attn_norm, nullptr);
    Start(*mtp.in, lane);
    Finish(lane);
    FromC(lane, static_cast<int>(mtp.in->rows()), lane.in.data(), false);
    RopeTable(lane);
    AppendKv(lane, mtp, mtp.in->rows(), state.layers[cfg.layers]);
    profile.other_seconds += SecondsSince(start) - (Profiled() - before);
  }
};

NpuPrefill::NpuPrefill(const gguf::File& file,
                       const std::filesystem::path& kernel_dir)
    : impl_(std::make_unique<Impl>(file, kernel_dir)) {}

NpuPrefill::~NpuPrefill() = default;

const Config& NpuPrefill::config() const {
  return impl_->config;
}

void NpuPrefill::Run(const int* tokens, int count, DecodeState& state,
                     float* hidden) {
  const BurstPool::Burst burst(impl_->pool);
  impl_->Run(tokens, count, state, hidden);
}

void NpuPrefill::FillMtp(const float* hidden, const int* tokens, int count,
                         int position, DecodeState& mtp) {
  if (!impl_->has_mtp) {
    throw std::runtime_error("qwen35: model has no MTP layer");
  }
  const int d = impl_->config.hidden;
  impl_->CheckTokens(tokens, count);
  const BurstPool::Burst burst(impl_->pool);
  // One lane, and almost no CPU work between its two GEMMs.
  int done = 0;
  for (const Pass& pass :
       PlanPasses(count, 1, impl_->mtp_flops_per_token, 0.0)) {
    impl_->FillMtpChunk(hidden + static_cast<std::size_t>(done) * d,
                        tokens + done, pass, position + done, mtp);
    done += pass.tokens;
  }
}

const NpuPrefill::Profile& NpuPrefill::profile() const {
  return impl_->profile;
}

PromptPrefill PrefillPrompt(NpuPrefill& prefill) {
  return [&prefill](const std::vector<int>& prompt, DecodeState& state,
                    DecodeState* mtp, float* hidden) {
    // A prefill pass costs ~0.10 s however short the prompt (98 GEMM calls
    // of at least 256 rows each), the decode path ~28 ms per two tokens.
    // Measured TTFT (BENCHMARKS.md): 6 tokens 0.086 s on the decode path
    // against 0.102 s with prefill, 8 tokens 0.115 against 0.102. Shorter
    // prompts go through the decoder alone.
    constexpr int kMinPrefillPrompt = 7;
    const int k = static_cast<int>(prompt.size()) - 1;
    if (static_cast<int>(prompt.size()) < kMinPrefillPrompt) {
      return 0;
    }
    // Every id, including the last (the MTP cache reads prompt[k]), before
    // any state changes.
    for (const int id : prompt) {
      if (id < 0 || id >= prefill.config().vocab) {
        throw std::out_of_range("qwen35: token id out of range");
      }
    }
    prefill.Run(prompt.data(), k, state, hidden);
    if (mtp != nullptr) {
      // Pairs (h[p - 1], prompt[p]) at p = 1 .. k.
      prefill.FillMtp(hidden, prompt.data() + 1, k, 1, *mtp);
    }
    return k;
  };
}

}  // namespace gufo_npu::qwen35
