#include "models/qwen3vl_embed_2b/npu_vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "core/burst_pool.hpp"
#include "core/quant.hpp"
#include "core/simd.hpp"
#include "models/qwen3vl_embed_2b/npu/embed_gemm.hpp"
#include "models/qwen3vl_embed_2b/npu/gemm_pass.hpp"
#include "models/qwen3vl_embed_2b/npu/vit_attention.hpp"
#include "models/qwen3vl_embed_2b/ops.hpp"
#include "npu/gemm.hpp"

namespace gufo_npu::qwen3vl_embed_2b {
namespace {

using gguf::TensorInfo;
using gufo_npu::npu::Bfp16Gemm;
using gufo_npu::npu::GemmRuntime;
using npu::Chunk;
using Clock = std::chrono::steady_clock;
using Bits = std::uint16_t;  // bf16

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<float> ToFloats(const TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

// gelu_pytorch_tanh, 0.5 x (1 + tanh(u)), as x sigmoid(2u) with u =
// sqrt(2 / pi) (x + 0.044715 x^3).
constexpr float kGeluTanhScale = 0.7978845608028654f;
constexpr float kGeluTanhCubic = 0.044715f;

float GeluTanh(float x) {
  const float u = kGeluTanhScale * (x + kGeluTanhCubic * x * x * x);
  return x / (1.0f + std::exp(-2.0f * u));
}

float GeluErf(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
}

#if defined(__AVX2__) && defined(__FMA__)
float HorizontalSum(__m256 v) {
  float lanes[8];
  _mm256_storeu_ps(lanes, v);
  float total = 0;
  for (const float lane : lanes) {
    total += lane;
  }
  return total;
}
#endif

// out = bf16(LayerNorm(x) * w + b) over n values (mean, then the centred
// variance, in two passes).
void LayerNormToBf16(const float* x, int n, const float* w, const float* b,
                     float eps, Bits* out) {
  float sum = 0;
  int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  __m256 acc = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    acc = _mm256_add_ps(acc, _mm256_loadu_ps(x + i));
  }
  sum = HorizontalSum(acc);
#endif
  for (; i < n; ++i) {
    sum += x[i];
  }
  const float mean = sum / static_cast<float>(n);
  float var = 0;
  i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  const __m256 m = _mm256_set1_ps(mean);
  acc = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    const __m256 c = _mm256_sub_ps(_mm256_loadu_ps(x + i), m);
    acc = _mm256_fmadd_ps(c, c, acc);
  }
  var = HorizontalSum(acc);
#endif
  for (; i < n; ++i) {
    const float c = x[i] - mean;
    var += c * c;
  }
  const float inv = 1.0f / std::sqrt(var / static_cast<float>(n) + eps);
  i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  const __m256 s = _mm256_set1_ps(inv);
  for (; i + 8 <= n; i += 8) {
    const __m256 c = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), m), s);
    simd::StoreBf16x8(out + i, _mm256_fmadd_ps(c, _mm256_loadu_ps(w + i),
                                               _mm256_loadu_ps(b + i)));
  }
#endif
  for (; i < n; ++i) {
    out[i] = simd::ToBf16((x[i] - mean) * inv * w[i] + b[i]);
  }
}

// dst = c + bias, or dst += c + bias with `add` (c bf16).
void AddBias(const Bits* c, const float* bias, float* dst, int n, bool add) {
  int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  for (; i + 8 <= n; i += 8) {
    __m256 v =
        _mm256_add_ps(simd::LoadBf16x8(c + i), _mm256_loadu_ps(bias + i));
    if (add) {
      v = _mm256_add_ps(v, _mm256_loadu_ps(dst + i));
    }
    _mm256_storeu_ps(dst + i, v);
  }
#endif
  for (; i < n; ++i) {
    const float v = simd::FromBf16(c[i]) + bias[i];
    dst[i] = add ? dst[i] + v : v;
  }
}

// out = bf16(gelu_tanh(c + bias)) (c bf16).
void BiasGeluTanhToBf16(const Bits* c, const float* bias, Bits* out, int n) {
  int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
  const __m256 k = _mm256_set1_ps(-2.0f * kGeluTanhScale);
  const __m256 k3 = _mm256_set1_ps(kGeluTanhCubic);
  const __m256 one = _mm256_set1_ps(1.0f);
  for (; i + 8 <= n; i += 8) {
    const __m256 x =
        _mm256_add_ps(simd::LoadBf16x8(c + i), _mm256_loadu_ps(bias + i));
    const __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    const __m256 e = simd::Exp8(_mm256_mul_ps(k, _mm256_fmadd_ps(k3, x3, x)));
    simd::StoreBf16x8(out + i, _mm256_div_ps(x, _mm256_add_ps(one, e)));
  }
#endif
  for (; i < n; ++i) {
    out[i] = simd::ToBf16(GeluTanh(simd::FromBf16(c[i]) + bias[i]));
  }
}

// out = bf16(gelu_erf(c + bias)) (c bf16).
void BiasGeluErfToBf16(const Bits* c, const float* bias, Bits* out, int n) {
  for (int i = 0; i < n; ++i) {
    out[i] = simd::ToBf16(GeluErf(simd::FromBf16(c[i]) + bias[i]));
  }
}

}  // namespace

struct NpuVision::Impl {
  struct Block {
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
    std::vector<float> qkv_b, out_b, up_b, down_b;
    std::unique_ptr<Bfp16Gemm> qkv, out, up, down;
  };
  struct Merger {
    std::vector<float> norm_w, norm_b, fc1_b, fc2_b;
    std::unique_ptr<Bfp16Gemm> fc1, fc2;
  };

  VisionConfig config;
  GemmRuntime& runtime;
  // The attention program, in its own hardware context.
  npu::VitAttention attention;
  // hidden x kPatchDim: the patch kernel with its temporal slots summed,
  // K zero-padded from 3 * patch^2.
  std::unique_ptr<Bfp16Gemm> patch;
  std::vector<float> patch_b;
  std::vector<float> pos_embed;  // pos_side^2 x hidden
  std::vector<float> inv_freq;   // head_dim / 4 values
  std::vector<Block> blocks;
  Merger merger;
  std::vector<Merger> deepstack;
  std::optional<npu::GemmPass> pass;
  // CPU workers, spinning for the length of an Encode (as the embedder's).
  BurstPool pool{
      std::max<std::size_t>(std::thread::hardware_concurrency() / 2, 1) - 1};
  Profile profile;

  // One image: residual stream, qkv (fp32, rotated in place), fc1 output
  // after GELU (bf16, fc2's A), a merger's fc1 output after GELU (bf16),
  // each patch's 2D RoPE angles and grid position.
  std::vector<float> h, qkv;
  std::vector<Bits> act, mid;
  std::vector<float> rcos, rsin;
  std::vector<int> prow, pcol;
  std::vector<Chunk> patch_chunks, token_chunks;

  static void CheckShapes(const VisionConfig& c) {
    if (static_cast<std::size_t>(c.hidden) != npu::kVitHidden ||
        static_cast<std::size_t>(c.ffn) != npu::kVitFfn ||
        static_cast<std::size_t>(c.MergedDim()) != npu::kVitMerged ||
        static_cast<std::size_t>(c.out_hidden) != npu::kHidden ||
        static_cast<std::size_t>(c.PatchValues()) > npu::kPatchDim ||
        c.heads != npu::kVitAttnHeads ||
        c.hidden != c.heads * npu::kVitAttnHeadDim) {
      throw std::runtime_error(
          "qwen3vl npu vision: this mmproj's shapes (hidden " +
          std::to_string(c.hidden) + ", ffn " + std::to_string(c.ffn) +
          ", output " + std::to_string(c.out_hidden) + ", " +
          std::to_string(c.heads) +
          " heads) are not Qwen3-VL-Embedding-2B's, which the GEMM and "
          "attention programs are built for");
    }
  }

  std::unique_ptr<Bfp16Gemm> Gemm(const TensorInfo& t) {
    const auto w = ToFloats(t);
    return std::make_unique<Bfp16Gemm>(
        runtime, w.data(), static_cast<std::size_t>(t.Rows()),
        static_cast<std::size_t>(t.RowElements()));
  }

  Merger LoadMerger(const gguf::File& file, const std::string& norm,
                    const std::string& fc1, const std::string& fc2) {
    Merger m;
    m.norm_w = ToFloats(file.Tensor(norm + ".weight"));
    m.norm_b = ToFloats(file.Tensor(norm + ".bias"));
    m.fc1 = Gemm(file.Tensor(fc1 + ".weight"));
    m.fc1_b = ToFloats(file.Tensor(fc1 + ".bias"));
    m.fc2 = Gemm(file.Tensor(fc2 + ".weight"));
    m.fc2_b = ToFloats(file.Tensor(fc2 + ".bias"));
    return m;
  }

  Impl(const gguf::File& file, GemmRuntime& rt,
       const std::filesystem::path& kernel_dir)
      : config(ReadVisionConfig(file)), runtime(rt), attention(kernel_dir) {
    const VisionConfig& c = config;
    CheckShapes(c);
    const int pv = c.PatchValues();
    // As VisionReference: W0 + W1, or a single 2D kernel doubled (the image
    // fills both temporal slots).
    std::vector<float> w = ToFloats(file.Tensor("v.patch_embd.weight"));
    if (const TensorInfo* second = file.FindTensor("v.patch_embd.weight.1")) {
      const auto w1 = ToFloats(*second);
      if (w1.size() != w.size()) {
        throw std::runtime_error("qwen3vl mmproj: patch embedding shapes");
      }
      for (std::size_t i = 0; i < w1.size(); ++i) {
        w[i] += w1[i];
      }
    } else {
      for (float& v : w) {
        v *= 2.0f;
      }
    }
    if (w.size() != static_cast<std::size_t>(c.hidden) * pv) {
      throw std::runtime_error("qwen3vl mmproj: patch embedding shape");
    }
    std::vector<float> padded(
        static_cast<std::size_t>(c.hidden) * npu::kPatchDim, 0.0f);
    for (int o = 0; o < c.hidden; ++o) {
      std::copy_n(w.data() + static_cast<std::size_t>(o) * pv, pv,
                  padded.data() + static_cast<std::size_t>(o) * npu::kPatchDim);
    }
    patch = std::make_unique<Bfp16Gemm>(runtime, padded.data(),
                                        static_cast<std::size_t>(c.hidden),
                                        npu::kPatchDim);
    patch_b = ToFloats(file.Tensor("v.patch_embd.bias"));
    pos_embed = ToFloats(file.Tensor("v.position_embd.weight"));
    const int quarter = c.HeadDim() / 4;
    inv_freq.resize(quarter);
    for (int j = 0; j < quarter; ++j) {
      inv_freq[j] = static_cast<float>(
          1.0 / std::pow(static_cast<double>(c.rope_theta),
                         static_cast<double>(2 * j) / (2.0 * quarter)));
    }
    for (int il = 0; il < c.layers; ++il) {
      const std::string p = "v.blk." + std::to_string(il) + ".";
      auto t = [&](const char* name) -> const TensorInfo& {
        return file.Tensor(p + name);
      };
      Block b;
      b.ln1_w = ToFloats(t("ln1.weight"));
      b.ln1_b = ToFloats(t("ln1.bias"));
      b.ln2_w = ToFloats(t("ln2.weight"));
      b.ln2_b = ToFloats(t("ln2.bias"));
      b.qkv = Gemm(t("attn_qkv.weight"));
      b.qkv_b = ToFloats(t("attn_qkv.bias"));
      b.out = Gemm(t("attn_out.weight"));
      b.out_b = ToFloats(t("attn_out.bias"));
      b.up = Gemm(t("ffn_up.weight"));
      b.up_b = ToFloats(t("ffn_up.bias"));
      b.down = Gemm(t("ffn_down.weight"));
      b.down_b = ToFloats(t("ffn_down.bias"));
      if (b.qkv->rows() != static_cast<std::size_t>(3 * c.hidden) ||
          b.up->rows() != static_cast<std::size_t>(c.ffn)) {
        throw std::runtime_error("qwen3vl mmproj: block shapes");
      }
      blocks.push_back(std::move(b));
    }
    merger = LoadMerger(file, "v.post_ln", "mm.0", "mm.2");
    for (const int il : c.deepstack) {
      const std::string p = "v.deepstack." + std::to_string(il) + ".";
      deepstack.push_back(LoadMerger(file, p + "norm", p + "fc1", p + "fc2"));
    }
    std::vector<const Bfp16Gemm*> gemms = {patch.get()};
    for (const Block& b : blocks) {
      for (const auto* g : {&b.qkv, &b.out, &b.up, &b.down}) {
        gemms.push_back(g->get());
      }
    }
    gemms.push_back(merger.fc1.get());
    gemms.push_back(merger.fc2.get());
    for (const Merger& m : deepstack) {
      gemms.push_back(m.fc1.get());
      gemms.push_back(m.fc2.get());
    }
    pass.emplace(runtime, gemms);
  }

  void Run(std::size_t n, const std::function<void(std::size_t)>& fn) {
    pool.Run(n, fn);
  }

  // A rows = bf16(LayerNorm(h row) * w + b).
  auto NormRows(const std::vector<float>& w, const std::vector<float>& b) {
    return [this, &w, &b](const Chunk& ch, Bits* a) {
      const int d = config.hidden;
      Run(static_cast<std::size_t>(ch.n), [&](std::size_t t) {
        LayerNormToBf16(h.data() + (ch.begin + t) * d, d, w.data(), b.data(),
                        config.eps, a + t * d);
      });
    };
  }

  // C rows + bias onto (add) or into fp32 rows of `dst`.
  auto BiasRows(std::vector<float>& dst, const std::vector<float>& bias,
                int width, bool add) {
    return [this, &dst, &bias, width, add](const Chunk& ch, const Bits* cb) {
      Run(static_cast<std::size_t>(ch.n), [&](std::size_t t) {
        AddBias(cb + t * width, bias.data(),
                dst.data() + (ch.begin + t) * width, width, add);
      });
    };
  }

  // A rows = a copy of `src`'s rows (bf16, `width` each).
  auto CopyRows(const std::vector<Bits>& src, int width) {
    return [&src, width](const Chunk& ch, Bits* a) {
      std::memcpy(a, src.data() + static_cast<std::size_t>(ch.begin) * width,
                  static_cast<std::size_t>(ch.n) * width * sizeof(Bits));
    };
  }

  // Each patch's grid position and 2D RoPE angles: pair i < head_dim / 4
  // turns by its row, the rest by its column.
  void Positions(const Patches& patches) {
    const VisionConfig& c = config;
    const auto rows = static_cast<std::size_t>(patches.Count());
    const int half = c.HeadDim() / 2;
    const int quarter = c.HeadDim() / 4;
    prow.resize(rows);
    pcol.resize(rows);
    rcos.resize(rows * half);
    rsin.resize(rows * half);
    Run(rows, [&](std::size_t t) {
      PatchPosition(static_cast<int>(t), patches.grid_w, c.merge, &prow[t],
                    &pcol[t]);
      for (int i = 0; i < half; ++i) {
        const float angle =
            i < quarter ? static_cast<float>(prow[t]) * inv_freq[i]
                        : static_cast<float>(pcol[t]) * inv_freq[i - quarter];
        rcos[t * half + i] = std::cos(angle);
        rsin[t * half + i] = std::sin(angle);
      }
    });
  }

  // out += the learned position embedding resampled to patch t of the
  // grid (4 bilinear taps).
  void AddPositions(std::size_t t, const Patches& patches, float* out) const {
    const VisionConfig& c = config;
    const int d = c.hidden;
    int th[2], tw[2];
    float wh[2], ww[2];
    InterpolationTaps(prow[t], patches.grid_h, c.pos_side, th, wh);
    InterpolationTaps(pcol[t], patches.grid_w, c.pos_side, tw, ww);
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        const float w = wh[a] * ww[b];
        const float* src =
            pos_embed.data() +
            (static_cast<std::size_t>(th[a]) * c.pos_side + tw[b]) * d;
        for (int i = 0; i < d; ++i) {
          out[i] += w * src[i];
        }
      }
    }
  }

  // q and k of patch t's qkv row rotated by its 2D RoPE, q scaled.
  void Rotate(std::size_t t, float* row) const {
    const VisionConfig& c = config;
    const int d = c.hidden;
    const int hd = c.HeadDim();
    const int half = hd / 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const float* cs = rcos.data() + t * half;
    const float* sn = rsin.data() + t * half;
    for (int part = 0; part < 2; ++part) {  // q, k
      for (int head = 0; head < c.heads; ++head) {
        float* v = row + part * d + head * hd;
        for (int i = 0; i < half; ++i) {
          const float a = v[i];
          const float b = v[i + half];
          v[i] = a * cs[i] - b * sn[i];
          v[i + half] = b * cs[i] + a * sn[i];
        }
        if (part == 0) {
          for (int e = 0; e < hd; ++e) {
            v[e] *= scale;
          }
        }
      }
    }
  }

  // A merger over the image's tokens (merge^2 consecutive rows of h each):
  // LayerNorm per patch (main merger) or per merged row (DeepStack,
  // `postshuffle`), fc1, exact GELU, fc2.
  std::vector<float> Merge(const Merger& m, int tokens, bool postshuffle) {
    const VisionConfig& c = config;
    const int d = c.hidden;
    const int md = c.MergedDim();
    const int parts = c.merge * c.merge;
    const int oh = c.out_hidden;
    pass->Run(
        *m.fc1, token_chunks,
        [&](const Chunk& ch, Bits* a) {
          Run(static_cast<std::size_t>(ch.n), [&](std::size_t t) {
            const float* src = h.data() + (ch.begin + t) * md;
            Bits* dst = a + t * md;
            if (postshuffle) {
              LayerNormToBf16(src, md, m.norm_w.data(), m.norm_b.data(), c.eps,
                              dst);
              return;
            }
            for (int p = 0; p < parts; ++p) {
              LayerNormToBf16(src + p * d, d, m.norm_w.data(), m.norm_b.data(),
                              c.eps, dst + p * d);
            }
          });
        },
        [&](const Chunk& ch, const Bits* cb) {
          Run(static_cast<std::size_t>(ch.n), [&](std::size_t t) {
            BiasGeluErfToBf16(cb + t * md, m.fc1_b.data(),
                              mid.data() + (ch.begin + t) * md, md);
          });
        });
    std::vector<float> out(static_cast<std::size_t>(tokens) * oh);
    pass->Run(*m.fc2, token_chunks, CopyRows(mid, md),
              BiasRows(out, m.fc2_b, oh, false));
    return out;
  }

  ImageFeatures Encode(const Patches& patches) {
    const VisionConfig& c = config;
    if (patches.Count() <= 0 || patches.grid_h % c.merge != 0 ||
        patches.grid_w % c.merge != 0 ||
        patches.values.size() !=
            static_cast<std::size_t>(patches.Count()) * c.PatchValues()) {
      throw std::invalid_argument("qwen3vl: bad patch grid");
    }
    const auto start = Clock::now();
    const BurstPool::Burst burst(pool);
    const int n = patches.Count();
    const int d = c.hidden;
    const int ffn = c.ffn;
    const int pv = c.PatchValues();
    const int tokens = n / (c.merge * c.merge);
    const auto rows = static_cast<std::size_t>(n);
    h.resize(rows * d);
    qkv.resize(rows * 3 * d);
    act.resize(rows * ffn);
    mid.resize(static_cast<std::size_t>(tokens) * c.MergedDim());
    patch_chunks = npu::Chunks(n);
    token_chunks = npu::Chunks(tokens);
    attention.Reset(n);
    Positions(patches);

    // Patch embedding + bias + resampled learned positions.
    pass->Run(
        *patch, patch_chunks,
        [&](const Chunk& ch, Bits* a) {
          Run(static_cast<std::size_t>(ch.n), [&](std::size_t t) {
            Bits* row = a + t * npu::kPatchDim;
            simd::ToBf16Row(patches.values.data() + (ch.begin + t) * pv, row,
                            pv);
            std::fill(row + pv, row + npu::kPatchDim, Bits{0});
          });
        },
        [&](const Chunk& ch, const Bits* cb) {
          Run(static_cast<std::size_t>(ch.n), [&](std::size_t t) {
            const std::size_t r = ch.begin + t;
            float* out = h.data() + r * d;
            AddBias(cb + t * d, patch_b.data(), out, d, false);
            AddPositions(r, patches, out);
          });
        });

    ImageFeatures f;
    f.rows = patches.grid_h / c.merge;
    f.cols = patches.grid_w / c.merge;
    const ParallelRun run = [this](std::size_t count,
                                   const std::function<void(std::size_t)>& fn) {
      Run(count, fn);
    };
    for (int il = 0; il < c.layers; ++il) {
      const Block& b = blocks[il];
      pass->Run(*b.qkv, patch_chunks, NormRows(b.ln1_w, b.ln1_b),
                [&](const Chunk& ch, const Bits* cb) {
                  Run(static_cast<std::size_t>(ch.n), [&](std::size_t t) {
                    const std::size_t r = ch.begin + t;
                    float* row = qkv.data() + r * 3 * d;
                    AddBias(cb + t * 3 * d, b.qkv_b.data(), row, 3 * d, false);
                    Rotate(r, row);
                  });
                });
      const auto attention_start = Clock::now();
      attention.Run(qkv.data(), run);
      profile.attention_seconds += SecondsSince(attention_start);
      pass->Run(
          *b.out, patch_chunks,
          [&](const Chunk& ch, Bits* a) {
            Run(static_cast<std::size_t>(ch.n), [&](std::size_t t) {
              attention.Output(static_cast<int>(ch.begin + t), a + t * d);
            });
          },
          BiasRows(h, b.out_b, d, true));
      pass->Run(*b.up, patch_chunks, NormRows(b.ln2_w, b.ln2_b),
                [&](const Chunk& ch, const Bits* cb) {
                  Run(static_cast<std::size_t>(ch.n), [&](std::size_t t) {
                    BiasGeluTanhToBf16(cb + t * ffn, b.up_b.data(),
                                       act.data() + (ch.begin + t) * ffn, ffn);
                  });
                });
      pass->Run(*b.down, patch_chunks, CopyRows(act, ffn),
                BiasRows(h, b.down_b, d, true));
      const auto ds = std::find(c.deepstack.begin(), c.deepstack.end(), il);
      if (ds != c.deepstack.end()) {
        f.deepstack.push_back(
            Merge(deepstack[static_cast<std::size_t>(ds - c.deepstack.begin())],
                  tokens, true));
      }
    }
    f.embeds = Merge(merger, tokens, false);

    ++profile.images;
    profile.patches += n;
    profile.gemms = pass->gemms;
    profile.gemm_seconds = pass->gemm_seconds;
    profile.sync_seconds = pass->sync_seconds;
    profile.total_seconds += SecondsSince(start);
    profile.other_seconds = profile.total_seconds - profile.gemm_seconds -
                            profile.sync_seconds - profile.attention_seconds;
    return f;
  }
};

NpuVision::NpuVision(const gguf::File& mmproj, NpuEmbedder& embedder,
                     const std::filesystem::path& kernel_dir)
    : impl_(std::make_unique<Impl>(mmproj, embedder.gemm_runtime(),
                                   kernel_dir)) {}

NpuVision::~NpuVision() = default;

const VisionConfig& NpuVision::config() const {
  return impl_->config;
}

ImageFeatures NpuVision::Encode(const Patches& patches) {
  return impl_->Encode(patches);
}

const NpuVision::Profile& NpuVision::profile() const {
  return impl_->profile;
}

}  // namespace gufo_npu::qwen3vl_embed_2b
