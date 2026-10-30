#include "models/qwen3vl_embed_2b/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>

#include "core/parallel.hpp"
#include "core/quant.hpp"
#include "core/vec.hpp"

namespace gufo_npu::qwen3vl_embed_2b {
namespace {

using gguf::TensorInfo;

std::vector<float> ToFloats(const TensorInfo& t) {
  std::vector<float> out(t.num_elements);
  quant::Dequantize(t.type, t.data, out.data(), out.size());
  return out;
}

std::string ProjectorType(const gguf::File& file) {
  for (const char* key :
       {"clip.vision.projector_type", "clip.projector_type"}) {
    const gguf::Value* v = file.Find(key);
    if (v != nullptr && v->kind == gguf::Value::Kind::kString) {
      return v->s;
    }
  }
  return {};
}

int UInt(const gguf::File& file, const char* key) {
  return static_cast<int>(file.GetUInt(key));
}

// Pillow's bicubic kernel (a = -0.5) and its fixed-point precision.
constexpr int kPrecisionBits = 32 - 8 - 2;

double BicubicFilter(double x) {
  constexpr double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  }
  if (x < 2.0) {
    return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  }
  return 0.0;
}

// One axis of Pillow's resample (precompute_coeffs + normalize_coeffs_8bpc):
// output i reads `count[i]` inputs from `first[i]` with weights
// coeffs[i * ksize + j] in 22-bit fixed point.
struct Coefficients {
  int ksize = 0;
  std::vector<int> first, count;
  std::vector<int> coeffs;
};

Coefficients Precompute(int in_size, int out_size) {
  const double scale = static_cast<double>(in_size) / out_size;
  const double filterscale = std::max(scale, 1.0);
  const double support = 2.0 * filterscale;
  Coefficients c;
  c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  c.first.resize(out_size);
  c.count.resize(out_size);
  c.coeffs.assign(static_cast<std::size_t>(out_size) * c.ksize, 0);
  std::vector<double> k(static_cast<std::size_t>(c.ksize));
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    xmin = std::max(xmin, 0);
    int xmax = static_cast<int>(center + support + 0.5);
    xmax = std::min(xmax, in_size) - xmin;
    double ww = 0.0;
    for (int x = 0; x < xmax; ++x) {
      const double w = BicubicFilter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x) {
      const double w = ww != 0.0 ? k[x] / ww : k[x];
      const double fixed = w * (1 << kPrecisionBits);
      c.coeffs[static_cast<std::size_t>(xx) * c.ksize + x] =
          static_cast<int>(w < 0 ? fixed - 0.5 : fixed + 0.5);
    }
    c.first[xx] = xmin;
    c.count[xx] = xmax;
  }
  return c;
}

std::uint8_t Clip8(long long v) {
  const long long s = v >> kPrecisionBits;
  return static_cast<std::uint8_t>(std::clamp<long long>(s, 0, 255));
}

float GeluTanh(float x) {
  return 0.5f * x *
         (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

float GeluErf(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
}

// LayerNorm of `count` rows of `dim` values, in place.
void LayerNormRows(float* x, int count, int dim, const float* w, const float* b,
                   float eps) {
  ParallelFor(
      static_cast<std::size_t>(count), [&](std::size_t begin, std::size_t end) {
        for (std::size_t t = begin; t < end; ++t) {
          float* row = x + t * dim;
          double mean = 0;
          for (int i = 0; i < dim; ++i) {
            mean += row[i];
          }
          mean /= dim;
          double var = 0;
          for (int i = 0; i < dim; ++i) {
            const double d = row[i] - mean;
            var += d * d;
          }
          var /= dim;
          const auto m = static_cast<float>(mean);
          const auto inv = static_cast<float>(1.0 / std::sqrt(var + eps));
          for (int i = 0; i < dim; ++i) {
            row[i] = (row[i] - m) * inv * w[i] + b[i];
          }
        }
      });
}

// out[t][o] = x_t . W_o + bias[o] for `count` rows x of W's row length;
// W (rows x dim, any GGUF type) is dequantized a block of rows at a time.
void Linear(const TensorInfo& w, const std::vector<float>& bias, const float* x,
            int count, float* out) {
  const auto dim = static_cast<std::size_t>(w.RowElements());
  const auto rows = static_cast<std::size_t>(w.Rows());
  const std::size_t row_bytes = quant::RowBytes(w.type, dim);
  constexpr std::size_t kBlock = 16;
  const std::size_t blocks = (rows + kBlock - 1) / kBlock;
  ParallelFor(blocks, [&](std::size_t begin, std::size_t end) {
    std::vector<float> wb(kBlock * dim);
    for (std::size_t blk = begin; blk < end; ++blk) {
      const std::size_t o0 = blk * kBlock;
      const std::size_t n = std::min(kBlock, rows - o0);
      for (std::size_t j = 0; j < n; ++j) {
        quant::Dequantize(w.type, w.data + (o0 + j) * row_bytes,
                          wb.data() + j * dim, dim);
      }
      for (int t = 0; t < count; ++t) {
        const float* xt = x + static_cast<std::size_t>(t) * dim;
        float* ot = out + static_cast<std::size_t>(t) * rows;
        for (std::size_t j = 0; j < n; ++j) {
          ot[o0 + j] =
              vec::Dot(xt, wb.data() + j * dim, static_cast<int>(dim)) +
              bias[o0 + j];
        }
      }
    }
  });
}

}  // namespace

bool IsVisionProjector(const gguf::File& file) {
  return ProjectorType(file) == kProjectorType;
}

VisionConfig ReadVisionConfig(const gguf::File& file) {
  const std::string type = ProjectorType(file);
  if (type != kProjectorType) {
    throw std::runtime_error("not a Qwen3-VL mmproj (projector type '" + type +
                             "')");
  }
  VisionConfig c;
  c.layers = UInt(file, "clip.vision.block_count");
  c.hidden = UInt(file, "clip.vision.embedding_length");
  c.ffn = UInt(file, "clip.vision.feed_forward_length");
  c.heads = UInt(file, "clip.vision.attention.head_count");
  c.patch = UInt(file, "clip.vision.patch_size");
  c.merge = file.Find("clip.vision.spatial_merge_size") != nullptr
                ? UInt(file, "clip.vision.spatial_merge_size")
                : 2;
  if (file.Find("clip.vision.attention.layer_norm_epsilon") != nullptr) {
    c.eps = static_cast<float>(
        file.GetFloat("clip.vision.attention.layer_norm_epsilon"));
  }
  c.out_hidden = static_cast<int>(file.Tensor("mm.2.weight").Rows());
  const auto& pos = file.Tensor("v.position_embd.weight");
  c.pos_side =
      static_cast<int>(std::lround(std::sqrt(static_cast<double>(pos.Rows()))));
  for (int il = 0; il < c.layers; ++il) {
    if (file.FindTensor("v.deepstack." + std::to_string(il) + ".fc1.weight") !=
        nullptr) {
      c.deepstack.push_back(il);
    }
  }
  if (c.layers <= 0 || c.hidden <= 0 || c.heads <= 0 ||
      c.hidden % c.heads != 0 || c.HeadDim() % 4 != 0 || c.patch <= 0 ||
      c.merge <= 0 || c.ffn <= 0 || c.out_hidden <= 0 ||
      pos.RowElements() != static_cast<std::uint64_t>(c.hidden) ||
      static_cast<std::uint64_t>(c.pos_side) * c.pos_side != pos.Rows()) {
    throw std::runtime_error("qwen3vl mmproj: inconsistent shape metadata");
  }
  return c;
}

ImageSize SmartResize(int height, int width, int factor, int min_pixels,
                      int max_pixels) {
  if (height < 1 || width < 1) {
    throw std::invalid_argument("qwen3vl: empty image");
  }
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200.0) {
    throw std::invalid_argument("qwen3vl: image aspect ratio above 200");
  }
  // Python's round() rounds half to even, as nearbyint does by default.
  auto round_by = [&](double v) {
    return static_cast<int>(std::nearbyint(v / factor)) * factor;
  };
  int h = std::max(factor, round_by(height));
  int w = std::max(factor, round_by(width));
  const double area = static_cast<double>(height) * width;
  if (static_cast<double>(h) * w > max_pixels) {
    const double beta = std::sqrt(area / max_pixels);
    // qwen_vl_utils has no lower bound here; one factor keeps a valid size.
    h = std::max(factor,
                 static_cast<int>(std::floor(height / beta / factor)) * factor);
    w = std::max(factor,
                 static_cast<int>(std::floor(width / beta / factor)) * factor);
  } else if (static_cast<double>(h) * w < min_pixels) {
    const double beta = std::sqrt(min_pixels / area);
    h = static_cast<int>(std::ceil(height * beta / factor)) * factor;
    w = static_cast<int>(std::ceil(width * beta / factor)) * factor;
  }
  return {h, w};
}

std::vector<std::uint8_t> ResizeBicubic(const std::uint8_t* rgb, int width,
                                        int height, int out_width,
                                        int out_height) {
  if (width < 1 || height < 1 || out_width < 1 || out_height < 1) {
    throw std::invalid_argument("qwen3vl: empty image");
  }
  std::vector<std::uint8_t> src(
      rgb, rgb + static_cast<std::size_t>(width) * height * 3);
  if (out_width != width) {
    const Coefficients c = Precompute(width, out_width);
    std::vector<std::uint8_t> dst(static_cast<std::size_t>(out_width) * height *
                                  3);
    for (int y = 0; y < height; ++y) {
      const std::uint8_t* in =
          src.data() + static_cast<std::size_t>(y) * width * 3;
      std::uint8_t* out =
          dst.data() + static_cast<std::size_t>(y) * out_width * 3;
      for (int x = 0; x < out_width; ++x) {
        const int* k = c.coeffs.data() + static_cast<std::size_t>(x) * c.ksize;
        for (int ch = 0; ch < 3; ++ch) {
          long long ss = 1LL << (kPrecisionBits - 1);
          for (int j = 0; j < c.count[x]; ++j) {
            ss += static_cast<long long>(in[(c.first[x] + j) * 3 + ch]) * k[j];
          }
          out[x * 3 + ch] = Clip8(ss);
        }
      }
    }
    src = std::move(dst);
    width = out_width;
  }
  if (out_height != height) {
    const Coefficients c = Precompute(height, out_height);
    std::vector<std::uint8_t> dst(static_cast<std::size_t>(width) * out_height *
                                  3);
    const std::size_t stride = static_cast<std::size_t>(width) * 3;
    for (int y = 0; y < out_height; ++y) {
      const int* k = c.coeffs.data() + static_cast<std::size_t>(y) * c.ksize;
      for (std::size_t i = 0; i < stride; ++i) {
        long long ss = 1LL << (kPrecisionBits - 1);
        for (int j = 0; j < c.count[y]; ++j) {
          ss +=
              static_cast<long long>(
                  src[static_cast<std::size_t>(c.first[y] + j) * stride + i]) *
              k[j];
        }
        dst[static_cast<std::size_t>(y) * stride + i] = Clip8(ss);
      }
    }
    src = std::move(dst);
  }
  return src;
}

void PatchPosition(int index, int grid_w, int merge, int* row, int* col) {
  const int unit = merge * merge;
  const int block = index / unit;
  const int within = index % unit;
  const int blocks_w = grid_w / merge;
  *row = (block / blocks_w) * merge + within / merge;
  *col = (block % blocks_w) * merge + within % merge;
}

Patches Patchify(const std::uint8_t* rgb, int width, int height, int patch,
                 int merge) {
  const int factor = patch * merge;
  if (width < factor || height < factor || width % factor != 0 ||
      height % factor != 0) {
    throw std::invalid_argument(
        "qwen3vl: image sides must be multiples of patch * merge");
  }
  Patches p;
  p.grid_h = height / patch;
  p.grid_w = width / patch;
  const int values = 3 * patch * patch;
  p.values.resize(static_cast<std::size_t>(p.Count()) * values);
  for (int i = 0; i < p.Count(); ++i) {
    int row = 0;
    int col = 0;
    PatchPosition(i, p.grid_w, merge, &row, &col);
    float* out = p.values.data() + static_cast<std::size_t>(i) * values;
    for (int ch = 0; ch < 3; ++ch) {
      for (int y = 0; y < patch; ++y) {
        const std::size_t line =
            (static_cast<std::size_t>(row) * patch + y) * width;
        for (int x = 0; x < patch; ++x) {
          const std::uint8_t v =
              rgb[(line + static_cast<std::size_t>(col) * patch + x) * 3 + ch];
          // Rescale by 1/255 and normalize (mean = std = 0.5), fused as the
          // processor fuses them.
          out[(ch * patch + y) * patch + x] =
              (static_cast<float>(v) - 127.5f) / 127.5f;
        }
      }
    }
  }
  return p;
}

Patches PreprocessImage(const std::uint8_t* rgb, int width, int height,
                        int patch, int merge) {
  const ImageSize size = SmartResize(height, width, patch * merge);
  if (size.width == width && size.height == height) {
    return Patchify(rgb, width, height, patch, merge);
  }
  const auto resized =
      ResizeBicubic(rgb, width, height, size.width, size.height);
  return Patchify(resized.data(), size.width, size.height, patch, merge);
}

void InterpolationTaps(int index, int size, int side, int* taps,
                       float* weights) {
  // HF (fp32): src = index * (side - 1) / max(size - 1, 1).
  const float src = static_cast<float>(index) * static_cast<float>(side - 1) /
                    static_cast<float>(std::max(size - 1, 1));
  const float fl = std::floor(src);
  for (int o = 0; o < 2; ++o) {
    taps[o] = std::clamp(static_cast<int>(fl) + o, 0, side - 1);
    weights[o] =
        std::max(0.0f, 1.0f - std::fabs(src - fl - static_cast<float>(o)));
  }
}

std::vector<int> MropePositions(const std::vector<int>& ids, int image_pad_id,
                                const std::vector<ImageFeatures>& images) {
  std::vector<int> out(ids.size() * 3);
  std::size_t next = 0;
  int pos = 0;
  std::size_t i = 0;
  while (i < ids.size()) {
    if (ids[i] != image_pad_id) {
      out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = pos++;
      ++i;
      continue;
    }
    if (next >= images.size()) {
      throw std::invalid_argument("qwen3vl: more image tokens than images");
    }
    const ImageFeatures& image = images[next++];
    const auto tokens = static_cast<std::size_t>(image.Tokens());
    if (tokens == 0 || i + tokens > ids.size()) {
      throw std::invalid_argument("qwen3vl: image tokens do not match image");
    }
    for (int r = 0; r < image.rows; ++r) {
      for (int c = 0; c < image.cols; ++c) {
        const std::size_t k = i + static_cast<std::size_t>(r) * image.cols + c;
        if (ids[k] != image_pad_id) {
          throw std::invalid_argument(
              "qwen3vl: image tokens do not match image");
        }
        out[3 * k] = pos;
        out[3 * k + 1] = pos + r;
        out[3 * k + 2] = pos + c;
      }
    }
    pos += std::max(image.rows, image.cols);
    i += tokens;
  }
  if (next != images.size()) {
    throw std::invalid_argument("qwen3vl: fewer image tokens than images");
  }
  return out;
}

struct VisionReference::Impl {
  struct Block {
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
    std::vector<float> qkv_b, out_b, up_b, down_b;
    const TensorInfo* qkv = nullptr;
    const TensorInfo* out = nullptr;
    const TensorInfo* up = nullptr;
    const TensorInfo* down = nullptr;
  };
  struct Merger {
    std::vector<float> norm_w, norm_b, fc1_b, fc2_b;
    const TensorInfo* fc1 = nullptr;
    const TensorInfo* fc2 = nullptr;
  };

  VisionConfig config;
  std::vector<float> patch_w;  // hidden x 3 * patch^2, temporal slots summed
  std::vector<float> patch_b;
  std::vector<float> pos_embed;  // pos_side^2 x hidden
  std::vector<float> inv_freq;   // head_dim / 4 values
  std::vector<Block> blocks;
  Merger merger;
  std::vector<Merger> deepstack;

  explicit Impl(const gguf::File& file) : config(ReadVisionConfig(file)) {
    const VisionConfig& c = config;
    patch_w = ToFloats(file.Tensor("v.patch_embd.weight"));
    if (const TensorInfo* second = file.FindTensor("v.patch_embd.weight.1")) {
      const auto w1 = ToFloats(*second);
      if (w1.size() != patch_w.size()) {
        throw std::runtime_error("qwen3vl mmproj: patch embedding shapes");
      }
      for (std::size_t i = 0; i < w1.size(); ++i) {
        patch_w[i] += w1[i];
      }
    } else {
      // A single 2D kernel covers one frame; the image fills both slots.
      for (float& v : patch_w) {
        v *= 2.0f;
      }
    }
    if (patch_w.size() !=
        static_cast<std::size_t>(c.hidden) * c.PatchValues()) {
      throw std::runtime_error("qwen3vl mmproj: patch embedding shape");
    }
    patch_b = ToFloats(file.Tensor("v.patch_embd.bias"));
    pos_embed = ToFloats(file.Tensor("v.position_embd.weight"));
    const int quarter = c.HeadDim() / 4;
    inv_freq.resize(quarter);
    for (int j = 0; j < quarter; ++j) {
      // HF: 1 / theta^(arange(0, head_dim/2, 2) / (head_dim/2)).
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
      b.qkv = &t("attn_qkv.weight");
      b.qkv_b = ToFloats(t("attn_qkv.bias"));
      b.out = &t("attn_out.weight");
      b.out_b = ToFloats(t("attn_out.bias"));
      b.up = &t("ffn_up.weight");
      b.up_b = ToFloats(t("ffn_up.bias"));
      b.down = &t("ffn_down.weight");
      b.down_b = ToFloats(t("ffn_down.bias"));
      if (b.qkv->Rows() != static_cast<std::uint64_t>(3 * c.hidden) ||
          b.up->Rows() != static_cast<std::uint64_t>(c.ffn)) {
        throw std::runtime_error("qwen3vl mmproj: block shapes");
      }
      blocks.push_back(std::move(b));
    }
    merger.norm_w = ToFloats(file.Tensor("v.post_ln.weight"));
    merger.norm_b = ToFloats(file.Tensor("v.post_ln.bias"));
    merger.fc1 = &file.Tensor("mm.0.weight");
    merger.fc1_b = ToFloats(file.Tensor("mm.0.bias"));
    merger.fc2 = &file.Tensor("mm.2.weight");
    merger.fc2_b = ToFloats(file.Tensor("mm.2.bias"));
    for (const int il : c.deepstack) {
      const std::string p = "v.deepstack." + std::to_string(il) + ".";
      Merger m;
      m.norm_w = ToFloats(file.Tensor(p + "norm.weight"));
      m.norm_b = ToFloats(file.Tensor(p + "norm.bias"));
      m.fc1 = &file.Tensor(p + "fc1.weight");
      m.fc1_b = ToFloats(file.Tensor(p + "fc1.bias"));
      m.fc2 = &file.Tensor(p + "fc2.weight");
      m.fc2_b = ToFloats(file.Tensor(p + "fc2.bias"));
      deepstack.push_back(std::move(m));
    }
  }

  // A merger over `n` patch rows of h: LayerNorm per patch (main merger)
  // or per merged row (DeepStack, `postshuffle`), fc1, exact GELU, fc2.
  std::vector<float> Merge(const Merger& m, const std::vector<float>& h, int n,
                           bool postshuffle) const {
    const VisionConfig& c = config;
    const int tokens = n / (c.merge * c.merge);
    const int md = c.MergedDim();
    std::vector<float> x(h);
    if (postshuffle) {
      LayerNormRows(x.data(), tokens, md, m.norm_w.data(), m.norm_b.data(),
                    c.eps);
    } else {
      LayerNormRows(x.data(), n, c.hidden, m.norm_w.data(), m.norm_b.data(),
                    c.eps);
    }
    std::vector<float> mid(static_cast<std::size_t>(tokens) * md);
    Linear(*m.fc1, m.fc1_b, x.data(), tokens, mid.data());
    for (float& v : mid) {
      v = GeluErf(v);
    }
    std::vector<float> out(static_cast<std::size_t>(tokens) * c.out_hidden);
    Linear(*m.fc2, m.fc2_b, mid.data(), tokens, out.data());
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
    const int n = patches.Count();
    const int d = c.hidden;
    const int hd = c.HeadDim();
    const int half = hd / 2;
    const int quarter = hd / 4;
    const auto rows = static_cast<std::size_t>(n);

    // Patch embedding + bias + resampled learned positions.
    std::vector<float> h(rows * d);
    std::vector<int> prow(rows), pcol(rows);
    ParallelFor(rows, [&](std::size_t begin, std::size_t end) {
      for (std::size_t t = begin; t < end; ++t) {
        const float* px = patches.values.data() + t * c.PatchValues();
        float* out = h.data() + t * d;
        for (int o = 0; o < d; ++o) {
          out[o] = vec::Dot(px,
                            patch_w.data() +
                                static_cast<std::size_t>(o) * c.PatchValues(),
                            c.PatchValues()) +
                   patch_b[o];
        }
        int r = 0;
        int col = 0;
        PatchPosition(static_cast<int>(t), patches.grid_w, c.merge, &r, &col);
        prow[t] = r;
        pcol[t] = col;
        int th[2], tw[2];
        float wh[2], ww[2];
        InterpolationTaps(r, patches.grid_h, c.pos_side, th, wh);
        InterpolationTaps(col, patches.grid_w, c.pos_side, tw, ww);
        std::vector<float> pe(static_cast<std::size_t>(d), 0.0f);
        for (int a = 0; a < 2; ++a) {
          for (int b = 0; b < 2; ++b) {
            const float w = wh[a] * ww[b];
            const float* src =
                pos_embed.data() +
                (static_cast<std::size_t>(th[a]) * c.pos_side + tw[b]) * d;
            for (int i = 0; i < d; ++i) {
              pe[i] += w * src[i];
            }
          }
        }
        for (int i = 0; i < d; ++i) {
          out[i] += pe[i];
        }
      }
    });

    // 2D RoPE per patch: pair i < quarter turns by row, the rest by col.
    std::vector<float> rcos(rows * half), rsin(rows * half);
    for (std::size_t t = 0; t < rows; ++t) {
      for (int i = 0; i < half; ++i) {
        const float angle =
            i < quarter ? static_cast<float>(prow[t]) * inv_freq[i]
                        : static_cast<float>(pcol[t]) * inv_freq[i - quarter];
        rcos[t * half + i] = std::cos(angle);
        rsin[t * half + i] = std::sin(angle);
      }
    }

    ImageFeatures f;
    f.rows = patches.grid_h / c.merge;
    f.cols = patches.grid_w / c.merge;
    std::vector<float> x(rows * d), qkv(rows * 3 * d), mixed(rows * d),
        tmp(rows * d), up(rows * c.ffn);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (int il = 0; il < c.layers; ++il) {
      const Block& b = blocks[il];
      x = h;
      LayerNormRows(x.data(), n, d, b.ln1_w.data(), b.ln1_b.data(), c.eps);
      Linear(*b.qkv, b.qkv_b, x.data(), n, qkv.data());
      ParallelFor(rows, [&](std::size_t begin, std::size_t end) {
        for (std::size_t t = begin; t < end; ++t) {
          const float* cs = rcos.data() + t * half;
          const float* sn = rsin.data() + t * half;
          for (int part = 0; part < 2; ++part) {  // q, k
            for (int head = 0; head < c.heads; ++head) {
              float* v = qkv.data() + t * 3 * d + part * d + head * hd;
              for (int i = 0; i < half; ++i) {
                const float a = v[i];
                const float bb = v[i + half];
                v[i] = a * cs[i] - bb * sn[i];
                v[i + half] = bb * cs[i] + a * sn[i];
              }
              if (part == 0) {
                for (int e = 0; e < hd; ++e) {
                  v[e] *= scale;
                }
              }
            }
          }
        }
      });
      // Full (bidirectional) attention over the image's patches.
      ParallelFor(static_cast<std::size_t>(c.heads) * rows,
                  [&](std::size_t begin, std::size_t end) {
                    std::vector<float> p(rows);
                    for (std::size_t item = begin; item < end; ++item) {
                      const auto head = static_cast<int>(item / rows);
                      const std::size_t t = item % rows;
                      const float* q = qkv.data() + t * 3 * d +
                                       static_cast<std::size_t>(head) * hd;
                      float mx = -INFINITY;
                      for (std::size_t j = 0; j < rows; ++j) {
                        p[j] = vec::Dot(q,
                                        qkv.data() + j * 3 * d + d +
                                            static_cast<std::size_t>(head) * hd,
                                        hd);
                        mx = std::max(mx, p[j]);
                      }
                      float* o = mixed.data() + t * d +
                                 static_cast<std::size_t>(head) * hd;
                      std::fill(o, o + hd, 0.0f);
                      double total = 0;
                      for (std::size_t j = 0; j < rows; ++j) {
                        const float w = std::exp(p[j] - mx);
                        total += w;
                        const float* v = qkv.data() + j * 3 * d + 2 * d +
                                         static_cast<std::size_t>(head) * hd;
                        for (int e = 0; e < hd; ++e) {
                          o[e] += w * v[e];
                        }
                      }
                      const auto inv = static_cast<float>(1.0 / total);
                      for (int e = 0; e < hd; ++e) {
                        o[e] *= inv;
                      }
                    }
                  });
      Linear(*b.out, b.out_b, mixed.data(), n, tmp.data());
      for (std::size_t i = 0; i < h.size(); ++i) {
        h[i] += tmp[i];
      }
      x = h;
      LayerNormRows(x.data(), n, d, b.ln2_w.data(), b.ln2_b.data(), c.eps);
      Linear(*b.up, b.up_b, x.data(), n, up.data());
      for (float& v : up) {
        v = GeluTanh(v);
      }
      Linear(*b.down, b.down_b, up.data(), n, tmp.data());
      for (std::size_t i = 0; i < h.size(); ++i) {
        h[i] += tmp[i];
      }
      const auto ds = std::find(c.deepstack.begin(), c.deepstack.end(), il);
      if (ds != c.deepstack.end()) {
        f.deepstack.push_back(
            Merge(deepstack[static_cast<std::size_t>(ds - c.deepstack.begin())],
                  h, n, true));
      }
    }
    f.embeds = Merge(merger, h, n, false);
    return f;
  }
};

VisionReference::VisionReference(const gguf::File& file)
    : impl_(std::make_unique<Impl>(file)) {}

VisionReference::~VisionReference() = default;

const VisionConfig& VisionReference::config() const {
  return impl_->config;
}

ImageFeatures VisionReference::Encode(const Patches& patches) {
  return impl_->Encode(patches);
}

}  // namespace gufo_npu::qwen3vl_embed_2b
