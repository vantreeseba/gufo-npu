// Qwen3-VL-Embedding-2B's image input without a model: smart_resize,
// Pillow's bicubic resize (fixtures from Pillow 12.1), patch order and
// values, the position-embedding taps, MRoPE positions, and MRoPE attention
// against the text path when t = h = w.
#include "models/qwen3vl_embed_2b/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/qwen3vl_embed_2b/ops.hpp"

namespace {

namespace e = gufo_npu::qwen3vl_embed_2b;

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

template<typename Fn>
bool Throws(Fn&& fn) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void TestSmartResize() {
  // Expected values from qwen_vl_utils.smart_resize (factor 32, 4096 to
  // 1843200 pixels).
  struct Case {
    int h, w, eh, ew;
  };
  const Case cases[] = {
      {1440, 1280, 1440, 1280},  // a page at the cap: unchanged
      {100, 100, 96, 96},        // rounded to the factor
      {80, 48, 64, 64},          // 2.5 and 1.5 round half to even
      {4000, 3000, 1536, 1152},  // scaled down under the cap
      {10, 10, 64, 64},          // scaled up to the minimum
      {17, 1000, 32, 992},      {1080, 1920, 992, 1792},
      {3508, 2480, 1600, 1120},  // A4 at 300 dpi
  };
  for (const Case& c : cases) {
    const e::ImageSize s = e::SmartResize(c.h, c.w);
    Check(s.height == c.eh && s.width == c.ew,
          "smart_resize " + std::to_string(c.h) + "x" + std::to_string(c.w) +
              " -> " + std::to_string(s.height) + "x" +
              std::to_string(s.width));
  }
  Check(Throws([] { e::SmartResize(1, 201); }), "aspect ratio above 200");
  Check(Throws([] { e::SmartResize(0, 10); }), "empty image");
}

std::vector<std::uint8_t> Pattern(int width, int height) {
  std::vector<std::uint8_t> px;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        px.push_back(
            static_cast<std::uint8_t>(((x * 37 + y * 91 + c * 53) * 7) % 256));
      }
    }
  }
  return px;
}

void TestResize() {
  const auto src = Pattern(7, 5);
  // Image.frombytes("RGB", (7, 5), src).resize((w, h)).tobytes().
  const std::vector<std::uint8_t> down = {
      53,  168, 177, 58,  173, 182, 64, 180, 188, 69, 121, 193,
      199, 173, 162, 78,  178, 167, 63, 185, 173, 74, 154, 178,
      193, 158, 167, 201, 163, 172, 95, 169, 178, 56, 179, 183};
  const std::vector<std::uint8_t> up = {
      0,   106, 239, 0,   108, 241, 0,   110, 243, 0,   113, 246, 0,   115, 248,
      2,   117, 250, 5,   120, 253, 7,   128, 255, 9,   143, 255, 43,  171, 172,
      45,  173, 174, 53,  175, 176, 63,  178, 179, 65,  180, 181, 67,  182, 183,
      70,  185, 186, 72,  150, 188, 74,  74,  190, 133, 238, 100, 135, 240, 102,
      133, 242, 104, 130, 245, 107, 132, 247, 109, 134, 249, 111, 137, 253, 114,
      139, 173, 116, 141, 1,   118, 230, 140, 195, 234, 142, 197, 146, 144, 199,
      31,  147, 202, 34,  149, 204, 36,  151, 206, 39,  154, 209, 41,  131, 211,
      43,  79,  213, 219, 138, 193, 224, 140, 195, 135, 142, 197, 20,  145, 200,
      23,  147, 202, 32,  149, 204, 37,  152, 207, 39,  160, 209, 41,  175, 211,
      121, 233, 95,  123, 235, 97,  120, 237, 99,  118, 240, 102, 120, 242, 104,
      128, 244, 106, 132, 247, 109, 134, 249, 111, 136, 251, 113, 188, 161, 162,
      190, 163, 164, 198, 165, 166, 210, 168, 169, 210, 170, 171, 84,  172, 173,
      59,  175, 176, 62,  177, 178, 64,  179, 180, 253, 94,  227, 255, 96,  229,
      255, 98,  231, 255, 101, 234, 255, 103, 236, 38,  105, 238, 0,   108, 241,
      0,   110, 243, 0,   112, 245};
  const std::vector<std::uint8_t> vertical = {
      52,  167, 176, 55,  170, 179, 58,  173, 182, 61,  176, 185, 64,
      179, 188, 67,  182, 191, 70,  71,  194, 196, 172, 161, 199, 175,
      164, 54,  178, 167, 57,  181, 170, 69,  184, 173, 72,  187, 176,
      75,  127, 179, 192, 157, 166, 195, 160, 169, 198, 163, 172, 201,
      166, 175, 54,  169, 178, 57,  172, 181, 60,  183, 184};
  Check(e::ResizeBicubic(src.data(), 7, 5, 4, 3) == down,
        "Pillow bicubic, shrinking both sides");
  Check(e::ResizeBicubic(src.data(), 7, 5, 9, 8) == up,
        "Pillow bicubic, enlarging both sides");
  Check(e::ResizeBicubic(src.data(), 7, 5, 7, 3) == vertical,
        "Pillow bicubic, one side");
  Check(e::ResizeBicubic(src.data(), 7, 5, 7, 5) == src, "same size: a copy");
  const std::vector<std::uint8_t> flat(64 * 48 * 3, 200);
  const auto flat_out = e::ResizeBicubic(flat.data(), 64, 48, 37, 29);
  bool constant = flat_out.size() == 37u * 29u * 3u;
  for (const auto v : flat_out) {
    constant = constant && v == 200;
  }
  Check(constant, "a flat image stays flat");
}

void TestPatchify() {
  // 64 x 32: grid 4 wide, 2 high; one merge block row of two blocks.
  const int w = 64;
  const int h = 32;
  std::vector<std::uint8_t> px(static_cast<std::size_t>(w) * h * 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        // Encodes the patch (y / 16, x / 16) and the channel.
        px[(static_cast<std::size_t>(y) * w + x) * 3 + c] =
            static_cast<std::uint8_t>((y / 16) * 40 + (x / 16) * 10 + c);
      }
    }
  }
  const e::Patches p = e::Patchify(px.data(), w, h, 16, 2);
  Check(p.grid_h == 2 && p.grid_w == 4 && p.Count() == 8 &&
            p.values.size() == 8u * 768u,
        "patch grid");
  // Merge-block order: (0,0) (0,1) (1,0) (1,1) (0,2) (0,3) (1,2) (1,3).
  const int expect_row[] = {0, 0, 1, 1, 0, 0, 1, 1};
  const int expect_col[] = {0, 1, 0, 1, 2, 3, 2, 3};
  for (int i = 0; i < 8; ++i) {
    int row = -1;
    int col = -1;
    e::PatchPosition(i, p.grid_w, 2, &row, &col);
    Check(row == expect_row[i] && col == expect_col[i],
          "patch position " + std::to_string(i));
    for (int c = 0; c < 3; ++c) {
      const float v =
          p.values[static_cast<std::size_t>(i) * 768 + c * 256 + 17];
      const float raw = static_cast<float>(row * 40 + col * 10 + c);
      Check(std::fabs(v - (raw / 255.0f - 0.5f) / 0.5f) < 1e-6f,
            "patch " + std::to_string(i) + " channel " + std::to_string(c));
    }
  }
  Check(Throws([&] { e::Patchify(px.data(), 48, 32, 16, 2); }),
        "a side not a multiple of 32 throws");
  // 64 x 64 is exactly the minimum area: PreprocessImage keeps it.
  const auto square = Pattern(64, 64);
  Check(e::PreprocessImage(square.data(), 64, 64, 16, 2).values ==
            e::Patchify(square.data(), 64, 64, 16, 2).values,
        "no resize at a kept size");
  // 64 x 32 is below it: scaled to 96 x 64 first.
  const e::Patches small = e::PreprocessImage(px.data(), w, h, 16, 2);
  Check(small.grid_w == 6 && small.grid_h == 4, "small images are enlarged");
}

float MaxDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = a.size() == b.size() ? 0.0f : INFINITY;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    m = std::max(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

void TestInterpolation() {
  int taps[2];
  float weights[2];
  e::InterpolationTaps(0, 1, 48, taps, weights);
  Check(
      taps[0] == 0 && taps[1] == 1 && weights[0] == 1.0f && weights[1] == 0.0f,
      "a one-patch side reads entry 0");
  e::InterpolationTaps(1, 2, 48, taps, weights);
  Check(taps[0] == 47 && taps[1] == 47 && weights[0] == 1.0f &&
            weights[1] == 0.0f,
        "the last patch reads the last entry (align_corners)");
  e::InterpolationTaps(5, 48, 48, taps, weights);
  Check(taps[0] == 5 && weights[0] == 1.0f && weights[1] == 0.0f,
        "the table's own size reads it as is");
  e::InterpolationTaps(1, 3, 48, taps, weights);  // src 23.5
  Check(taps[0] == 23 && taps[1] == 24 &&
            std::fabs(weights[0] - 0.5f) < 1e-6f &&
            std::fabs(weights[1] - 0.5f) < 1e-6f,
        "halfway between entries");
}

void TestMropePositions() {
  const int pad = 99;
  e::ImageFeatures image;
  image.rows = 2;
  image.cols = 3;
  const std::vector<int> ids = {5, 6, pad, pad, pad, pad, pad, pad, 7, 8};
  const auto p = e::MropePositions(ids, pad, {image});
  const std::vector<int> expect = {
      0, 0, 0, 1, 1, 1,                                      // text
      2, 2, 2, 2, 2, 3, 2, 2, 4, 2, 3, 2, 2, 3, 3, 2, 3, 4,  // image
      5, 5, 5, 6, 6, 6};  // text after s + max(rows, cols)
  Check(p == expect, "MRoPE positions around an image");
  Check(Throws([&] { e::MropePositions(ids, pad, {}); }),
        "image tokens without an image throw");
  Check(Throws([&] { e::MropePositions({5, pad, pad, 6}, pad, {image}); }),
        "a short run of image tokens throws");
  Check(Throws([&] { e::MropePositions({5, 6}, pad, {image}); }),
        "an image without tokens throws");
  const auto text = e::MropePositions({1, 2, 3}, pad, {});
  Check(text == std::vector<int>({0, 0, 0, 1, 1, 1, 2, 2, 2}),
        "text only: one position on all axes");
}

void TestMrope() {
  const int hd = 128;
  e::RopeTable rope(hd, 5e6f);
  rope.Reserve(40);
  e::MropeTable mrope(hd, 5e6f);
  std::mt19937 rng(3);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(hd);
  for (auto& v : x) {
    v = dist(rng);
  }
  std::vector<float> cs(64), sn(64);
  auto a = x;
  auto b = x;
  rope.Apply(a.data(), 37);
  const int same[3] = {37, 37, 37};
  mrope.Angles(same, cs.data(), sn.data());
  mrope.Apply(b.data(), cs.data(), sn.data());
  Check(MaxDiff(a, b) < 1e-6f, "MRoPE with t = h = w is the text RoPE");
  // Only h turns: frequencies 1, 4, ..., 58 move, the rest stay.
  auto c = x;
  // Far enough that the slowest h frequency (58) still turns visibly.
  const int h_only[3] = {0, 100000, 0};
  mrope.Angles(h_only, cs.data(), sn.data());
  mrope.Apply(c.data(), cs.data(), sn.data());
  bool ok = true;
  for (int i = 0; i < 64; ++i) {
    const bool h_axis = i % 3 == 1 && i < 60;
    const bool moved = c[i] != x[i] || c[i + 64] != x[i + 64];
    ok = ok && moved == h_axis;
  }
  Check(ok, "the h section interleaves at i % 3 == 1 below 60");
}

void TestAttentionMrope() {
  e::Config cfg;
  cfg.heads = 4;
  cfg.kv_heads = 2;
  cfg.head_dim = 8;
  cfg.hidden = 32;
  cfg.rms_eps = 1e-6f;
  cfg.rope_theta = 10000.0f;
  const std::vector<e::Segment> segments = {{0, 70}, {70, 30}};
  const int rows = 100;
  std::mt19937 rng(11);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> in(static_cast<std::size_t>(rows) * cfg.InRows());
  for (auto& v : in) {
    v = dist(rng);
  }
  std::vector<float> q_norm(cfg.head_dim, 1.1f), k_norm(cfg.head_dim, 0.9f);
  e::RopeTable rope(cfg.head_dim, cfg.rope_theta);
  rope.Reserve(70);
  e::MropeTable mrope(cfg.head_dim, cfg.rope_theta);
  std::vector<int> positions;
  for (const auto& s : segments) {
    for (int t = 0; t < s.length; ++t) {
      positions.insert(positions.end(), {t, t, t});
    }
  }
  const e::ParallelRun run = [](std::size_t n,
                                const std::function<void(std::size_t)>& fn) {
    for (std::size_t i = 0; i < n; ++i) {
      fn(i);
    }
  };
  auto in_a = in;
  auto in_b = in;
  std::vector<float> out_a(static_cast<std::size_t>(rows) * cfg.QDim());
  std::vector<float> out_b(out_a.size());
  e::Attention(cfg, rope, q_norm.data(), k_norm.data(), segments, in_a.data(),
               out_a.data(), run);
  e::AttentionMrope(cfg, mrope, positions.data(), q_norm.data(), k_norm.data(),
                    segments, in_b.data(), out_b.data(), run);
  Check(MaxDiff(out_a, out_b) < 1e-5f,
        "MRoPE attention at text positions is Attention");
}

}  // namespace

int main() {
  TestSmartResize();
  TestResize();
  TestPatchify();
  TestInterpolation();
  TestMropePositions();
  TestMrope();
  TestAttentionMrope();
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS: qwen3vl_embed_2b vision");
  return EXIT_SUCCESS;
}
