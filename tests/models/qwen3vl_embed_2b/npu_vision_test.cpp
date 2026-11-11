// The NPU vision path (NpuVision, NpuEmbedder::EmbedImage) against HF in
// float32 and the CPU reference. The oracle's image cases
// (tools/reference/qwen3vl_embedding_oracle.py --images) go through
// PreprocessImage, NpuVision and NpuEmbedder::EmbedImage: the embedding
// against HF (cosine >= 0.99, as the text NPU test; the GEMMs round weights
// and activations to bfp16 and outputs to bf16) and the image features
// against VisionReference's. A synthetic 640 x 480 card (1,200 patches, two
// GEMM chunks) checks the features across chunks and its embedding against
// the CPU reference's (cosine >= 0.99), and a 1280 x 1440 page (7,200
// patches, the embedder's cap) is timed.
//
// The features' bar is looser than the embeddings': bfp16 noise builds up
// in a few tokens through the tower's 24 blocks (QUALITY.md), so it is set
// to catch a wrong patch order or chunk, not that noise. The quality gate is
// the embedding. Timings are warm, per phase. Needs an NPU, the model's
// variables and GUFO_NPU_QWEN3VL_EMBED_2B_MMPROJ; exits 77 without them.
#include "models/qwen3vl_embed_2b/npu_vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen3vl_embed_2b/input.hpp"
#include "models/qwen3vl_embed_2b/npu_embedder.hpp"
#include "models/qwen3vl_embed_2b/reference.hpp"
#include "models/qwen3vl_embed_2b/vision.hpp"
#include "oracle.hpp"

namespace {

namespace e = gufo_npu::qwen3vl_embed_2b;
using Clock = std::chrono::steady_clock;

constexpr double kEmbeddingCosine = 0.99;
constexpr double kFeatureCosine = 0.95;

double Since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// A deterministic card: horizontal and vertical gradients, a checkerboard
// in blue, and a little noise, so neighbouring patches differ.
std::vector<std::uint8_t> TestCard(int width, int height, std::uint32_t seed) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
  std::uint32_t state = seed * 2654435761U + 1U;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      std::uint8_t* px =
          rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
      const int base[3] = {x * 255 / std::max(width - 1, 1),
                           y * 255 / std::max(height - 1, 1),
                           ((x / 24 + y / 24) % 2) != 0 ? 220 : 30};
      for (int ch = 0; ch < 3; ++ch) {
        state = state * 1664525U + 1013904223U;
        const int noise = static_cast<int>((state >> 24) & 31U) - 16;
        px[ch] =
            static_cast<std::uint8_t>(std::clamp(base[ch] + noise, 0, 255));
      }
    }
  }
  return rgb;
}

// How two feature blocks (rows of `width` values) agree: the cosine as a
// whole, the worst row's, and how many rows fall below 0.99.
struct Agreement {
  double whole = 0;
  double worst_row = 1;
  std::size_t rows = 0;
  std::size_t below = 0;
};

Agreement Compare(const std::vector<float>& a, const std::vector<float>& b,
                  int width) {
  Agreement g;
  g.whole = qwen3vl_test::Cosine(a, b);
  g.rows = a.size() / static_cast<std::size_t>(width);
  for (std::size_t r = 0; r < g.rows && a.size() == b.size(); ++r) {
    const std::vector<float> ra(a.begin() + r * width,
                                a.begin() + (r + 1) * width);
    const std::vector<float> rb(b.begin() + r * width,
                                b.begin() + (r + 1) * width);
    const double cos = qwen3vl_test::Cosine(ra, rb);
    g.worst_row = std::min(g.worst_row, cos);
    g.below += cos < 0.99 ? 1 : 0;
  }
  return g;
}

int CheckFeatures(const e::ImageFeatures& got, const e::ImageFeatures& want,
                  int width, const std::string& name) {
  if (got.rows != want.rows || got.cols != want.cols ||
      got.deepstack.size() != want.deepstack.size()) {
    std::fprintf(stderr, "FAIL: %s: feature grid or DeepStack count differs\n",
                 name.c_str());
    return 1;
  }
  int failures = 0;
  const auto check = [&](const std::vector<float>& a,
                         const std::vector<float>& b, const std::string& what) {
    const Agreement g = Compare(a, b, width);
    std::printf(
        "  %s %s vs the CPU reference: cosine %.6f, worst token %.6f, %zu of "
        "%zu tokens below 0.99\n",
        name.c_str(), what.c_str(), g.whole, g.worst_row, g.below, g.rows);
    if (g.whole < kFeatureCosine) {
      std::fprintf(stderr, "FAIL: %s %s: cosine %.6f < %.2f\n", name.c_str(),
                   what.c_str(), g.whole, kFeatureCosine);
      ++failures;
    }
  };
  check(got.embeds, want.embeds, "features");
  for (std::size_t k = 0; k < got.deepstack.size(); ++k) {
    check(got.deepstack[k], want.deepstack[k],
          "DeepStack " + std::to_string(k));
  }
  return failures;
}

void PrintVision(const e::NpuVision::Profile& a, const e::NpuVision::Profile& b,
                 double seconds) {
  std::printf(
      "  vision %.3f s: %ld GEMM calls, gemm %.3f s, sync %.3f s, attention "
      "%.3f s, other %.3f s\n",
      seconds, b.gemms - a.gemms, b.gemm_seconds - a.gemm_seconds,
      b.sync_seconds - a.sync_seconds,
      b.attention_seconds - a.attention_seconds,
      b.other_seconds - a.other_seconds);
}

void PrintText(const e::NpuEmbedder::Profile& a,
               const e::NpuEmbedder::Profile& b, double seconds) {
  std::printf(
      "  text %.3f s: %ld GEMM calls, gemm %.3f s, sync %.3f s, attention "
      "%.3f s, other %.3f s\n",
      seconds, b.gemms - a.gemms, b.gemm_seconds - a.gemm_seconds,
      b.sync_seconds - a.sync_seconds,
      b.attention_seconds - a.attention_seconds,
      b.other_seconds - a.other_seconds);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr,
                 "usage: qwen3vl_embed_2b_npu_vision_test <kernel-dir>\n");
    return 2;
  }
  if (!std::filesystem::exists("/dev/accel/accel0")) {
    std::puts("SKIP: no NPU at /dev/accel/accel0");
    return 77;
  }
  const auto oracle = qwen3vl_test::Load();
  if (!oracle) {
    return 77;
  }
  const char* mmproj = std::getenv("GUFO_NPU_QWEN3VL_EMBED_2B_MMPROJ");
  if (mmproj == nullptr || *mmproj == '\0') {
    std::puts("SKIP: set GUFO_NPU_QWEN3VL_EMBED_2B_MMPROJ");
    return 77;
  }
  std::ifstream in(std::getenv("GUFO_NPU_QWEN3VL_EMBED_2B_ORACLE"));
  const auto j = nlohmann::json::parse(in);
  if (!j.contains("image_cases")) {
    std::puts("SKIP: the oracle has no image cases (run it with --images)");
    return 77;
  }
  const auto file = gufo_npu::gguf::File::Open(oracle->gguf);
  const auto vision_file = gufo_npu::gguf::File::Open(mmproj);
  const gufo_npu::Tokenizer tokenizer(*file);
  e::NpuEmbedder embedder(*file, argv[1]);
  e::NpuVision vision(*vision_file, embedder);
  e::VisionReference reference(*vision_file);
  e::Reference text_reference(*file);
  const e::VisionConfig& vc = vision.config();
  const int pad = tokenizer.Find(e::kImagePad);
  int failures = 0;

  for (const auto& c : j.at("image_cases")) {
    const auto name = c.at("name").get<std::string>();
    const int width = c.at("width").get<int>();
    const int height = c.at("height").get<int>();
    const auto rgb = c.at("rgb").get<std::vector<std::uint8_t>>();
    const auto grid = c.at("grid").get<std::vector<int>>();
    const e::Patches patches =
        e::PreprocessImage(rgb.data(), width, height, vc.patch, vc.merge);
    if (grid.size() != 3 || patches.grid_h != grid[1] ||
        patches.grid_w != grid[2]) {
      std::fprintf(stderr, "FAIL: %s: patch grid differs from HF\n",
                   name.c_str());
      ++failures;
      continue;
    }
    vision.Encode(patches);  // warm-up
    auto start = Clock::now();
    const e::ImageFeatures features = vision.Encode(patches);
    const double vision_seconds = Since(start);
    const auto ids = e::ImageInputIds(tokenizer, features.Tokens(),
                                      c.at("text").get<std::string>(),
                                      c.at("instruction").get<std::string>());
    if (ids != c.at("ids").get<std::vector<int>>()) {
      std::fprintf(stderr, "FAIL: %s: token ids differ from HF\n",
                   name.c_str());
      ++failures;
      continue;
    }
    embedder.EmbedImage(ids, pad, {features});  // warm-up
    start = Clock::now();
    const auto got = embedder.EmbedImage(ids, pad, {features});
    const double text_seconds = Since(start);
    const double cos =
        qwen3vl_test::Cosine(got, c.at("embedding").get<std::vector<float>>());
    std::printf(
        "%s: %d patches, %zu tokens, cosine to HF %.6f; vision %.3f s, text "
        "%.3f s (warm)\n",
        name.c_str(), patches.Count(), ids.size(), cos, vision_seconds,
        text_seconds);
    if (cos < kEmbeddingCosine) {
      std::fprintf(stderr, "FAIL: %s: cosine %.6f to HF < %.2f\n", name.c_str(),
                   cos, kEmbeddingCosine);
      ++failures;
    }
    failures +=
        CheckFeatures(features, reference.Encode(patches), vc.out_hidden, name);
  }

  {
    // Across GEMM chunks: 1,200 patches (two chunks), 300 tokens.
    const auto card = TestCard(640, 480, 3);
    const e::Patches patches =
        e::PreprocessImage(card.data(), 640, 480, vc.patch, vc.merge);
    const auto start = Clock::now();
    const e::ImageFeatures features = vision.Encode(patches);
    std::printf("card 640x480: %d patches, vision %.3f s (cold)\n",
                patches.Count(), Since(start));
    const e::ImageFeatures want = reference.Encode(patches);
    failures += CheckFeatures(features, want, vc.out_hidden, "card 640x480");
    // End to end: the NPU path against the CPU reference's (vision and text
    // in fp32), which matches HF at cosine 1.000 (QUALITY.md).
    const auto ids = e::ImageInputIds(tokenizer, features.Tokens(),
                                      "A test card with two gradients.", "");
    const double cos =
        qwen3vl_test::Cosine(embedder.EmbedImage(ids, pad, {features}),
                             text_reference.EmbedImage(ids, pad, {want}));
    std::printf(
        "card 640x480: %zu tokens, embedding cosine to the CPU "
        "reference %.6f\n",
        ids.size(), cos);
    if (cos < kEmbeddingCosine) {
      std::fprintf(stderr, "FAIL: card 640x480: cosine %.6f < %.2f\n", cos,
                   kEmbeddingCosine);
      ++failures;
    }
  }

  {
    // A page at the embedder's cap: 7,200 patches, 1,800 image tokens.
    const auto card = TestCard(1280, 1440, 5);
    const e::Patches patches =
        e::PreprocessImage(card.data(), 1280, 1440, vc.patch, vc.merge);
    e::ImageFeatures features = vision.Encode(patches);  // warm-up
    const auto ids = e::ImageInputIds(tokenizer, features.Tokens(), "", "");
    embedder.EmbedImage(ids, pad, {features});  // warm-up
    std::printf("page 1280x1440: %d patches, %zu tokens\n", patches.Count(),
                ids.size());
    for (int rep = 0; rep < 3; ++rep) {
      const auto vision_before = vision.profile();
      auto start = Clock::now();
      features = vision.Encode(patches);
      const double vision_seconds = Since(start);
      const auto text_before = embedder.profile();
      start = Clock::now();
      embedder.EmbedImage(ids, pad, {features});
      const double text_seconds = Since(start);
      std::printf("run %d: %.3f s\n", rep + 1, vision_seconds + text_seconds);
      PrintVision(vision_before, vision.profile(), vision_seconds);
      PrintText(text_before, embedder.profile(), text_seconds);
    }
  }

  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS: NPU image embeddings match HF and the CPU reference");
  return EXIT_SUCCESS;
}
