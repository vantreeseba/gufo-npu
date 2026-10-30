// The CPU vision path against HF in float32: the oracle's image cases
// (tools/reference/qwen3vl_embedding_oracle.py --images) through
// PreprocessImage, VisionReference and Reference::EmbedImage: the patch
// grid, the token ids and the embedding (cosine >= 0.999). Needs
// GUFO_NPU_QWEN3VL_EMBED_2B_MMPROJ (the mmproj GGUF) besides the text
// model's variables; exits 77 without them or without image cases.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "models/qwen3vl_embed_2b/input.hpp"
#include "models/qwen3vl_embed_2b/reference.hpp"
#include "models/qwen3vl_embed_2b/vision.hpp"
#include "oracle.hpp"

int main() {
  namespace e = gufo_npu::qwen3vl_embed_2b;
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
  e::Reference reference(*file);
  e::VisionReference vision(*vision_file);
  const e::VisionConfig& vc = vision.config();
  const int pad = tokenizer.Find(e::kImagePad);
  int failures = 0;
  for (const auto& c : j.at("image_cases")) {
    const auto name = c.at("name").get<std::string>();
    const int width = c.at("width").get<int>();
    const int height = c.at("height").get<int>();
    const auto rgb = c.at("rgb").get<std::vector<std::uint8_t>>();
    const auto grid = c.at("grid").get<std::vector<int>>();
    const auto start = std::chrono::steady_clock::now();
    const e::Patches patches =
        e::PreprocessImage(rgb.data(), width, height, vc.patch, vc.merge);
    if (grid.size() != 3 || patches.grid_h != grid[1] ||
        patches.grid_w != grid[2]) {
      std::fprintf(stderr, "FAIL: %s: patch grid %dx%d, HF %dx%d\n",
                   name.c_str(), patches.grid_h, patches.grid_w,
                   grid.size() == 3 ? grid[1] : -1,
                   grid.size() == 3 ? grid[2] : -1);
      ++failures;
      continue;
    }
    const e::ImageFeatures features = vision.Encode(patches);
    const auto ids = e::ImageInputIds(tokenizer, features.Tokens(),
                                      c.at("text").get<std::string>(),
                                      c.at("instruction").get<std::string>());
    if (ids != c.at("ids").get<std::vector<int>>()) {
      std::fprintf(stderr, "FAIL: %s: token ids differ from HF\n",
                   name.c_str());
      ++failures;
      continue;
    }
    const auto got = reference.EmbedImage(ids, pad, {features});
    const double cos =
        qwen3vl_test::Cosine(got, c.at("embedding").get<std::vector<float>>());
    std::printf(
        "%s: %d patches, %zu tokens, cosine to HF %.6f, %.1f s\n", name.c_str(),
        patches.Count(), ids.size(), cos,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    if (cos < 0.999) {
      std::fprintf(stderr, "FAIL: %s: cosine %.6f to HF < 0.999\n",
                   name.c_str(), cos);
      ++failures;
    }
  }
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS: the vision reference matches HF");
  return EXIT_SUCCESS;
}
