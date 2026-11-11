// Image inputs as /v1/embeddings runs them, through the server's engine
// (serve::LoadEngine with the mmproj): each of the oracle's image cards
// (tools/reference/qwen3vl_embedding_oracle.py --images) written as a PNG,
// decoded by image::Decode, its ids from Engine::EmbeddingIds and its
// embedding from Engine::Embed, against HF in float32 (cosine >= 0.99, as
// qwen3vl_embed_2b_npu_vision_test, which runs the same cards through
// NpuVision directly). One batch then mixes a text input with all the
// cards; each input keeps the vector it gets alone. Needs an NPU, the
// model's variables and GUFO_NPU_QWEN3VL_EMBED_2B_MMPROJ; exits 77 without
// them.
#include <png.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/image.hpp"
#include "oracle.hpp"
#include "serve/engine.hpp"

namespace {

namespace serve = gufo_npu::serve;

constexpr double kEmbeddingCosine = 0.99;
// The same input alone and in a batch: the same NPU products.
constexpr double kBatchCosine = 0.99999;

std::vector<std::uint8_t> WritePng(const std::vector<std::uint8_t>& rgb,
                                   int width, int height) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  image.width = static_cast<png_uint_32>(width);
  image.height = static_cast<png_uint_32>(height);
  image.format = PNG_FORMAT_RGB;
  png_alloc_size_t size = 0;
  if (!png_image_write_to_memory(&image, nullptr, &size, 0, rgb.data(), 0,
                                 nullptr)) {
    throw std::runtime_error(std::string("PNG size: ") + image.message);
  }
  std::vector<std::uint8_t> out(size);
  if (!png_image_write_to_memory(&image, out.data(), &size, 0, rgb.data(), 0,
                                 nullptr)) {
    throw std::runtime_error(std::string("PNG write: ") + image.message);
  }
  out.resize(size);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen3vl_embed_2b_engine_image_test KERNELS\n");
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

  serve::EngineOptions options;
  options.model_path = oracle->gguf;
  options.device = "npu";
  options.kernels = argv[1];
  options.mmproj_path = mmproj;
  const auto engine = serve::LoadEngine(options);
  int failures = 0;
  if (!engine->info().embedding_images) {
    std::fprintf(stderr, "FAIL: the engine does not report image inputs\n");
    return 1;
  }

  std::vector<serve::PreparedEmbedding> batch;
  std::vector<std::vector<float>> alone;
  // A text input first: the batch below mixes it with the images.
  const auto& text_case = oracle->cases.front();
  batch.push_back({engine->EmbeddingIds({text_case.text, std::nullopt, nullptr},
                                        text_case.instruction),
                   nullptr});
  alone.push_back(engine->Embed({batch.back()}).front());

  for (const auto& c : j.at("image_cases")) {
    const auto name = c.at("name").get<std::string>();
    const int width = c.at("width").get<int>();
    const int height = c.at("height").get<int>();
    const auto rgb = c.at("rgb").get<std::vector<std::uint8_t>>();
    auto decoded = gufo_npu::image::Decode(WritePng(rgb, width, height));
    if (decoded.rgb != rgb) {
      std::fprintf(stderr, "FAIL: %s: the PNG round trip changed pixels\n",
                   name.c_str());
      ++failures;
      continue;
    }
    serve::EmbeddingInput input;
    input.text = c.at("text").get<std::string>();
    input.image =
        std::make_shared<const gufo_npu::image::Decoded>(std::move(decoded));
    const auto ids =
        engine->EmbeddingIds(input, c.at("instruction").get<std::string>());
    if (ids != c.at("ids").get<std::vector<int>>()) {
      std::fprintf(stderr, "FAIL: %s: token ids differ from HF\n",
                   name.c_str());
      ++failures;
      continue;
    }
    batch.push_back({ids, input.image});
    alone.push_back(engine->Embed({batch.back()}).front());
    const double cos = qwen3vl_test::Cosine(
        alone.back(), c.at("embedding").get<std::vector<float>>());
    std::printf("%s: %zu tokens, cosine to HF %.6f\n", name.c_str(), ids.size(),
                cos);
    if (cos < kEmbeddingCosine) {
      std::fprintf(stderr, "FAIL: %s: cosine %.6f to HF < %.2f\n", name.c_str(),
                   cos, kEmbeddingCosine);
      ++failures;
    }
  }

  const auto together = engine->Embed(batch);
  for (std::size_t i = 0; i < batch.size(); ++i) {
    const double cos = qwen3vl_test::Cosine(together[i], alone[i]);
    if (cos < kBatchCosine) {
      std::fprintf(stderr, "FAIL: input %zu: cosine %.7f alone vs batched\n", i,
                   cos);
      ++failures;
    }
  }
  std::printf("batch of %zu inputs (1 text, %zu images): alone = batched\n",
              batch.size(), batch.size() - 1);

  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::puts("PASS: qwen3vl_embed_2b image inputs through the engine");
  return 0;
}
