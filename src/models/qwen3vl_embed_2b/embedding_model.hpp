#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

#include "core/gguf.hpp"
#include "models/qwen3vl_embed_2b/config.hpp"
#include "models/qwen3vl_embed_2b/embedder.hpp"
#include "models/qwen3vl_embed_2b/vision.hpp"

// Qwen3-VL-Embedding-2B as the CLI and the server run it: the text embedder
// on a device ("npu", or "cpu" for the references) and, given the mmproj
// GGUF, the vision tower on the same device, an image input going through
// them as the reference embedder runs it (VISION.md).
namespace gufo_npu::qwen3vl_embed_2b {

// An image as the model takes it: width x height RGB8, row-major and
// unpadded, its alpha already flattened (FlattenAlpha).
struct Image {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;
};

class EmbeddingModel {
public:
  // Keeps pointers into `file` and `mmproj` (nullptr: text only), which
  // must outlive it; `kernel_dir` serves "npu". Throws
  // std::invalid_argument for another device or a vision tower that does
  // not fit the text model, and what the embedders' constructors throw.
  EmbeddingModel(const gguf::File& file, const gguf::File* mmproj,
                 std::string_view device,
                 const std::filesystem::path& kernel_dir);
  ~EmbeddingModel();
  EmbeddingModel(const EmbeddingModel&) = delete;
  EmbeddingModel& operator=(const EmbeddingModel&) = delete;

  const Config& config() const;
  bool has_vision() const { return vision_ != nullptr; }

  // Text inputs, as Embedder::Embed.
  std::vector<std::vector<float>> Embed(
      const std::vector<std::vector<int>>& inputs);
  // The unit embedding of one input holding `image`: `ids` from
  // ImageInputIds with ImageTokens(image.width, image.height) image tokens.
  // Throws std::invalid_argument without a vision tower, for an image
  // whose pixels are not width x height x 3 values, or when the ids
  // disagree with the image.
  std::vector<float> EmbedImage(const std::vector<int>& ids, int image_pad_id,
                                const Image& image);

private:
  std::unique_ptr<Embedder> text_;
  // Destroyed first: NpuVision runs on the NpuEmbedder's GEMM program.
  std::unique_ptr<VisionEncoder> vision_;
};

}  // namespace gufo_npu::qwen3vl_embed_2b
