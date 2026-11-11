#include "models/qwen3vl_embed_2b/embedding_model.hpp"

#include <stdexcept>
#include <string>

#include "models/qwen3vl_embed_2b/npu_embedder.hpp"
#include "models/qwen3vl_embed_2b/npu_vision.hpp"
#include "models/qwen3vl_embed_2b/reference.hpp"

namespace gufo_npu::qwen3vl_embed_2b {

EmbeddingModel::EmbeddingModel(const gguf::File& file, const gguf::File* mmproj,
                               std::string_view device,
                               const std::filesystem::path& kernel_dir) {
  if (device == "cpu") {
    text_ = std::make_unique<Reference>(file);
    if (mmproj != nullptr) {
      vision_ = std::make_unique<VisionReference>(*mmproj);
    }
  } else if (device == "npu") {
    auto npu = std::make_unique<NpuEmbedder>(file, kernel_dir);
    if (mmproj != nullptr) {
      vision_ = std::make_unique<NpuVision>(*mmproj, *npu);
    }
    text_ = std::move(npu);
  } else {
    throw std::invalid_argument("device must be npu or cpu, not " +
                                std::string(device));
  }
  if (vision_ != nullptr) {
    const VisionConfig& vc = vision_->config();
    if (vc.patch * vc.merge != kImageFactor ||
        vc.out_hidden != text_->config().hidden) {
      throw std::invalid_argument(
          "the mmproj does not fit this model: patch " +
          std::to_string(vc.patch) + " x merge " + std::to_string(vc.merge) +
          " (want " + std::to_string(kImageFactor) + "), output width " +
          std::to_string(vc.out_hidden) + " (want " +
          std::to_string(text_->config().hidden) + ")");
    }
  }
}

EmbeddingModel::~EmbeddingModel() = default;

const Config& EmbeddingModel::config() const {
  return text_->config();
}

std::vector<std::vector<float>> EmbeddingModel::Embed(
    const std::vector<std::vector<int>>& inputs) {
  return text_->Embed(inputs);
}

std::vector<float> EmbeddingModel::EmbedImage(const std::vector<int>& ids,
                                              int image_pad_id,
                                              const Image& image) {
  if (vision_ == nullptr) {
    throw std::invalid_argument(
        "image inputs need the vision tower (an mmproj GGUF)");
  }
  if (image.width < 1 || image.height < 1 ||
      image.rgb.size() != static_cast<std::size_t>(image.width) *
                              static_cast<std::size_t>(image.height) * 3) {
    throw std::invalid_argument(
        "an image of " + std::to_string(image.width) + " x " +
        std::to_string(image.height) + " pixels with " +
        std::to_string(image.rgb.size()) + " RGB values");
  }
  const VisionConfig& vc = vision_->config();
  const Patches patches = PreprocessImage(image.rgb.data(), image.width,
                                          image.height, vc.patch, vc.merge);
  return text_->EmbedImage(ids, image_pad_id, {vision_->Encode(patches)});
}

}  // namespace gufo_npu::qwen3vl_embed_2b
