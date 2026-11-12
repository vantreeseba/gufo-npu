#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/image.hpp"
#include "core/sampling.hpp"
#include "core/tokenizer.hpp"

// The server's view of a loaded model: token ids in, token ids out. One
// generation runs at a time; the server serializes requests. The engine owns
// any state kept between requests (the prompt cache), so callers always pass
// the whole prompt.
namespace gufo_npu::serve {

struct ModelInfo {
  std::string id;            // the name clients use, e.g. "qwen3.6-35b-a3b"
  std::string architecture;  // GGUF architecture, e.g. "qwen35moe"
  int context_length = 0;    // most prompt + generated tokens per request
  int vocab = 0;
  bool can_draft = false;  // MTP speculative decoding available
  // An embedding model (POST /v1/embeddings, no generation): the size of
  // its vectors, and the smallest size `dimensions` may cut them to.
  int embedding_length = 0;
  int embedding_min_dimensions = 0;
  // Image inputs too (an embedding model loaded with its vision tower).
  bool embedding_images = false;
  // The model's package can load a vision tower (TakesMmproj); a chat
  // model loaded with one sets chat_images: its prompts may hold images
  // (GenerateRequest::images, Engine::ChatImageLayout).
  bool takes_mmproj = false;
  bool chat_images = false;
};

// One /v1/embeddings input: text, content as token ids (an OpenAI token
// array), or an image with `text` as its optional caption; the model's
// input template wraps each.
struct EmbeddingInput {
  std::string text;
  std::optional<std::vector<int>> tokens;
  std::shared_ptr<const image::Decoded> image;
};

// An input as the model embeds it: EmbeddingIds' ids, and the image of an
// image input.
struct PreparedEmbedding {
  std::vector<int> ids;
  std::shared_ptr<const image::Decoded> image;
};

// How an image enters a chat prompt, as the model's processor resizes it:
// the size it is resized to, that size in patches, and the image-pad ids
// it takes (one per merged token: grid_w * grid_h / merge^2).
struct ImageLayout {
  int width = 0;  // resized, in pixels
  int height = 0;
  int grid_w = 0;  // in patches
  int grid_h = 0;
  int tokens = 0;
};

// An image of a chat prompt, as the engine receives it. The prompt holds
// layout.tokens image-pad ids (the chat format's ImagePadToken, Qwen's
// <|image_pad|>) from `offset`, between the template's vision start and end
// tokens; the model puts the image's features in those rows. Every image's
// pads are the same id, so a prompt cache must also compare `hash` before
// it reuses a prefix that holds an image.
struct PromptImage {
  std::shared_ptr<const image::Decoded> image;  // as decoded, alpha kept
  std::uint64_t hash = 0;  // of the encoded file (FNV-1a, 64-bit)
  ImageLayout layout;
  int offset = 0;  // index in GenerateRequest::prompt of its first pad
};

struct GenerateRequest {
  std::vector<int> prompt;
  // The prompt's images in order, for an engine with info().chat_images
  // (others refuse a request that has any); see CheckPromptImages.
  std::vector<PromptImage> images;
  int max_tokens = 0;
  SamplingParams sampling;
  std::vector<int> stop_ids;  // end the generation (the id is reported)
  bool draft = true;          // use MTP drafts when the model can
  // Checked between tokens; when set, the generation ends as kCancelled.
  const std::atomic<bool>* cancel = nullptr;
};

// Throws std::invalid_argument unless request.images fit request.prompt:
// each has an image and a run of layout.tokens >= 1 ids of `image_pad_id`
// from its offset, the runs in order and inside the prompt, and no other
// prompt id is `image_pad_id`.
inline void CheckPromptImages(const GenerateRequest& request,
                              int image_pad_id) {
  const auto& prompt = request.prompt;
  std::size_t next = 0;  // the first id not yet checked
  for (const auto& img : request.images) {
    const auto begin = static_cast<std::size_t>(img.offset);
    const auto end = begin + static_cast<std::size_t>(img.layout.tokens);
    if (!img.image || img.offset < 0 || img.layout.tokens < 1 ||
        begin < next || end > prompt.size()) {
      throw std::invalid_argument("an image's pads do not fit the prompt");
    }
    for (std::size_t i = next; i < end; ++i) {
      if ((prompt[i] == image_pad_id) != (i >= begin)) {
        throw std::invalid_argument(
            "the prompt's image pads do not match its images");
      }
    }
    next = end;
  }
  for (std::size_t i = next; i < prompt.size(); ++i) {
    if (prompt[i] == image_pad_id) {
      throw std::invalid_argument(
          "the prompt's image pads do not match its images");
    }
  }
}

enum class FinishReason { kStop, kLength, kCancelled };

struct GenerateResult {
  std::vector<int> tokens;  // generated ids, including a final stop id
  FinishReason finish = FinishReason::kLength;
  int prompt_tokens = 0;
  int cached_prompt_tokens = 0;  // prompt tokens reused from the cache
  double prefill_seconds = 0;    // time to the first generated token
  double decode_seconds = 0;
  int steps = 0;     // decode passes
  int accepted = 0;  // MTP drafts kept
};

class Engine {
public:
  virtual ~Engine() = default;

  virtual const ModelInfo& info() const = 0;
  virtual const Tokenizer& tokenizer() const = 0;

  // Text and ids, through tokenizer() unless overridden (the server uses
  // only these, so test fakes need no GGUF vocabulary).
  virtual std::vector<int> Encode(std::string_view text) const {
    return tokenizer().Encode(text);
  }
  virtual std::string TokenBytes(int id) const {
    return tokenizer().TokenBytes(id);
  }
  virtual int FindToken(std::string_view token) const {  // -1 if absent
    return tokenizer().Find(token);
  }
  // An added token (control or user-defined, e.g. "<|im_end|>"), whose
  // TokenBytes is its text, not bytes the model writes.
  virtual bool IsSpecialToken(int id) const {
    return tokenizer().IsSpecial(id);
  }

  // Runs one generation. `on_token` sees each generated id as it is
  // produced (including a stop id); returning false ends the generation
  // as kCancelled.
  virtual GenerateResult Generate(const GenerateRequest& request,
                                  const std::function<bool(int)>& on_token) = 0;

  // Chat models with info().chat_images only; others throw
  // std::invalid_argument. A width x height image's layout in a chat
  // prompt, resized as the model's processor resizes it, to at most
  // EngineOptions::max_image_tokens tokens. Needs no weights, so it works
  // before the model loads. Throws std::invalid_argument for an image the
  // model cannot take (e.g. an aspect ratio above 200).
  virtual ImageLayout ChatImageLayout(int width, int height) const;

  // Embedding models (info().embedding_length > 0) only; others throw
  // std::invalid_argument.
  //
  // The token ids the model embeds for `input`: its template around the
  // text or ids, with `instruction` (empty: the model's default) as the
  // task. Needs only the tokenizer, so it works before the model loads.
  virtual std::vector<int> EmbeddingIds(const EmbeddingInput& input,
                                        std::string_view instruction) const;
  // One unit vector of info().embedding_length per input, in order. One
  // call runs at a time, as Generate. Throws std::invalid_argument for an
  // image input when info().embedding_images is false.
  virtual std::vector<std::vector<float>> Embed(
      const std::vector<PreparedEmbedding>& inputs);
};

inline ImageLayout Engine::ChatImageLayout(int, int) const {
  throw std::invalid_argument(info().id + " takes no images");
}

inline std::vector<int> Engine::EmbeddingIds(const EmbeddingInput&,
                                             std::string_view) const {
  throw std::invalid_argument(info().id + " does not make embeddings");
}

inline std::vector<std::vector<float>> Engine::Embed(
    const std::vector<PreparedEmbedding>&) {
  throw std::invalid_argument(info().id + " does not make embeddings");
}

// The default of EngineOptions::max_image_tokens (--max-image-tokens).
inline constexpr int kDefaultMaxImageTokens = 1800;

struct EngineOptions {
  std::string model_path;              // GGUF
  std::string device = "npu";          // "npu" or "cpu" (the CPU reference)
  std::optional<std::string> kernels;  // kernel dir; default: installed
  std::string model_id;                // ModelInfo::id; default: from the GGUF
  // Most prompt + generated tokens per request; 0: kDefaultContextLength,
  // capped by the GGUF's trained context length (model_registry.hpp).
  int context_length = 0;
  // Prompt-cache checkpoints kept between requests (reuse of a multi-turn
  // chat's history); 0: off. The CPU reference has no cache.
  int prompt_cache = 2;
  // A vision tower (its mmproj GGUF), for image inputs; empty: text only.
  // Only a model whose package takes one (TakesMmproj) accepts it: today
  // the embedding model.
  std::string mmproj_path;
  // Most merged image tokens per image (the processor's smart_resize
  // max_pixels, in tokens of 32 x 32 pixels). A model with a vision tower
  // checks it against its own range at probe and load.
  int max_image_tokens = kDefaultMaxImageTokens;
};

// Whether the GGUF's model package can load a vision tower
// (EngineOptions::mmproj_path); reads only the GGUF's metadata. Throws
// what gguf::File::Open and the config readers throw.
bool TakesMmproj(const std::string& model_path);

// Loads the GGUF's model on the device, picking the model package by
// architecture and shape (Qwen3.5-2B, Qwen3.5-4B, Qwen3.6-35B-A3B, and the
// embedding model Qwen3-VL-Embedding-2B).
std::unique_ptr<Engine> LoadEngine(const EngineOptions& options);

}  // namespace gufo_npu::serve
