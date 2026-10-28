#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/sampling.hpp"

namespace gufo_npu {
class Tokenizer;
}

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
};

struct GenerateRequest {
  std::vector<int> prompt;
  int max_tokens = 0;
  SamplingParams sampling;
  std::vector<int> stop_ids;  // end the generation (the id is reported)
  bool draft = true;          // use MTP drafts when the model can
  // Checked between tokens; when set, the generation ends as kCancelled.
  const std::atomic<bool>* cancel = nullptr;
};

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

  // Runs one generation. `on_token` sees each generated id as it is
  // produced (including a stop id); returning false ends the generation
  // as kCancelled.
  virtual GenerateResult Generate(const GenerateRequest& request,
                                  const std::function<bool(int)>& on_token) = 0;
};

struct EngineOptions {
  std::string model_path;              // GGUF
  std::string device = "npu";          // "npu" or "cpu" (the CPU reference)
  std::optional<std::string> kernels;  // kernel dir; default: installed
  std::string model_id;                // ModelInfo::id; default: from the GGUF
  // Most prompt + generated tokens per request; 0: kDefaultContextLength,
  // capped by the GGUF's trained context length (model_registry.hpp).
  int context_length = 0;
};

// Loads the GGUF's model on the device, picking the model package by
// architecture and shape (Qwen3.5-2B, Qwen3.5-4B, Qwen3.6-35B-A3B).
std::unique_ptr<Engine> LoadEngine(const EngineOptions& options);

}  // namespace gufo_npu::serve
