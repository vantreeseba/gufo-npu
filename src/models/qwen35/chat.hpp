#pragma once

#include <string>
#include <vector>

// Qwen3.5 chat formatting (the subset of the GGUF chat template without
// tools or images): `<|im_start|>role\ncontent<|im_end|>\n` per message, then
// the assistant generation prompt.
namespace gufo_npu::qwen35 {

struct Message {
  std::string role;  // "system", "user" or "assistant"
  std::string content;
};

inline constexpr char kImStart[] = "<|im_start|>";
inline constexpr char kImEnd[] = "<|im_end|>";
inline constexpr char kEndOfText[] = "<|endoftext|>";

// With `thinking` false the template pre-fills an empty think block, which
// is what the model expects for direct answers.
std::string RenderChat(const std::vector<Message>& messages, bool thinking);

}  // namespace gufo_npu::qwen35
