#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// OpenAI chat messages to the model's prompt text, and the model's output
// text back to OpenAI message parts (content, reasoning, tool calls).
namespace gufo_npu::serve {

// Ordered JSON: the template prints tool schemas and call arguments in the
// order the client sent them (parse request bodies with ordered_json too).
struct ChatPrompt {
  nlohmann::ordered_json messages = nlohmann::ordered_json::array();
  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  bool thinking = false;  // let the model reason in a think block first
};

// A content part that is an image: OpenAI's {"type": "image_url",
// "image_url": {"url": ...}}, or any part with that type or key. A prompt's
// images are numbered in message and part order, as the request parser
// (DecodeChatImages, openai.hpp) decodes them.
inline bool IsImagePart(const nlohmann::ordered_json& part) {
  if (!part.is_object()) {
    return false;
  }
  const auto type = part.find("type");
  return part.contains("image_url") ||
         (type != part.end() && *type == "image_url");
}

// A prompt as the template renders it, cut where its images' pads go:
// text[k], then image images[k]'s pads, then text[k + 1]; text has
// images.size() + 1 pieces. images[k] numbers the image among the prompt's
// image parts (IsImagePart order). The template may drop an image part
// (one in a dropped system message or an earlier turn's reasoning), which
// then appears nowhere, and could place one twice.
struct RenderedPrompt {
  std::vector<std::string> text;
  std::vector<int> images;
};

// One piece of parsed output, in order. Streaming sends each as a delta.
struct OutputPart {
  enum class Kind { kContent, kReasoning, kToolCall };
  Kind kind = Kind::kContent;
  std::string text;       // kContent / kReasoning: text to append
  std::string tool_id;    // kToolCall: "call_..." id
  std::string tool_name;  // kToolCall
  std::string arguments;  // kToolCall: JSON object text
};

class OutputParser {
public:
  virtual ~OutputParser() = default;
  // Feeds decoded text (whole UTF-8 characters) and returns the parts that
  // are complete. Text that may still turn out to be markup is held back.
  virtual std::vector<OutputPart> Feed(const std::string& text) = 0;
  // End of output: returns what was held back.
  virtual std::vector<OutputPart> Finish() = 0;
};

class ChatFormat {
public:
  virtual ~ChatFormat() = default;
  // The prompt text, ending with the assistant's generation prompt; a final
  // assistant message is continued instead (prefill). Each image is one
  // ImagePadToken, as the template writes it. Throws std::invalid_argument
  // for messages the model cannot take.
  virtual std::string Render(const ChatPrompt& prompt) const = 0;
  // The same, cut at each image's pad so the caller can repeat the pad
  // once per image token. A format that takes no images returns Render's
  // text as the one piece.
  virtual RenderedPrompt RenderImages(const ChatPrompt& prompt) const {
    return {{Render(prompt)}, {}};
  }
  // The token an image's features replace, once per image token (e.g.
  // "<|image_pad|>"); empty: the format takes no images.
  virtual std::string ImagePadToken() const { return {}; }
  virtual std::unique_ptr<OutputParser> NewParser(
      const ChatPrompt& prompt) const = 0;
  // Special tokens that end a reply (e.g. "<|im_end|>").
  virtual std::vector<std::string> StopTokens() const = 0;
};

// The Qwen3.5 / Qwen3.6 chat template (with tools, think blocks and
// images).
std::unique_ptr<ChatFormat> NewQwen35ChatFormat();

}  // namespace gufo_npu::serve
