#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// OpenAI chat messages to the model's prompt text, and the model's output
// text back to OpenAI message parts (content, reasoning, tool calls).
namespace gufo_npu::serve {

struct ChatPrompt {
  nlohmann::json messages = nlohmann::json::array();  // OpenAI messages
  nlohmann::json tools = nlohmann::json::array();     // OpenAI tools
  bool thinking = false;  // let the model reason in a think block first
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
  // Throws std::invalid_argument for messages the model cannot take.
  virtual std::string Render(const ChatPrompt& prompt) const = 0;
  virtual std::unique_ptr<OutputParser> NewParser(
      const ChatPrompt& prompt) const = 0;
  // Special tokens that end a reply (e.g. "<|im_end|>").
  virtual std::vector<std::string> StopTokens() const = 0;
};

// The Qwen3.5 / Qwen3.6 chat template (with tools and think blocks).
std::unique_ptr<ChatFormat> NewQwen35ChatFormat();

}  // namespace gufo_npu::serve
