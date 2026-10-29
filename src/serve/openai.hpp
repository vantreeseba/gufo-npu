#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/sampling.hpp"
#include "serve/chat_format.hpp"

// The OpenAI chat completions request as the server takes it, and the
// OpenAI error shape.
namespace gufo_npu::serve {

// A request the server refuses, sent as
// {"error": {"message", "type", "param", "code"}} with `status`.
struct ApiError : std::runtime_error {
  ApiError(int status, std::string type, std::string code,
           const std::string& message, std::string param = "")
      : std::runtime_error(message),
        status(status),
        type(std::move(type)),
        code(std::move(code)),
        param(std::move(param)) {}

  int status;
  std::string type;  // e.g. "invalid_request_error"
  std::string code;  // e.g. "context_length_exceeded"
  std::string param;
};

nlohmann::json ErrorJson(const ApiError& error);

// 400 invalid_request_error for `param`.
ApiError InvalidRequest(const std::string& message,
                        const std::string& param = "");

// Most choices per request (n), and most top_logprobs.
inline constexpr int kMaxChoices = 8;
inline constexpr int kMaxTopLogprobs = 20;

struct ChatRequest {
  ChatPrompt prompt;
  std::optional<int> max_tokens;  // max_completion_tokens or max_tokens
  SamplingParams sampling;        // logit_bias ids are checked by the server
  std::vector<std::string> stop;
  bool stream = false;
  bool include_usage = false;  // stream_options.include_usage
  int n = 1;                   // choices, generated one after another
  bool logprobs = false;
  int top_logprobs = 0;             // 0..kMaxTopLogprobs, with logprobs
  bool parallel_tool_calls = true;  // false: end after the first tool call
  // Structured output (response_format): constrained to a JSON object or
  // to JSON matching a schema.
  enum class Format { kText, kJsonObject, kJsonSchema };
  Format format = Format::kText;
  nlohmann::ordered_json json_schema;  // kJsonSchema
  bool json_strict = false;            // json_schema.strict
  // tool_choice "required", or a named function (tool_name): the reply is
  // constrained to tool calls.
  bool tool_required = false;
  std::string tool_name;
};

// A /v1/completions request (the legacy text API): the prompt goes to the
// model as is, without the chat template.
struct CompletionRequest {
  // Each prompt as text, or as token ids (then `texts` is empty).
  std::vector<std::string> texts;
  std::vector<std::vector<int>> ids;
  int max_tokens = 16;  // OpenAI's default for this API
  SamplingParams sampling;
  std::vector<std::string> stop;
  bool stream = false;
  bool include_usage = false;
  bool echo = false;
  std::optional<int> logprobs;  // 0..5: top tokens per position
  int n = 1;

  std::size_t prompts() const {
    return texts.empty() ? ids.size() : texts.size();
  }
};

// Reads a /v1/chat/completions body, parsed as ordered JSON so messages
// and tools keep the client's key order. Throws ApiError for anything the
// server does not support or that is out of range.
ChatRequest ParseChatRequest(const nlohmann::ordered_json& body);

// Reads a /v1/completions body. Throws ApiError as ParseChatRequest does.
CompletionRequest ParseCompletionRequest(const nlohmann::ordered_json& body);

}  // namespace gufo_npu::serve
