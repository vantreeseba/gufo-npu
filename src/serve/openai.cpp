#include "serve/openai.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace gufo_npu::serve {
namespace {

using json = nlohmann::ordered_json;

// A field that is absent or null reads as missing.
const json* Field(const json& body, const char* name) {
  const auto it = body.find(name);
  if (it == body.end() || it->is_null()) {
    return nullptr;
  }
  return &*it;
}

std::optional<double> Number(const json& body, const char* name, double lo,
                             double hi) {
  const json* v = Field(body, name);
  if (v == nullptr) {
    return std::nullopt;
  }
  if (!v->is_number()) {
    throw InvalidRequest(std::string(name) + " must be a number", name);
  }
  const double x = v->get<double>();
  if (!std::isfinite(x) || x < lo || x > hi) {
    throw InvalidRequest(std::string(name) + " is out of range", name);
  }
  return x;
}

std::optional<std::int64_t> Integer(const json& body, const char* name,
                                    std::int64_t lo, std::int64_t hi) {
  const json* v = Field(body, name);
  if (v == nullptr) {
    return std::nullopt;
  }
  if (!v->is_number_integer()) {
    throw InvalidRequest(std::string(name) + " must be an integer", name);
  }
  const auto x = v->get<std::int64_t>();
  if (x < lo || x > hi) {
    throw InvalidRequest(std::string(name) + " is out of range", name);
  }
  return x;
}

std::optional<bool> Bool(const json& body, const char* name) {
  const json* v = Field(body, name);
  if (v == nullptr) {
    return std::nullopt;
  }
  if (!v->is_boolean()) {
    throw InvalidRequest(std::string(name) + " must be a boolean", name);
  }
  return v->get<bool>();
}

ApiError Unsupported(const std::string& message, const std::string& param) {
  return ApiError(400, "invalid_request_error", "unsupported_parameter",
                  message, param);
}

bool Thinking(const json& body) {
  if (const json* kwargs = Field(body, "chat_template_kwargs")) {
    if (!kwargs->is_object()) {
      throw InvalidRequest("chat_template_kwargs must be an object",
                           "chat_template_kwargs");
    }
    if (const auto on = Bool(*kwargs, "enable_thinking")) {
      return *on;
    }
  }
  if (const json* effort = Field(body, "reasoning_effort")) {
    if (!effort->is_string()) {
      throw InvalidRequest("reasoning_effort must be a string",
                           "reasoning_effort");
    }
    const auto e = effort->get<std::string>();
    if (e == "none" || e == "off" || e == "minimal") {
      return false;
    }
    if (e == "low" || e == "medium" || e == "high") {
      return true;
    }
    throw InvalidRequest(
        "reasoning_effort must be none, minimal, low, "
        "medium or high",
        "reasoning_effort");
  }
  return false;
}

}  // namespace

nlohmann::json ErrorJson(const ApiError& error) {
  return {{"error",
           {{"message", error.what()},
            {"type", error.type},
            {"param", error.param.empty() ? nlohmann::json(nullptr)
                                          : nlohmann::json(error.param)},
            {"code", error.code.empty() ? nlohmann::json(nullptr)
                                        : nlohmann::json(error.code)}}}};
}

ApiError InvalidRequest(const std::string& message, const std::string& param) {
  return ApiError(400, "invalid_request_error", "", message, param);
}

ChatRequest ParseChatRequest(const json& body) {
  if (!body.is_object()) {
    throw InvalidRequest("the request body must be a JSON object");
  }
  ChatRequest req;

  const json* messages = Field(body, "messages");
  if (messages == nullptr || !messages->is_array() || messages->empty()) {
    throw InvalidRequest("messages must be a non-empty array", "messages");
  }
  req.prompt.messages = *messages;

  if (const json* tools = Field(body, "tools")) {
    if (!tools->is_array()) {
      throw InvalidRequest("tools must be an array", "tools");
    }
    for (const auto& tool : *tools) {
      const json* fn = tool.is_object() ? Field(tool, "function") : nullptr;
      if (!tool.is_object() || tool.value("type", "") != "function" ||
          fn == nullptr || !fn->is_object() || !fn->contains("name") ||
          !(*fn)["name"].is_string()) {
        throw InvalidRequest(
            "each tool must be {\"type\": \"function\", \"function\": "
            "{\"name\": ...}}",
            "tools");
      }
    }
    req.prompt.tools = *tools;
  }
  if (const json* choice = Field(body, "tool_choice")) {
    if (*choice == "none") {
      req.prompt.tools = json::array();
    } else if (*choice == "required") {
      if (req.prompt.tools.empty()) {
        throw InvalidRequest("tool_choice \"required\" needs tools",
                             "tool_choice");
      }
      req.tool_required = true;
    } else if (choice->is_object()) {
      const json* fn = Field(*choice, "function");
      if (choice->value("type", "") != "function" || fn == nullptr ||
          !fn->is_object() || !fn->contains("name") ||
          !(*fn)["name"].is_string()) {
        throw InvalidRequest(
            "tool_choice must be \"auto\", \"none\", \"required\" or "
            "{\"type\": \"function\", \"function\": {\"name\": ...}}",
            "tool_choice");
      }
      req.tool_name = (*fn)["name"].get<std::string>();
      const bool known =
          std::any_of(req.prompt.tools.begin(), req.prompt.tools.end(),
                      [&](const json& tool) {
                        return tool["function"]["name"] == req.tool_name;
                      });
      if (!known) {
        throw InvalidRequest(
            "tool_choice names \"" + req.tool_name + "\", which is not in tools",
            "tool_choice");
      }
      req.tool_required = true;
    } else if (*choice != "auto") {
      throw InvalidRequest(
          "tool_choice must be \"auto\", \"none\", \"required\" or a "
          "function",
          "tool_choice");
    }
  }
  req.prompt.thinking = Thinking(body);

  if (const auto n = Integer(body, "n", 1, 1 << 20); n && *n != 1) {
    throw Unsupported("n must be 1", "n");
  }
  if (Bool(body, "logprobs").value_or(false) ||
      Field(body, "top_logprobs") != nullptr) {
    throw Unsupported("logprobs are not supported", "logprobs");
  }
  if (const json* format = Field(body, "response_format")) {
    const std::string type =
        format->is_object() ? format->value("type", "") : "";
    if (type == "json_object") {
      req.format = ChatRequest::Format::kJsonObject;
    } else if (type == "json_schema") {
      const json* spec = Field(*format, "json_schema");
      const json* schema =
          spec != nullptr && spec->is_object() ? Field(*spec, "schema") : nullptr;
      if (schema == nullptr || !(schema->is_object() || schema->is_boolean())) {
        throw InvalidRequest(
            "response_format json_schema needs {\"json_schema\": "
            "{\"name\": ..., \"schema\": {...}}}",
            "response_format");
      }
      req.format = ChatRequest::Format::kJsonSchema;
      req.json_schema = *schema;
      req.json_strict = Bool(*spec, "strict").value_or(false);
    } else if (type != "text") {
      throw Unsupported(
          "response_format type must be text, json_object or json_schema",
          "response_format");
    }
  }

  constexpr std::int64_t kMaxInt = std::numeric_limits<int>::max();
  auto max_tokens = Integer(body, "max_completion_tokens", 1, kMaxInt);
  if (!max_tokens) {
    max_tokens = Integer(body, "max_tokens", 1, kMaxInt);
  }
  if (max_tokens) {
    req.max_tokens = static_cast<int>(*max_tokens);
  }

  auto& s = req.sampling;
  s.temperature = static_cast<float>(
      Number(body, "temperature", 0, 2).value_or(s.temperature));
  if (const auto top_p = Number(body, "top_p", 0, 1)) {
    if (*top_p <= 0) {
      throw InvalidRequest("top_p must be above 0", "top_p");
    }
    s.top_p = static_cast<float>(*top_p);
  }
  s.top_k = static_cast<int>(Integer(body, "top_k", 0, kMaxInt).value_or(0));
  s.min_p = static_cast<float>(Number(body, "min_p", 0, 1).value_or(0));
  s.presence_penalty =
      static_cast<float>(Number(body, "presence_penalty", -2, 2).value_or(0));
  s.frequency_penalty =
      static_cast<float>(Number(body, "frequency_penalty", -2, 2).value_or(0));
  if (const auto rep = Number(body, "repetition_penalty", 0, 10)) {
    if (*rep <= 0) {
      throw InvalidRequest("repetition_penalty must be above 0",
                           "repetition_penalty");
    }
    s.repetition_penalty = static_cast<float>(*rep);
  }
  if (const auto seed =
          Integer(body, "seed", std::numeric_limits<std::int64_t>::min(),
                  std::numeric_limits<std::int64_t>::max())) {
    s.seed = static_cast<std::uint64_t>(*seed);
  }

  if (const json* stop = Field(body, "stop")) {
    if (stop->is_string()) {
      req.stop.push_back(stop->get<std::string>());
    } else if (stop->is_array() && stop->size() <= 16) {
      for (const auto& item : *stop) {
        if (!item.is_string()) {
          throw InvalidRequest("stop must be a string or strings", "stop");
        }
        req.stop.push_back(item.get<std::string>());
      }
    } else {
      throw InvalidRequest("stop must be a string or up to 16 strings", "stop");
    }
  }

  req.stream = Bool(body, "stream").value_or(false);
  if (const json* options = Field(body, "stream_options")) {
    if (!options->is_object()) {
      throw InvalidRequest("stream_options must be an object",
                           "stream_options");
    }
    req.include_usage = Bool(*options, "include_usage").value_or(false);
  }
  return req;
}

}  // namespace gufo_npu::serve
