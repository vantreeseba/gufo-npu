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

void ParseSampling(const json& body, SamplingParams& s) {
  constexpr std::int64_t kMaxInt = std::numeric_limits<int>::max();
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

  if (const json* bias = Field(body, "logit_bias")) {
    if (!bias->is_object()) {
      throw InvalidRequest("logit_bias must be an object", "logit_bias");
    }
    for (const auto& [key, value] : bias->items()) {
      std::size_t used = 0;
      long long id = -1;
      try {
        id = std::stoll(key, &used);
      } catch (const std::exception&) {
      }
      if (used != key.size() || id < 0 || id > kMaxInt) {
        throw InvalidRequest("logit_bias keys must be token ids", "logit_bias");
      }
      if (!value.is_number() || !std::isfinite(value.get<double>()) ||
          value.get<double>() < -100 || value.get<double>() > 100) {
        throw InvalidRequest("logit_bias values must be in [-100, 100]",
                             "logit_bias");
      }
      s.logit_bias.emplace_back(static_cast<int>(id),
                                static_cast<float>(value.get<double>()));
    }
  }
}

std::vector<std::string> Stop(const json& body) {
  std::vector<std::string> out;
  if (const json* stop = Field(body, "stop")) {
    if (stop->is_string()) {
      out.push_back(stop->get<std::string>());
    } else if (stop->is_array() && stop->size() <= 16) {
      for (const auto& item : *stop) {
        if (!item.is_string()) {
          throw InvalidRequest("stop must be a string or strings", "stop");
        }
        out.push_back(item.get<std::string>());
      }
    } else {
      throw InvalidRequest("stop must be a string or up to 16 strings", "stop");
    }
  }

  return out;
}

void ParseStream(const json& body, bool& stream, bool& include_usage) {
  stream = Bool(body, "stream").value_or(false);
  if (const json* options = Field(body, "stream_options")) {
    if (!options->is_object()) {
      throw InvalidRequest("stream_options must be an object",
                           "stream_options");
    }
    include_usage = Bool(*options, "include_usage").value_or(false);
  }
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
        throw InvalidRequest("tool_choice names \"" + req.tool_name +
                                 "\", which is not in tools",
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

  req.n = static_cast<int>(Integer(body, "n", 1, kMaxChoices).value_or(1));
  req.logprobs = Bool(body, "logprobs").value_or(false);
  if (const auto top = Integer(body, "top_logprobs", 0, kMaxTopLogprobs)) {
    if (!req.logprobs) {
      throw InvalidRequest("top_logprobs needs logprobs: true", "top_logprobs");
    }
    req.top_logprobs = static_cast<int>(*top);
  }
  req.parallel_tool_calls = Bool(body, "parallel_tool_calls").value_or(true);
  if (const json* format = Field(body, "response_format")) {
    const std::string type =
        format->is_object() ? format->value("type", "") : "";
    if (type == "json_object") {
      req.format = ChatRequest::Format::kJsonObject;
    } else if (type == "json_schema") {
      const json* spec = Field(*format, "json_schema");
      const json* schema = spec != nullptr && spec->is_object()
                               ? Field(*spec, "schema")
                               : nullptr;
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

  ParseSampling(body, req.sampling);
  req.stop = Stop(body);
  ParseStream(body, req.stream, req.include_usage);
  return req;
}

CompletionRequest ParseCompletionRequest(const json& body) {
  if (!body.is_object()) {
    throw InvalidRequest("the request body must be a JSON object");
  }
  CompletionRequest req;
  const json* prompt = Field(body, "prompt");
  const auto bad_prompt = [] {
    return InvalidRequest(
        "prompt must be a string, an array of strings, an array of token ids "
        "or an array of such arrays",
        "prompt");
  };
  const auto token_ids = [&](const json& arr) {
    std::vector<int> ids;
    for (const auto& t : arr) {
      if (!t.is_number_integer() || t.get<std::int64_t>() < 0 ||
          t.get<std::int64_t>() > std::numeric_limits<int>::max()) {
        throw bad_prompt();
      }
      ids.push_back(t.get<int>());
    }
    if (ids.empty()) {
      throw bad_prompt();
    }
    return ids;
  };
  if (prompt == nullptr) {
    throw InvalidRequest("prompt is required", "prompt");
  } else if (prompt->is_string()) {
    req.texts.push_back(prompt->get<std::string>());
  } else if (prompt->is_array() && !prompt->empty()) {
    const json& first = prompt->front();
    if (first.is_string()) {
      for (const auto& p : *prompt) {
        if (!p.is_string()) {
          throw bad_prompt();
        }
        req.texts.push_back(p.get<std::string>());
      }
    } else if (first.is_number_integer()) {
      req.ids.push_back(token_ids(*prompt));
    } else if (first.is_array()) {
      for (const auto& p : *prompt) {
        if (!p.is_array()) {
          throw bad_prompt();
        }
        req.ids.push_back(token_ids(p));
      }
    } else {
      throw bad_prompt();
    }
  } else {
    throw bad_prompt();
  }
  for (const auto& text : req.texts) {
    if (text.empty()) {
      throw InvalidRequest("prompt must not be empty", "prompt");
    }
  }
  if (req.prompts() > kMaxChoices) {
    throw InvalidRequest(
        "at most " + std::to_string(kMaxChoices) + " prompts per request",
        "prompt");
  }
  if (const json* suffix = Field(body, "suffix")) {
    if (!suffix->is_string() || !suffix->get<std::string>().empty()) {
      throw Unsupported("suffix is not supported", "suffix");
    }
  }
  req.n = static_cast<int>(Integer(body, "n", 1, kMaxChoices).value_or(1));
  if (req.n * static_cast<int>(req.prompts()) > kMaxChoices) {
    throw InvalidRequest(
        "prompts times n must be at most " + std::to_string(kMaxChoices), "n");
  }
  if (const auto best_of = Integer(body, "best_of", 1, kMaxChoices);
      best_of && *best_of != req.n) {
    throw Unsupported("best_of must equal n", "best_of");
  }
  if (const auto max_tokens =
          Integer(body, "max_tokens", 1, std::numeric_limits<int>::max())) {
    req.max_tokens = static_cast<int>(*max_tokens);
  }
  if (const auto lp = Integer(body, "logprobs", 0, 5)) {
    req.logprobs = static_cast<int>(*lp);
  }
  req.echo = Bool(body, "echo").value_or(false);
  if (req.echo && req.logprobs) {
    throw Unsupported(
        "echo with logprobs is not supported (prompt tokens have no "
        "logprobs here)",
        "echo");
  }
  ParseSampling(body, req.sampling);
  req.stop = Stop(body);
  ParseStream(body, req.stream, req.include_usage);
  return req;
}

EmbeddingsRequest ParseEmbeddingsRequest(const json& body) {
  if (!body.is_object()) {
    throw InvalidRequest("the request body must be a JSON object");
  }
  EmbeddingsRequest req;
  const auto bad_input = [] {
    return InvalidRequest(
        "input must be a non-empty string, an array of them, an array of "
        "token ids or an array of such arrays",
        "input");
  };
  const auto token_ids = [&](const json& arr) {
    std::vector<int> ids;
    for (const auto& t : arr) {
      if (!t.is_number_integer() || t.get<std::int64_t>() < 0 ||
          t.get<std::int64_t>() > std::numeric_limits<int>::max()) {
        throw bad_input();
      }
      ids.push_back(t.get<int>());
    }
    if (ids.empty()) {
      throw bad_input();
    }
    return ids;
  };
  const auto text = [&](const json& s) {
    if (!s.is_string() || s.get<std::string>().empty()) {
      throw bad_input();
    }
    return EmbeddingInput{s.get<std::string>(), std::nullopt};
  };
  const json* input = Field(body, "input");
  if (input == nullptr) {
    throw InvalidRequest("input is required", "input");
  }
  if (input->is_string()) {
    req.inputs.push_back(text(*input));
  } else if (input->is_array() && !input->empty()) {
    const json& first = input->front();
    if (first.is_string()) {
      for (const auto& s : *input) {
        req.inputs.push_back(text(s));
      }
    } else if (first.is_number_integer()) {
      req.inputs.push_back({"", token_ids(*input)});
    } else if (first.is_array()) {
      for (const auto& p : *input) {
        if (!p.is_array()) {
          throw bad_input();
        }
        req.inputs.push_back({"", token_ids(p)});
      }
    } else {
      throw bad_input();
    }
  } else {
    throw bad_input();
  }
  if (req.inputs.size() > static_cast<std::size_t>(kMaxEmbeddingInputs)) {
    throw InvalidRequest("at most " + std::to_string(kMaxEmbeddingInputs) +
                             " inputs per request",
                         "input");
  }
  if (const json* format = Field(body, "encoding_format")) {
    if (!format->is_string() || (format->get<std::string>() != "float" &&
                                 format->get<std::string>() != "base64")) {
      throw InvalidRequest("encoding_format must be \"float\" or \"base64\"",
                           "encoding_format");
    }
    req.base64 = format->get<std::string>() == "base64";
  }
  if (const auto dims =
          Integer(body, "dimensions", 1, std::numeric_limits<int>::max())) {
    req.dimensions = static_cast<int>(*dims);
  }
  if (const json* instruction = Field(body, "instruction")) {
    if (!instruction->is_string()) {
      throw InvalidRequest("instruction must be a string", "instruction");
    }
    req.instruction = instruction->get<std::string>();
  }
  if (const json* user = Field(body, "user"); user && !user->is_string()) {
    throw InvalidRequest("user must be a string", "user");
  }
  return req;
}

}  // namespace gufo_npu::serve
