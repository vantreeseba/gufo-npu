// The Qwen3.5 / Qwen3.6 chat template, as the GGUFs ship it (Unsloth's
// copy with developer roles and tool calling), rendered the way Hugging Face
// transformers renders it. tools/reference/qwen35_chat_template.py renders
// the fixtures this is tested against (tests/serve/fixtures).
//
// The three models' templates differ only where this never looks: the
// default of enable_thinking (always given here) and the 35B's
// preserve_thinking switch (never set).

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "serve/chat_format.hpp"

namespace gufo_npu::serve {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::string_view kImStart = "<|im_start|>";
constexpr std::string_view kImEnd = "<|im_end|>\n";
constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kToolCallOpen = "<tool_call>";
constexpr std::string_view kToolCallClose = "</tool_call>";

// ---- Python string semantics the template relies on ----

// One code point at `i` (valid UTF-8 assumed; a stray byte counts as one).
char32_t DecodeAt(std::string_view s, std::size_t i, std::size_t* len) {
  const auto c = static_cast<unsigned char>(s[i]);
  int n = c < 0x80           ? 1
          : (c >> 5) == 0x6  ? 2
          : (c >> 4) == 0xE  ? 3
          : (c >> 3) == 0x1E ? 4
                             : 1;
  if (i + n > s.size()) {
    n = 1;
  }
  char32_t cp = n == 1 ? c : c & (0x7F >> n);
  for (int k = 1; k < n; ++k) {
    cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  }
  *len = static_cast<std::size_t>(n);
  return cp;
}

// Python's str.isspace().
bool IsPySpace(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 ||
         c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
         c == 0x3000;
}

// Jinja's trim filter: Python's str.strip().
std::string Trim(std::string_view s) {
  std::size_t begin = 0;
  while (begin < s.size()) {
    std::size_t len = 0;
    if (!IsPySpace(DecodeAt(s, begin, &len))) {
      break;
    }
    begin += len;
  }
  std::size_t end = s.size();
  while (end > begin) {
    std::size_t start = end - 1;
    while (start > begin &&
           (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) {
      --start;
    }
    std::size_t len = 0;
    if (!IsPySpace(DecodeAt(s, start, &len)) || start + len != end) {
      break;
    }
    end = start;
  }
  return std::string(s.substr(begin, end - begin));
}

bool IsAsciiSpace(char c) {
  return c == ' ' || c == '\n' || c == '\t' || c == '\r';
}

std::string_view LStripNewlines(std::string_view s) {
  while (!s.empty() && s.front() == '\n') {
    s.remove_prefix(1);
  }
  return s;
}

std::string_view RStripNewlines(std::string_view s) {
  while (!s.empty() && s.back() == '\n') {
    s.remove_suffix(1);
  }
  return s;
}

// Python's repr(float): the shortest round-trip digits, fixed notation for
// decimal exponents in [-4, 16), scientific otherwise.
std::string PyFloat(double v) {
  if (std::isnan(v)) {
    return "NaN";
  }
  if (std::isinf(v)) {
    return v < 0 ? "-Infinity" : "Infinity";
  }
  char buf[64];
  const auto r =
      std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  std::string_view s(buf, static_cast<std::size_t>(r.ptr - buf));
  std::string out;
  if (s.front() == '-') {
    out = "-";
    s.remove_prefix(1);
  }
  const auto e = s.find('e');
  std::string digits;
  for (const char c : s.substr(0, e)) {
    if (c != '.') {
      digits += c;
    }
  }
  int exp = 0;
  const auto exp_text = s.substr(e + 1);
  std::from_chars(exp_text.data() + (exp_text.front() == '+' ? 1 : 0),
                  exp_text.data() + exp_text.size(), exp);
  const int n = static_cast<int>(digits.size());
  if (exp >= -4 && exp < 16) {
    const int point = exp + 1;
    if (point <= 0) {
      out += "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
    } else if (point >= n) {
      out +=
          digits + std::string(static_cast<std::size_t>(point - n), '0') + ".0";
    } else {
      out += digits.substr(0, static_cast<std::size_t>(point)) + "." +
             digits.substr(static_cast<std::size_t>(point));
    }
  } else {
    out += digits.substr(0, 1);
    if (n > 1) {
      out += "." + digits.substr(1);
    }
    char tail[16];
    std::snprintf(tail, sizeof tail, "e%c%02d", exp < 0 ? '-' : '+',
                  exp < 0 ? -exp : exp);
    out += tail;
  }
  return out;
}

void AppendJsonString(std::string_view s, std::string& out) {
  out += '"';
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (c < 0x20) {
          char esc[8];
          std::snprintf(esc, sizeof esc, "\\u%04x", c);
          out += esc;
        } else {
          out += ch;
        }
    }
  }
  out += '"';
}

// json.dumps(value, ensure_ascii=False): the template's tojson.
void AppendPyJson(const Json& v, std::string& out) {
  switch (v.type()) {
    case Json::value_t::null:
      out += "null";
      break;
    case Json::value_t::boolean:
      out += v.get<bool>() ? "true" : "false";
      break;
    case Json::value_t::number_integer:
      out += std::to_string(v.get<std::int64_t>());
      break;
    case Json::value_t::number_unsigned:
      out += std::to_string(v.get<std::uint64_t>());
      break;
    case Json::value_t::number_float:
      out += PyFloat(v.get<double>());
      break;
    case Json::value_t::string:
      AppendJsonString(v.get_ref<const std::string&>(), out);
      break;
    case Json::value_t::array: {
      out += '[';
      bool first = true;
      for (const auto& item : v) {
        if (!first) {
          out += ", ";
        }
        first = false;
        AppendPyJson(item, out);
      }
      out += ']';
      break;
    }
    case Json::value_t::object: {
      out += '{';
      bool first = true;
      for (const auto& [key, item] : v.items()) {
        if (!first) {
          out += ", ";
        }
        first = false;
        AppendJsonString(key, out);
        out += ": ";
        AppendPyJson(item, out);
      }
      out += '}';
      break;
    }
    default:
      throw std::invalid_argument("chat: unsupported JSON value");
  }
}

std::string PyJson(const Json& v) {
  std::string out;
  AppendPyJson(v, out);
  return out;
}

// Jinja's `| string` / `{{ value }}` on a scalar: Python's str().
std::string PyStr(const Json& v) {
  switch (v.type()) {
    case Json::value_t::string:
      return v.get<std::string>();
    case Json::value_t::null:
      return "None";
    case Json::value_t::boolean:
      return v.get<bool>() ? "True" : "False";
    case Json::value_t::number_float:
      return PyFloat(v.get<double>());
    default:
      return PyJson(v);
  }
}

std::vector<std::string_view> Split(std::string_view s, std::string_view sep) {
  std::vector<std::string_view> parts;
  std::size_t start = 0;
  for (auto pos = s.find(sep); pos != std::string_view::npos;
       pos = s.find(sep, start)) {
    parts.push_back(s.substr(start, pos - start));
    start = pos + sep.size();
  }
  parts.push_back(s.substr(start));
  return parts;
}

// ---- The template ----

const std::string& Role(const Json& message) {
  if (!message.is_object() || !message.contains("role") ||
      !message["role"].is_string()) {
    throw std::invalid_argument("chat: every message needs a string role");
  }
  return message["role"].get_ref<const std::string&>();
}

bool IsSystem(const std::string& role) {
  return role == "system" || role == "developer";
}

// render_content: a string, a list of text parts, or nothing.
std::string RenderContent(const Json& message) {
  if (!message.contains("content")) {
    return {};
  }
  const Json& content = message["content"];
  if (content.is_null()) {
    return {};
  }
  if (content.is_string()) {
    return content.get<std::string>();
  }
  if (!content.is_array()) {
    throw std::invalid_argument("chat: content must be a string or parts");
  }
  std::string out;
  for (const auto& item : content) {
    if (!item.is_object()) {
      throw std::invalid_argument("chat: content parts must be objects");
    }
    const auto type = item.contains("type") && item["type"].is_string()
                          ? item["type"].get<std::string>()
                          : std::string();
    if (item.contains("image") || item.contains("image_url") ||
        type == "image" || item.contains("video") || type == "video") {
      throw std::invalid_argument(
          "chat: images and video are not supported by this model build");
    }
    if (!item.contains("text")) {
      throw std::invalid_argument("chat: unsupported content part");
    }
    out += PyStr(item["text"]);
  }
  return out;
}

// A tool call's arguments as the template wants them: an object. OpenAI
// sends a JSON string, which is parsed when it holds an object.
Json Arguments(const Json& function) {
  if (!function.contains("arguments")) {
    return nullptr;
  }
  const Json& args = function["arguments"];
  if (args.is_string()) {
    auto parsed = Json::parse(args.get_ref<const std::string&>(), nullptr,
                              /*allow_exceptions=*/false);
    if (parsed.is_object()) {
      return parsed;
    }
  }
  return args;
}

void RenderToolCalls(const Json& calls, const std::string& content,
                     std::string& out) {
  if (calls.is_null() || (calls.is_array() && calls.empty())) {
    return;
  }
  if (!calls.is_array()) {
    throw std::invalid_argument("chat: tool_calls must be a list");
  }
  bool first = true;
  for (const auto& call : calls) {
    if (!call.is_object()) {
      throw std::invalid_argument("chat: a tool call must be an object");
    }
    const Json& function = call.contains("function") ? call["function"] : call;
    if (!function.is_object() || !function.contains("name") ||
        !function["name"].is_string()) {
      throw std::invalid_argument("chat: a tool call needs a function name");
    }
    const auto& name = function["name"].get_ref<const std::string&>();
    if (first) {
      out += Trim(content).empty() ? "<tool_call>\n<function="
                                   : "\n\n<tool_call>\n<function=";
    } else {
      out += "\n<tool_call>\n<function=";
    }
    first = false;
    out += name + ">\n";
    const Json args = Arguments(function);
    if (args.is_object()) {
      for (const auto& [key, value] : args.items()) {
        out += "<parameter=" + key + ">\n";
        out += value.is_object() || value.is_array() ? PyJson(value)
                                                     : PyStr(value);
        out += "\n</parameter>\n";
      }
    }
    out += "</function>\n</tool_call>";
  }
}

constexpr char kToolsPreamble[] =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr char kToolsInstructions[] =
    "\n\nIf you choose to call a function ONLY reply in the following format "
    "with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n"
    "</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
    "specified format: an inner <function=...></function> block must be "
    "nested within <tool_call></tool_call> XML tags\n- Required parameters "
    "MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT "
    "after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about "
    "function calls\n</IMPORTANT>";

bool HasTools(const Json& tools) {
  return tools.is_array() && !tools.empty();
}

std::string RenderTemplate(const ChatPrompt& prompt, bool* prefill) {
  const Json& messages = prompt.messages;
  if (!messages.is_array() || messages.empty()) {
    throw std::invalid_argument("chat: no messages");
  }
  const std::size_t count = messages.size();
  for (const auto& message : messages) {
    const auto& role = Role(message);
    if (role != "user" && role != "assistant" && role != "tool" &&
        !IsSystem(role)) {
      throw std::invalid_argument("chat: unsupported role " + role);
    }
  }

  std::string out;
  // System prompt: the first one or two system/developer messages, merged.
  std::size_t num_sys = 0;
  std::string merged_system;
  if (IsSystem(Role(messages[0]))) {
    merged_system = Trim(RenderContent(messages[0]));
    num_sys = 1;
    if (count > 1 && IsSystem(Role(messages[1]))) {
      merged_system += "\n" + Trim(RenderContent(messages[1]));
      num_sys = 2;
    }
  }
  if (HasTools(prompt.tools)) {
    out += std::string(kImStart) + "system\n" + kToolsPreamble;
    for (const auto& tool : prompt.tools) {
      out += "\n" + PyJson(tool);
    }
    out += "\n</tools>";
    out += kToolsInstructions;
    if (!merged_system.empty()) {
      out += "\n\n" + merged_system;
    }
    out += kImEnd;
  } else if (!merged_system.empty()) {
    out += std::string(kImStart) + "system\n" + merged_system +
           std::string(kImEnd);
  }

  // The last user message that is a query, not a wrapped tool response:
  // assistant turns after it keep their reasoning.
  std::size_t last_query = count - 1;
  for (std::size_t k = count; k-- > 0;) {
    if (Role(messages[k]) != "user") {
      continue;
    }
    const auto content = Trim(RenderContent(messages[k]));
    if (!(content.starts_with("<tool_response>") &&
          content.ends_with("</tool_response>"))) {
      last_query = k;
      break;
    }
  }

  for (std::size_t i = num_sys; i < count; ++i) {
    const Json& message = messages[i];
    const auto& role = Role(message);
    if (IsSystem(role)) {
      continue;  // later system messages are dropped
    }
    std::string content = Trim(RenderContent(message));
    if (role == "user") {
      out += std::string(kImStart) + "user\n" + content + std::string(kImEnd);
    } else if (role == "assistant") {
      std::string reasoning;
      if (message.contains("reasoning_content") &&
          message["reasoning_content"].is_string()) {
        reasoning = message["reasoning_content"].get<std::string>();
      } else if (content.find(kThinkClose) != std::string::npos) {
        const auto before = Split(content, kThinkClose);
        reasoning = std::string(LStripNewlines(
            Split(RStripNewlines(before.front()), kThinkOpen).back()));
        content = std::string(LStripNewlines(before.back()));
      }
      reasoning = Trim(reasoning);
      out += std::string(kImStart) + "assistant\n";
      if (i > last_query) {
        out += "<think>\n" + reasoning + "\n</think>\n\n";
      }
      out += content;
      if (message.contains("tool_calls")) {
        RenderToolCalls(message["tool_calls"], content, out);
      }
      out += kImEnd;
    } else {  // tool
      if (i > 0 && Role(messages[i - 1]) != "tool") {
        out += std::string(kImStart) + "user";
      }
      out += "\n<tool_response>\n" + content + "\n</tool_response>";
      if (i + 1 == count || Role(messages[i + 1]) != "tool") {
        out += kImEnd;
      }
    }
  }

  *prefill = Role(messages[count - 1]) == "assistant";
  if (*prefill) {
    out.resize(out.size() - kImEnd.size());  // continue the final message
  } else {
    out += std::string(kImStart) + "assistant\n";
    out += prompt.thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  return out;
}

// ---- Output parsing ----

// The JSON-schema types a tool's parameter accepts, from `type`, `anyOf`
// or `oneOf`.
void SchemaTypes(const Json& schema, std::vector<std::string>& types) {
  if (!schema.is_object()) {
    return;
  }
  if (schema.contains("type")) {
    const Json& type = schema["type"];
    if (type.is_string()) {
      types.push_back(type.get<std::string>());
    } else if (type.is_array()) {
      for (const auto& t : type) {
        if (t.is_string()) {
          types.push_back(t.get<std::string>());
        }
      }
    }
  }
  for (const char* key : {"anyOf", "oneOf"}) {
    if (schema.contains(key) && schema[key].is_array()) {
      for (const auto& alt : schema[key]) {
        SchemaTypes(alt, types);
      }
    }
  }
}

// A parameter's text as the type its schema names; text otherwise.
Json TypedValue(const std::string& text,
                const std::vector<std::string>& types) {
  for (const auto& type : types) {
    if (type == "string") {
      continue;  // tried last: any text is a string
    }
    if (type == "boolean") {
      if (text == "true" || text == "True") {
        return true;
      }
      if (text == "false" || text == "False") {
        return false;
      }
      continue;
    }
    if (type == "null") {
      if (text == "null" || text == "None") {
        return nullptr;
      }
      continue;
    }
    auto parsed = Json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (parsed.is_discarded()) {
      continue;
    }
    if ((type == "integer" && parsed.is_number_integer()) ||
        (type == "number" && parsed.is_number()) ||
        (type == "object" && parsed.is_object()) ||
        (type == "array" && parsed.is_array())) {
      return parsed;
    }
  }
  return text;
}

class Qwen35Parser : public OutputParser {
public:
  Qwen35Parser(Json tools, bool reasoning)
      : tools_(std::move(tools)),
        use_tools_(HasTools(tools_)),
        mode_(reasoning ? Mode::kReasoning : Mode::kContent),
        rng_(std::random_device{}()) {}

  std::vector<OutputPart> Feed(const std::string& text) override {
    pending_ += text;
    return Drain(false);
  }

  std::vector<OutputPart> Finish() override { return Drain(true); }

private:
  enum class Mode { kReasoning, kContent, kToolCall };

  // Length of the longest suffix of `s` that is a proper prefix of `marker`.
  static std::size_t PartialSuffix(std::string_view s,
                                   std::string_view marker) {
    for (std::size_t k = std::min(s.size(), marker.size() - 1); k > 0; --k) {
      if (s.substr(s.size() - k) == marker.substr(0, k)) {
        return k;
      }
    }
    return 0;
  }

  // How much of pending_'s end to hold back: a possible start of `marker`
  // (when given) and the newlines before it, which are dropped if markup
  // or the end follows.
  std::size_t HoldBack(std::string_view marker) const {
    std::string_view s = pending_;
    std::size_t hold = marker.empty() ? 0 : PartialSuffix(s, marker);
    s.remove_suffix(hold);
    while (!s.empty() && s.back() == '\n') {
      s.remove_suffix(1);
      ++hold;
    }
    return hold;
  }

  void Emit(OutputPart::Kind kind, std::string_view text,
            std::vector<OutputPart>& parts) {
    bool& at_start = kind == OutputPart::Kind::kReasoning ? reasoning_start_
                                                          : content_start_;
    if (at_start) {
      text = LStripNewlines(text);
      if (text.empty()) {
        return;
      }
      at_start = false;
    }
    if (text.empty()) {
      return;
    }
    if (!parts.empty() && parts.back().kind == kind) {
      parts.back().text += text;
    } else {
      OutputPart part;
      part.kind = kind;
      part.text = std::string(text);
      parts.push_back(std::move(part));
    }
  }

  std::string NewCallId() {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    std::uniform_int_distribution<int> pick(0, sizeof kAlphabet - 2);
    std::string id = "call_";
    for (int i = 0; i < 24; ++i) {
      id += kAlphabet[pick(rng_)];
    }
    return id;
  }

  const Json* ParameterSchema(const std::string& function,
                              const std::string& parameter) const {
    if (!use_tools_) {
      return nullptr;
    }
    for (const auto& tool : tools_) {
      const Json& f = tool.is_object() && tool.contains("function")
                          ? tool["function"]
                          : tool;
      if (!f.is_object() || !f.contains("name") || f["name"] != function) {
        continue;
      }
      if (f.contains("parameters") && f["parameters"].is_object() &&
          f["parameters"].contains("properties") &&
          f["parameters"]["properties"].is_object() &&
          f["parameters"]["properties"].contains(parameter)) {
        return &f["parameters"]["properties"][parameter];
      }
      return nullptr;
    }
    return nullptr;
  }

  // `<function=NAME>` then `<parameter=KEY>\nVALUE\n</parameter>` blocks,
  // then `</function>`, with only whitespace around; false otherwise.
  bool ParseToolCall(std::string_view body, OutputPart& part) {
    const std::string trimmed = Trim(body);
    std::string_view s = trimmed;
    constexpr std::string_view kFunction = "<function=";
    if (!s.starts_with(kFunction)) {
      return false;
    }
    s.remove_prefix(kFunction.size());
    const auto name_end = s.find('>');
    if (name_end == std::string_view::npos || name_end == 0) {
      return false;
    }
    const std::string name(s.substr(0, name_end));
    if (name.find_first_of("\n<") != std::string::npos) {
      return false;
    }
    s.remove_prefix(name_end + 1);
    Json args = Json::object();
    constexpr std::string_view kParam = "<parameter=";
    constexpr std::string_view kParamClose = "</parameter>";
    constexpr std::string_view kFunctionClose = "</function>";
    while (true) {
      while (!s.empty() && IsAsciiSpace(s.front())) {
        s.remove_prefix(1);
      }
      if (s.starts_with(kFunctionClose)) {
        s.remove_prefix(kFunctionClose.size());
        break;
      }
      if (!s.starts_with(kParam)) {
        return false;
      }
      s.remove_prefix(kParam.size());
      const auto key_end = s.find('>');
      if (key_end == std::string_view::npos || key_end == 0) {
        return false;
      }
      const std::string key(s.substr(0, key_end));
      if (key.find_first_of("\n<") != std::string::npos) {
        return false;
      }
      s.remove_prefix(key_end + 1);
      const auto value_end = s.find(kParamClose);
      if (value_end == std::string_view::npos) {
        return false;
      }
      std::string_view value = s.substr(0, value_end);
      if (value.starts_with('\n')) {
        value.remove_prefix(1);
      }
      if (value.ends_with('\n')) {
        value.remove_suffix(1);
      }
      std::vector<std::string> types;
      if (const Json* schema = ParameterSchema(name, key)) {
        SchemaTypes(*schema, types);
      }
      args[key] = TypedValue(std::string(value), types);
      s.remove_prefix(value_end + kParamClose.size());
    }
    if (!Trim(s).empty()) {
      return false;
    }
    part.kind = OutputPart::Kind::kToolCall;
    part.tool_id = NewCallId();
    part.tool_name = name;
    part.arguments = args.dump(-1, ' ', false, Json::error_handler_t::replace);
    return true;
  }

  std::vector<OutputPart> Drain(bool final) {
    std::vector<OutputPart> parts;
    while (true) {
      if (mode_ == Mode::kReasoning) {
        if (reasoning_start_ && !pending_.empty()) {
          // A model may repeat the opening tag the prompt already holds.
          const auto lead = LStripNewlines(pending_);
          if (lead.starts_with(kThinkOpen)) {
            pending_.erase(0,
                           pending_.size() - lead.size() + kThinkOpen.size());
          } else if (!final && kThinkOpen.starts_with(lead)) {
            return parts;
          }
        }
        const auto close = pending_.find(kThinkClose);
        if (close != std::string::npos) {
          Emit(OutputPart::Kind::kReasoning,
               RStripNewlines(std::string_view(pending_).substr(0, close)),
               parts);
          pending_.erase(0, close + kThinkClose.size());
          mode_ = Mode::kContent;
          content_start_ = true;
          continue;
        }
        const std::size_t hold = final ? 0 : HoldBack(kThinkClose);
        std::string_view out(pending_);
        out.remove_suffix(hold);
        Emit(OutputPart::Kind::kReasoning, final ? RStripNewlines(out) : out,
             parts);
        pending_.erase(0, pending_.size() - hold);
        return parts;
      }
      if (mode_ == Mode::kContent) {
        const auto open =
            use_tools_ ? pending_.find(kToolCallOpen) : std::string::npos;
        if (open != std::string::npos) {
          const auto before = std::string_view(pending_).substr(0, open);
          const auto kept = RStripNewlines(before);
          call_lead_ = before.size() - kept.size();
          Emit(OutputPart::Kind::kContent, kept, parts);
          pending_.erase(0, open + kToolCallOpen.size());
          mode_ = Mode::kToolCall;
          continue;
        }
        const std::size_t hold =
            final ? 0
                  : HoldBack(use_tools_ ? kToolCallOpen : std::string_view());
        std::string_view out(pending_);
        out.remove_suffix(hold);
        Emit(OutputPart::Kind::kContent, final ? RStripNewlines(out) : out,
             parts);
        pending_.erase(0, pending_.size() - hold);
        return parts;
      }
      // Inside a tool call: wait for its end.
      const auto close = pending_.find(kToolCallClose);
      if (close == std::string::npos && !final) {
        return parts;
      }
      const std::string_view body = std::string_view(pending_).substr(
          0, close == std::string::npos ? pending_.size() : close);
      OutputPart call;
      if (close != std::string::npos && ParseToolCall(body, call)) {
        parts.push_back(std::move(call));
      } else {
        // Not a well-formed call: pass the text through as content.
        std::string raw = std::string(call_lead_, '\n') +
                          std::string(kToolCallOpen) + std::string(body);
        if (close != std::string::npos) {
          raw += kToolCallClose;
        }
        content_start_ = false;
        Emit(OutputPart::Kind::kContent, raw, parts);
      }
      pending_.erase(0, close == std::string::npos
                            ? pending_.size()
                            : close + kToolCallClose.size());
      mode_ = Mode::kContent;
      content_start_ = true;
      if (final && pending_.empty()) {
        return parts;
      }
    }
  }

  Json tools_;
  bool use_tools_;
  Mode mode_;
  std::string pending_;
  bool reasoning_start_ = true;
  bool content_start_ = true;
  std::size_t call_lead_ = 0;  // newlines dropped before the open call
  std::mt19937_64 rng_;
};

class Qwen35ChatFormat : public ChatFormat {
public:
  std::string Render(const ChatPrompt& prompt) const override {
    bool prefill = false;
    return RenderTemplate(prompt, &prefill);
  }

  std::unique_ptr<OutputParser> NewParser(
      const ChatPrompt& prompt) const override {
    // The generation prompt opens a think block when thinking is on; a
    // continued assistant message is past its think block.
    const bool prefill =
        prompt.messages.is_array() && !prompt.messages.empty() &&
        prompt.messages.back().is_object() &&
        prompt.messages.back().value("role", "") == "assistant";
    return std::make_unique<Qwen35Parser>(
        HasTools(prompt.tools) ? prompt.tools : Json::array(),
        prompt.thinking && !prefill);
  }

  std::vector<std::string> StopTokens() const override {
    return {"<|im_end|>", "<|endoftext|>"};
  }
};

}  // namespace

std::unique_ptr<ChatFormat> NewQwen35ChatFormat() {
  return std::make_unique<Qwen35ChatFormat>();
}

}  // namespace gufo_npu::serve
