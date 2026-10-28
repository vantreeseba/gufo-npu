// Structured outputs: the grammar constraint over a toy vocabulary (every
// mask checked against token-by-token simulation, random constrained
// generations validated against their schema), forced tool calls read back
// with the real Qwen3.5 chat format, thinking, MTP draft rejection through
// the Sampler, errors, the server end to end over a constrained fake engine,
// and the mask's cost on a 248k-token synthetic vocabulary.
#include "serve/structured.hpp"

#include <httplib/httplib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/sampling.hpp"
#include "serve/chat_format.hpp"
#include "serve/engine.hpp"
#include "serve/engine_pool.hpp"
#include "serve/server.hpp"

namespace {

using gufo_npu::Sampler;
using gufo_npu::SamplingParams;
using gufo_npu::TokenConstraint;
using gufo_npu::serve::ChatPrompt;
using gufo_npu::serve::GrammarError;
using gufo_npu::serve::JsonObjectGrammar;
using gufo_npu::serve::JsonSchemaGrammar;
using gufo_npu::serve::NewGrammarConstraint;
using gufo_npu::serve::OutputPart;
using gufo_npu::serve::ToolCallGrammar;
using gufo_npu::serve::VocabIndex;
using nlohmann::json;
using nlohmann::ordered_json;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) {
    ++failures;
  }
}

// A toy vocabulary: every printable ASCII byte, whitespace, UTF-8 pieces,
// multi-byte tokens that straddle JSON structure, tool-call markup and the
// Qwen3.5 added tokens.
struct Toy {
  std::vector<std::string> bytes;
  std::vector<std::uint8_t> special;
  int end = -1;
  int think_close = -1;

  int Add(const std::string& b, bool is_special = false) {
    bytes.push_back(b);
    special.push_back(is_special ? 1 : 0);
    return static_cast<int>(bytes.size()) - 1;
  }
  int Find(const std::string& b) const {
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      if (bytes[i] == b) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  Toy() {
    for (int c = 0x20; c < 0x7f; ++c) {
      Add(std::string(1, static_cast<char>(c)));
    }
    for (const char* t : {"\n",
                          "\t",
                          "  ",
                          "\n  ",
                          "\n    ",
                          "{\"",
                          "\":",
                          "\": ",
                          "\": \"",
                          "\",",
                          "\", \"",
                          "\"}",
                          "\"]",
                          "},",
                          "}]",
                          "true",
                          "false",
                          "null",
                          "name",
                          "age",
                          "tags",
                          "city",
                          "unit",
                          "celsius",
                          "fahrenheit",
                          "Paris",
                          "123",
                          "0.5",
                          "-1",
                          "e+",
                          "1e3",
                          "\\\"",
                          "\\n",
                          "\\u00e9",
                          "\xc3\xa9",
                          "\xc3",
                          "\xa9",
                          "<function=",
                          "<parameter=",
                          "</parameter>",
                          "</function>",
                          "\n</parameter>\n",
                          "\n</parameter>",
                          "get_weather",
                          "get_time",
                          "zone",
                          "days",
                          ">\n",
                          "\n<",
                          "hello",
                          " world",
                          "ok\"",
                          "\"\n"}) {
      Add(t);
    }
    Add("");  // padding: never allowed
    end = Add("<|im_end|>", true);
    Add("<|im_start|>", true);
    Add("<tool_call>", true);
    Add("</tool_call>", true);
    Add("<think>", true);
    think_close = Add("</think>", true);
  }

  std::shared_ptr<const VocabIndex> Index() const {
    return std::make_shared<const VocabIndex>(bytes, special);
  }
};

const Toy& toy() {
  static const Toy t;
  return t;
}

const std::shared_ptr<const VocabIndex>& ToyIndex() {
  static const auto index = toy().Index();
  return index;
}

// Allowed() against IsAllowed() for every token.
bool MaskMatches(TokenConstraint& c, int vocab, std::string* why) {
  std::vector<std::uint8_t> mask;
  c.Allowed(vocab, mask);
  for (int id = 0; id < vocab; ++id) {
    const bool one = c.IsAllowed(id);
    if ((mask[static_cast<std::size_t>(id)] != 0) != one) {
      *why = "token " + std::to_string(id) + " \"" +
             toy().bytes[static_cast<std::size_t>(id)] + "\": mask " +
             std::to_string(mask[static_cast<std::size_t>(id)]) + ", single " +
             std::to_string(one);
      return false;
    }
  }
  return true;
}

// A random constrained generation: each step checks the mask against
// single-token simulation and draws an allowed token, favouring tokens that
// close things so walks end. Returns the text, or nullopt on a failure.
std::optional<std::string> Walk(TokenConstraint& c, std::mt19937& rng,
                                const std::string& label,
                                const std::vector<int>& prefix = {}) {
  const Toy& t = toy();
  const int vocab = static_cast<int>(t.bytes.size());
  c.Reset();
  std::string text;
  for (const int id : prefix) {
    c.Advance(id);
    text += t.bytes[static_cast<std::size_t>(id)];
  }
  std::vector<std::uint8_t> mask;
  for (int step = 0; step < 3000; ++step) {
    std::string why;
    if (!MaskMatches(c, vocab, &why)) {
      Check(false, label + ": mask at step " + std::to_string(step) + ": " +
                       why + " after \"" + text + "\"");
      return std::nullopt;
    }
    c.Allowed(vocab, mask);
    std::vector<double> weights(mask.size(), 0.0);
    double total = 0;
    for (std::size_t i = 0; i < mask.size(); ++i) {
      if (mask[i] == 0) {
        continue;
      }
      const std::string& b = t.bytes[i];
      double w = 1.0;
      if (static_cast<int>(i) == t.end) {
        w = 200.0;
      } else if (!b.empty() && (b.back() == '"' || b.back() == '}' ||
                                b.back() == ']' || b.back() == '>')) {
        w = 25.0;
      } else if (step > 200) {
        w = b.size() > 1 ? 0.5 : 0.2;
      }
      weights[i] = w;
      total += w;
    }
    if (total == 0) {
      Check(false, label + ": no token allowed after \"" + text + "\"");
      return std::nullopt;
    }
    std::discrete_distribution<int> pick(weights.begin(), weights.end());
    const int id = pick(rng);
    if (id == t.end) {
      return text;
    }
    c.Advance(id);
    text += t.bytes[static_cast<std::size_t>(id)];
  }
  Check(false, label + ": did not end: \"" + text.substr(0, 200) + "\"");
  return std::nullopt;
}

// --- A JSON schema validator for the keywords the grammar enforces. -------

bool Validate(const json& v, const json& s, const json& root);

const json& Resolve(const json& s, const json& root) {
  if (s.is_object() && s.contains("$ref")) {
    const std::string ref = s["$ref"];
    if (ref == "#") {
      return root;
    }
    const auto slash = ref.rfind('/');
    const std::string defs =
        ref.starts_with("#/$defs/") ? "$defs" : "definitions";
    return Resolve(root[defs][ref.substr(slash + 1)], root);
  }
  return s;
}

bool OfType(const json& v, const std::string& type, const json& s,
            const json& root) {
  if (type == "string") {
    if (!v.is_string()) {
      return false;
    }
    // Characters, as the grammar counts them (UTF-8 lead bytes).
    int chars = 0;
    for (const char ch : v.get<std::string>()) {
      chars += (static_cast<unsigned char>(ch) & 0xC0) != 0x80;
    }
    return chars >= s.value("minLength", 0) &&
           chars <= s.value("maxLength", 1 << 30);
  }
  if (type == "integer") {
    return v.is_number_integer();
  }
  if (type == "number") {
    return v.is_number();
  }
  if (type == "boolean") {
    return v.is_boolean();
  }
  if (type == "null") {
    return v.is_null();
  }
  if (type == "array") {
    if (!v.is_array()) {
      return false;
    }
    const int n = static_cast<int>(v.size());
    if (n < s.value("minItems", 0) || n > s.value("maxItems", 1 << 30)) {
      return false;
    }
    for (const auto& item : v) {
      if (s.contains("items") && !Validate(item, s["items"], root)) {
        return false;
      }
    }
    return true;
  }
  if (type == "object") {
    if (!v.is_object()) {
      return false;
    }
    const json props = s.value("properties", json::object());
    for (const auto& r : s.value("required", json::array())) {
      if (!v.contains(r.get<std::string>())) {
        return false;
      }
    }
    for (const auto& [key, value] : v.items()) {
      if (props.contains(key)) {
        if (!Validate(value, props[key], root)) {
          return false;
        }
      } else if (s.contains("additionalProperties")) {
        const json& extra = s["additionalProperties"];
        if (extra.is_boolean() ? !extra.get<bool>()
                               : !Validate(value, extra, root)) {
          return false;
        }
      }
    }
    return true;
  }
  return false;
}

bool Validate(const json& v, const json& schema, const json& root) {
  const json& s = Resolve(schema, root);
  if (s.is_boolean()) {
    return s.get<bool>();
  }
  if (s.contains("const")) {
    return v == s["const"];
  }
  if (s.contains("enum")) {
    return std::find(s["enum"].begin(), s["enum"].end(), v) != s["enum"].end();
  }
  for (const char* key : {"anyOf", "oneOf"}) {
    if (s.contains(key)) {
      for (const auto& alt : s[key]) {
        if (Validate(v, alt, root)) {
          return true;
        }
      }
      return false;
    }
  }
  if (s.contains("allOf")) {
    return Validate(v, s["allOf"][0], root);
  }
  if (!s.contains("type")) {
    if (s.contains("properties") || s.contains("required")) {
      return OfType(v, "object", s, root);
    }
    return true;
  }
  if (s["type"].is_array()) {
    for (const auto& t : s["type"]) {
      if (OfType(v, t.get<std::string>(), s, root)) {
        return true;
      }
    }
    return false;
  }
  return OfType(v, s["type"].get<std::string>(), s, root);
}

// --- Tests. ----------------------------------------------------------------

void SchemaWalks(const std::string& label, const json& schema, bool strict,
                 int walks = 40) {
  const ordered_json ordered = ordered_json::parse(schema.dump());
  auto c = NewGrammarConstraint(JsonSchemaGrammar(ordered, strict), ToyIndex(),
                                {toy().end}, false);
  std::mt19937 rng(7);
  int valid = 0;
  for (int i = 0; i < walks; ++i) {
    const auto text = Walk(*c, rng, label);
    if (!text) {
      return;
    }
    const json v = json::parse(*text, nullptr, false);
    if (v.is_discarded() || !Validate(v, schema, schema)) {
      Check(false, label + ": invalid output: " + *text);
      return;
    }
    ++valid;
  }
  Check(valid == walks, label + ": " + std::to_string(walks) +
                            " random generations parse and validate");
}

void Schemas() {
  SchemaWalks("strict object",
              {{"type", "object"},
               {"properties",
                {{"name", {{"type", "string"}}},
                 {"age", {{"type", "integer"}}},
                 {"tags",
                  {{"type", "array"},
                   {"items", {{"type", "string"}}},
                   {"maxItems", 3}}}}},
               {"required", {"name", "age"}},
               {"additionalProperties", false}},
              true);
  SchemaWalks(
      "non-strict object takes other keys",
      {{"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}},
      false, 20);
  SchemaWalks(
      "additionalProperties schema",
      {{"type", "object"}, {"additionalProperties", {{"type", "boolean"}}}},
      false, 20);
  SchemaWalks(
      "nested objects and arrays",
      {{"type", "object"},
       {"properties",
        {{"unit",
          {{"type", "object"},
           {"properties", {{"days", {{"type", "number"}}}}},
           {"required", {"days"}},
           {"additionalProperties", false}}},
         {"tags",
          {{"type", "array"},
           {"minItems", 1},
           {"maxItems", 2},
           {"items", {{"type", "array"}, {"items", {{"type", "null"}}}}}}}}},
       {"required", {"unit", "tags"}},
       {"additionalProperties", false}},
      true);
  SchemaWalks(
      "enum and const",
      {{"type", "object"},
       {"properties",
        {{"unit", {{"enum", {"celsius", "fahrenheit", 1, 10, nullptr}}}},
         {"city", {{"const", "Paris"}}}}},
       {"required", {"unit", "city"}},
       {"additionalProperties", false}},
      true);
  SchemaWalks("anyOf with the same first byte",
              {{"anyOf",
                {{{"type", "object"},
                  {"properties", {{"name", {{"type", "string"}}}}},
                  {"required", {"name"}},
                  {"additionalProperties", false}},
                 {{"type", "object"},
                  {"properties", {{"age", {{"type", "integer"}}}}},
                  {"required", {"age"}},
                  {"additionalProperties", false}}}}},
              true);
  SchemaWalks(
      "oneOf and type arrays",
      {{"type", "object"},
       {"properties",
        {{"name", {{"type", {"string", "null"}}}},
         {"age", {{"oneOf", {{{"type", "integer"}}, {{"type", "boolean"}}}}}}}},
       {"required", {"name", "age"}},
       {"additionalProperties", false}},
      true);
  SchemaWalks("$ref and $defs, recursive",
              {{"$defs",
                {{"node",
                  {{"type", "object"},
                   {"properties",
                    {{"name", {{"type", "string"}, {"maxLength", 5}}},
                     {"tags",
                      {{"type", "array"},
                       {"maxItems", 2},
                       {"items", {{"$ref", "#/$defs/node"}}}}}}},
                   {"required", {"name"}},
                   {"additionalProperties", false}}}}},
               {"$ref", "#/$defs/node"}},
              true);
  SchemaWalks(
      "string lengths and numbers",
      {{"type", "object"},
       {"properties",
        {{"name", {{"type", "string"}, {"minLength", 2}, {"maxLength", 4}}},
         {"age", {{"type", "number"}}},
         {"days", {{"type", "integer"}}}}},
       {"required", {"name", "age", "days"}},
       {"additionalProperties", false}},
      true);
  SchemaWalks("any value", json::object(), false, 30);
  SchemaWalks("top-level array",
              {{"type", "array"}, {"items", {{"type", "integer"}}}}, true);
  SchemaWalks("empty strict object",
              {{"type", "object"}, {"additionalProperties", false}}, true, 5);

  // json_object: any object.
  {
    auto c = NewGrammarConstraint(JsonObjectGrammar(), ToyIndex(), {toy().end},
                                  false);
    std::mt19937 rng(3);
    bool ok = true;
    for (int i = 0; i < 40 && ok; ++i) {
      const auto text = Walk(*c, rng, "json_object");
      const json v = text ? json::parse(*text, nullptr, false) : json();
      ok = text && v.is_object();
      if (!ok) {
        Check(false, "json_object: " + text.value_or("(failed)"));
      }
    }
    Check(ok, "json_object: 40 random generations are objects");
  }
}

void Specifics() {
  const Toy& t = toy();
  const int vocab = static_cast<int>(t.bytes.size());
  auto allowed = [&](TokenConstraint& c, const std::string& token) {
    return c.IsAllowed(t.Find(token));
  };
  auto feed = [&](TokenConstraint& c, const std::vector<std::string>& toks) {
    c.Reset();
    for (const auto& tok : toks) {
      c.Advance(t.Find(tok));
    }
  };
  const ordered_json schema = {
      {"type", "object"},
      {"properties",
       {{"name", {{"type", "string"}}}, {"age", {{"type", "integer"}}}}},
      {"required", {"name"}},
      {"additionalProperties", false}};
  auto c = NewGrammarConstraint(JsonSchemaGrammar(schema, true), ToyIndex(),
                                {t.end}, false);
  c->Reset();
  Check(allowed(*c, "{") && allowed(*c, " ") && !allowed(*c, "[") &&
            !allowed(*c, "true") && !c->IsAllowed(t.end),
        "start: an object or whitespace, not the end");
  feed(*c, {"{"});
  Check(!allowed(*c, "}") && allowed(*c, "\""),
        "required name: no empty object");
  feed(*c, {"{", "\""});
  Check(allowed(*c, "name") && allowed(*c, "a") && !allowed(*c, "city") &&
            !allowed(*c, "x"),
        "keys: only declared names");
  feed(*c, {"{\"", "age", "\":", "1"});
  Check(allowed(*c, "123") && !allowed(*c, ".") && !allowed(*c, "0.5") &&
            allowed(*c, ","),
        "integer: digits, no fraction");
  feed(*c, {"{\"", "age", "\":", "1", ",", "\""});
  Check(!allowed(*c, "a") && !allowed(*c, "age") && allowed(*c, "name"),
        "keys: no repeats");
  feed(*c, {"{\"", "name", "\": \"", "hello"});
  Check(allowed(*c, " world") && allowed(*c, "\"}") &&
            !c->IsAllowed(t.Find("<|im_start|>")) &&
            !c->IsAllowed(t.Find("<tool_call>")),
        "strings: text yes, added tokens no");
  feed(*c, {"{\"", "name", "\": \"", "hello", "\"}"});
  Check(c->IsAllowed(t.end) && allowed(*c, "\n") && !allowed(*c, "{"),
        "complete: the end or whitespace");
  std::vector<std::uint8_t> mask;
  c->Allowed(vocab, mask);
  Check(mask[static_cast<std::size_t>(t.Find(""))] == 0,
        "padding tokens are never allowed");

  // Thinking: free until </think>, then the grammar.
  auto think =
      NewGrammarConstraint(JsonObjectGrammar(), ToyIndex(), {t.end}, true);
  think->Reset();
  Check(allowed(*think, "x") && allowed(*think, "hello") &&
            !think->IsAllowed(t.end),
        "thinking: free text, no end");
  think->Advance(t.Find("hello"));
  think->Advance(t.think_close);
  Check(!allowed(*think, "x") && allowed(*think, "{") && allowed(*think, "\n"),
        "after </think>: the grammar");
  think->Reset();
  for (const char* piece : {"ok", "<", "/", "t", "h", "i", "n", "k", ">"}) {
    think->Advance(t.Find(piece));
  }
  Check(!allowed(*think, "x") && allowed(*think, "{"),
        "</think> spelled in bytes ends thinking too");
}

void ToolCalls() {
  const Toy& t = toy();
  const ordered_json tools = ordered_json::parse(R"([
    {"type": "function", "function": {"name": "get_weather",
      "parameters": {"type": "object", "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer"}},
       "required": ["city", "days"]}}},
    {"type": "function", "function": {"name": "get_time",
      "parameters": {"type": "object", "properties": {
        "zone": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "hello"}}])");
  const auto format = gufo_npu::serve::NewQwen35ChatFormat();
  ChatPrompt prompt;
  prompt.tools = tools;
  for (const std::string name : {"", "get_weather", "hello"}) {
    auto c = NewGrammarConstraint(ToolCallGrammar(tools, name), ToyIndex(),
                                  {t.end}, false);
    std::mt19937 rng(11);
    int good = 0;
    const int walks = 40;
    for (int i = 0; i < walks; ++i) {
      const std::string label = "tool call (" + name + ")";
      const auto text = Walk(*c, rng, label);
      if (!text) {
        break;
      }
      auto parser = format->NewParser(prompt);
      auto parts = parser->Feed(*text);
      for (auto& p : parser->Finish()) {
        parts.push_back(std::move(p));
      }
      bool ok = !parts.empty();
      for (const auto& p : parts) {
        if (p.kind != OutputPart::Kind::kToolCall ||
            (!name.empty() && p.tool_name != name)) {
          ok = false;
          continue;
        }
        const json args = json::parse(p.arguments, nullptr, false);
        if (p.tool_name == "get_weather") {
          ok = ok && args.contains("city") && args["city"].is_string() &&
               args.contains("days") && args["days"].is_number_integer() &&
               (!args.contains("unit") || args["unit"] == "celsius" ||
                args["unit"] == "fahrenheit");
          for (const auto& [key, value] : args.items()) {
            ok = ok && (key == "city" || key == "unit" || key == "days");
          }
        }
      }
      if (!ok) {
        Check(false, label + ": " + *text);
        break;
      }
      ++good;
    }
    Check(good == walks, "tool call (" + (name.empty() ? "any" : name) +
                             "): " + std::to_string(walks) +
                             " random calls parse with the chat format");
  }
  auto c = NewGrammarConstraint(ToolCallGrammar(tools, "get_weather"),
                                ToyIndex(), {t.end}, false);
  c->Reset();
  Check(c->IsAllowed(t.Find("<tool_call>")) && c->IsAllowed(t.Find("<")) &&
            !c->IsAllowed(t.Find("hello")) && !c->IsAllowed(t.end),
        "forced call: starts with the markup");
}

void Errors() {
  auto refused = [](const std::function<void()>& f) {
    try {
      f();
    } catch (const GrammarError&) {
      return true;
    }
    return false;
  };
  Check(refused([] {
          JsonSchemaGrammar(ordered_json{{"$ref", "#/$defs/missing"}}, true);
        }),
        "error: unresolved $ref");
  Check(refused([] {
          JsonSchemaGrammar(ordered_json{{"enum", ordered_json::array()}},
                            true);
        }),
        "error: empty enum");
  Check(
      refused([] { JsonSchemaGrammar(ordered_json{{"type", "date"}}, true); }),
      "error: unknown type");
  Check(refused([] {
          JsonSchemaGrammar(ordered_json{{"type", "object"},
                                         {"properties", ordered_json::object()},
                                         {"required", {"x"}}},
                            true);
        }),
        "error: required names an undefined property");
  Check(refused([] {
          ToolCallGrammar(
              ordered_json::parse(
                  R"([{"type":"function","function":{"name":"a"}}])"),
              "b");
        }),
        "error: forced function not in tools");
  // A schema that refers to itself only cannot loop.
  auto self =
      NewGrammarConstraint(JsonSchemaGrammar(ordered_json{{"$ref", "#"}}, true),
                           ToyIndex(), {toy().end}, false);
  std::vector<std::uint8_t> mask;
  self->Allowed(static_cast<int>(toy().bytes.size()), mask);
  Check(true, "self-referencing schema: a mask without looping");
}

// The Sampler applies the constraint: greedy takes the best allowed token,
// and an MTP-style loop never keeps a disallowed draft.
void SamplerAndDrafts() {
  const Toy& t = toy();
  const int vocab = static_cast<int>(t.bytes.size());
  const ordered_json schema = {{"type", "object"},
                               {"properties", {{"age", {{"type", "integer"}}}}},
                               {"required", {"age"}},
                               {"additionalProperties", false}};
  SamplingParams p;
  p.constraint = NewGrammarConstraint(JsonSchemaGrammar(schema, true),
                                      ToyIndex(), {t.end}, false);
  Sampler greedy(p);
  std::vector<float> logits(static_cast<std::size_t>(vocab), 0.0f);
  logits[static_cast<std::size_t>(t.Find("x"))] = 10.0f;  // never allowed
  logits[static_cast<std::size_t>(t.Find("{\""))] = 5.0f;
  Check(greedy.Sample(logits.data(), vocab) == t.Find("{\""),
        "greedy: the best allowed token");
  Sampler none(SamplingParams{});
  Check(none.Sample(logits.data(), vocab) == t.Find("x"),
        "no constraint: unchanged");

  // A speculative loop as the decoders run it: the target samples z under
  // the mask; a draft is kept only when z equals it. The drafter always
  // proposes a disallowed "x"; the target prefers it too.
  for (const float temperature : {0.0f, 1.0f}) {
    SamplingParams sp;
    sp.temperature = temperature;
    sp.seed = 5;
    sp.constraint = NewGrammarConstraint(JsonSchemaGrammar(schema, true),
                                         ToyIndex(), {t.end}, false);
    Sampler sampler(sp);
    std::mt19937 rng(9);
    std::string text;
    int kept = 0;
    const bool ok = true;
    for (int step = 0; step < 400; ++step) {
      std::vector<float> l(static_cast<std::size_t>(vocab));
      std::uniform_real_distribution<float> u(-1.0f, 1.0f);
      for (auto& v : l) {
        v = u(rng);
      }
      const int draft = t.Find("x");
      l[static_cast<std::size_t>(draft)] = 8.0f;
      l[static_cast<std::size_t>(t.end)] = 3.0f;
      for (const char* s : {"{\"", "age", "\":", "1", "}"}) {
        l[static_cast<std::size_t>(t.Find(s))] += 2.0f;
      }
      const int z = sampler.Sample(l.data(), vocab);
      if (z == draft) {
        ++kept;
      }
      if (z == t.end) {
        break;
      }
      sampler.Accept(z);
      text += t.bytes[static_cast<std::size_t>(z)];
    }
    const json v = json::parse(text, nullptr, false);
    Check(ok && kept == 0 && v.is_object() && v.contains("age") &&
              v["age"].is_number_integer(),
          "drafts at temperature " + std::to_string(temperature) +
              ": a disallowed draft is never kept, output valid: " + text);
  }
}

// --- The server over a fake engine that samples under the constraint. ----

class SamplingEngine : public gufo_npu::serve::Engine {
public:
  explicit SamplingEngine(std::string id = "toy") {
    info_.id = std::move(id);
    info_.architecture = "toy";
    info_.context_length = 1 << 16;
    info_.vocab = static_cast<int>(toy().bytes.size());
  }
  const gufo_npu::serve::ModelInfo& info() const override { return info_; }
  const gufo_npu::Tokenizer& tokenizer() const override {
    throw std::logic_error("no tokenizer");
  }
  std::vector<int> Encode(std::string_view text) const override {
    return std::vector<int>(text.size(), 0);
  }
  std::string TokenBytes(int id) const override {
    return toy().bytes.at(static_cast<std::size_t>(id));
  }
  int FindToken(std::string_view token) const override {
    return toy().Find(std::string(token));
  }
  bool IsSpecialToken(int id) const override {
    return toy().special.at(static_cast<std::size_t>(id)) != 0;
  }
  gufo_npu::serve::GenerateResult Generate(
      const gufo_npu::serve::GenerateRequest& request,
      const std::function<bool(int)>& on_token) override {
    std::lock_guard lock(mutex_);
    gufo_npu::serve::GenerateResult result;
    result.prompt_tokens = static_cast<int>(request.prompt.size());
    SamplingParams sp = request.sampling;
    sp.temperature = 1.0f;
    sp.seed = 21;
    Sampler sampler(sp);
    std::mt19937 rng(4);
    const int vocab = info_.vocab;
    std::vector<float> logits(static_cast<std::size_t>(vocab));
    for (int i = 0; i < request.max_tokens; ++i) {
      std::uniform_real_distribution<float> u(-1.0f, 1.0f);
      for (auto& v : logits) {
        v = u(rng);
      }
      for (const int id : request.stop_ids) {
        logits[static_cast<std::size_t>(id)] = 4.0f;
      }
      for (const char* s : {"\"", "}", "]", ">", "\n</parameter>\n"}) {
        logits[static_cast<std::size_t>(toy().Find(s))] += 3.0f;
      }
      const int id = sampler.Sample(logits.data(), vocab);
      sampler.Accept(id);
      result.tokens.push_back(id);
      if (!on_token(id)) {
        result.finish = gufo_npu::serve::FinishReason::kCancelled;
        return result;
      }
      if (std::find(request.stop_ids.begin(), request.stop_ids.end(), id) !=
          request.stop_ids.end()) {
        result.finish = gufo_npu::serve::FinishReason::kStop;
        return result;
      }
    }
    result.finish = gufo_npu::serve::FinishReason::kLength;
    return result;
  }

private:
  gufo_npu::serve::ModelInfo info_;
  std::mutex mutex_;
};

void ServerEndToEnd() {
  SamplingEngine engine;
  const auto format = gufo_npu::serve::NewQwen35ChatFormat();
  gufo_npu::serve::ServerOptions options;
  options.host = "127.0.0.1";
  options.port = 0;
  gufo_npu::serve::Server server(engine, *format, options);
  const int port = server.Bind();
  std::thread thread([&] { server.Run(); });
  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(60, 0);
  const ordered_json messages = {{{"role", "user"}, {"content", "hi"}}};
  auto post = [&](const ordered_json& body) {
    return client.Post("/v1/chat/completions", body.dump(), "application/json");
  };

  const json schema = {{"type", "object"},
                       {"properties",
                        {{"name", {{"type", "string"}, {"maxLength", 8}}},
                         {"age", {{"type", "integer"}}}}},
                       {"required", {"name", "age"}},
                       {"additionalProperties", false}};
  const ordered_json body = {
      {"messages", messages},
      {"max_tokens", 400},
      {"response_format",
       {{"type", "json_schema"},
        {"json_schema",
         {{"name", "person"},
          {"strict", true},
          {"schema", ordered_json::parse(schema.dump())}}}}}};
  auto res = post(body);
  const json m1 = [&] {
    if (!res || res->status != 200) {
      return json();
    }
    const json b = json::parse(res->body, nullptr, false);
    json m = b["choices"][0]["message"];
    m["finish_reason"] = b["choices"][0]["finish_reason"];
    return m;
  }();
  const std::string content = m1.is_object() && m1["content"].is_string()
                                  ? m1["content"].get<std::string>()
                                  : std::string();
  const json value = json::parse(content, nullptr, false);
  Check(!value.is_discarded() && Validate(value, schema, schema) &&
            m1["finish_reason"] == "stop",
        "server: json_schema reply validates: " + content);

  // The reply's message, or null when the request failed.
  auto message = [](const httplib::Result& r) {
    if (!r || r->status != 200) {
      return json();
    }
    const json body = json::parse(r->body, nullptr, false);
    if (!body.is_object() || !body.contains("choices")) {
      return json();
    }
    json m = body["choices"][0]["message"];
    m["finish_reason"] = body["choices"][0]["finish_reason"];
    return m;
  };
  res = post({{"messages", messages},
              {"max_tokens", 400},
              {"response_format", {{"type", "json_object"}}}});
  const json m2 = message(res);
  const json object =
      m2.is_object() && m2["content"].is_string()
          ? json::parse(m2["content"].get<std::string>(), nullptr, false)
          : json();
  Check(res && res->status == 200 && object.is_object(),
        "server: json_object reply is an object");

  const ordered_json tools = ordered_json::parse(R"([
    {"type": "function", "function": {"name": "get_weather",
      "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}},
       "required": ["city", "days"]}}},
    {"type": "function", "function": {"name": "get_time"}}])");
  res =
      post({{"messages", messages},
            {"max_tokens", 400},
            {"tools", tools},
            {"tool_choice",
             {{"type", "function"}, {"function", {{"name", "get_weather"}}}}}});
  const json msg = message(res);
  const bool has_call = msg.is_object() && msg.contains("tool_calls") &&
                        msg["tool_calls"].is_array() &&
                        !msg["tool_calls"].empty();
  const json args =
      has_call ? json::parse(msg["tool_calls"][0]["function"]["arguments"]
                                 .get<std::string>(),
                             nullptr, false)
               : json();
  Check(has_call && msg["finish_reason"] == "tool_calls" &&
            msg["tool_calls"][0]["function"]["name"] == "get_weather" &&
            args.is_object() && args.contains("city") &&
            args["days"].is_number_integer(),
        "server: tool_choice named function gives that call: " + msg.dump());

  res = post({{"messages", messages},
              {"response_format",
               {{"type", "json_schema"},
                {"json_schema",
                 {{"name", "x"}, {"schema", {{"$ref", "#/$defs/none"}}}}}}}});
  Check(res && res->status == 400 &&
            json::parse(res->body)["error"]["param"] == "response_format",
        "server: a schema it cannot compile is a 400");

  server.Stop();
  thread.join();
}

// --- The mask's cost on a 248k-token vocabulary. --------------------------

void Benchmark() {
  std::mt19937 rng(1);
  // Byte frequencies roughly like a BPE vocabulary: letters, spaces,
  // digits, punctuation; quotes, backslashes and newlines are rarer.
  const std::string common =
      "etaoinshrdlucmfwypvbgkjqxzETAOINSHRDLUCMFWYPVBGKJQXZ     0123456789";
  const std::string rare = ".,:;!?-_()[]{}<>/=+*#@$%&'\"\\\n\t";
  std::uniform_int_distribution<int> len(1, 10);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<std::string> bytes;
  std::vector<std::uint8_t> special;
  bytes.reserve(248320);
  for (int c = 0; c < 256; ++c) {
    bytes.emplace_back(1, static_cast<char>(c));
    special.push_back(0);
  }
  while (bytes.size() < 248000) {
    std::string tok;
    const int n = len(rng);
    for (int i = 0; i < n; ++i) {
      const std::string& pool = u(rng) < 0.93 ? common : rare;
      tok += pool[static_cast<std::size_t>(rng() % pool.size())];
    }
    bytes.push_back(tok);
    special.push_back(0);
  }
  const int end = static_cast<int>(bytes.size());
  bytes.push_back("<|im_end|>");
  special.push_back(1);
  for (int i = 0; i < 300; ++i) {
    bytes.push_back("<|extra_" + std::to_string(i) + "|>");
    special.push_back(1);
  }
  const int vocab = static_cast<int>(bytes.size());
  auto find = [&](const std::string& b) {
    return static_cast<int>(std::find(bytes.begin(), bytes.end(), b) -
                            bytes.begin());
  };
  const auto t0 = std::chrono::steady_clock::now();
  const auto index = std::make_shared<const VocabIndex>(bytes, special);
  const double build =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std::printf("bench: vocabulary index for %d tokens built in %.0f ms\n", vocab,
              build * 1e3);

  const ordered_json schema = {
      {"type", "object"},
      {"properties",
       {{"name", {{"type", "string"}}},
        {"age", {{"type", "number"}}},
        {"tags", {{"type", "array"}, {"items", {{"type", "string"}}}}}}},
      {"required", {"name", "age"}},
      {"additionalProperties", false}};
  const ordered_json tools = ordered_json::parse(R"([
    {"type": "function", "function": {"name": "f",
      "parameters": {"type": "object", "properties": {
        "text": {"type": "string"}}}}}])");
  struct Case {
    const char* label;
    std::shared_ptr<TokenConstraint> c;
    std::string prefix;
  };
  const std::vector<Case> cases = {
      {"object start",
       NewGrammarConstraint(JsonSchemaGrammar(schema, true), index, {end},
                            false),
       ""},
      {"after '{'",
       NewGrammarConstraint(JsonSchemaGrammar(schema, true), index, {end},
                            false),
       "{"},
      {"in a string value",
       NewGrammarConstraint(JsonSchemaGrammar(schema, true), index, {end},
                            false),
       "{\"name\": \"ab"},
      {"in a number",
       NewGrammarConstraint(JsonSchemaGrammar(schema, true), index, {end},
                            false),
       "{\"name\": \"a\", \"age\": 1"},
      {"json_object value",
       NewGrammarConstraint(JsonObjectGrammar(), index, {end}, false),
       "{\"k\": "},
      {"tool call raw text",
       NewGrammarConstraint(ToolCallGrammar(tools, ""), index, {end}, false),
       "<tool_call>\n<function=f>\n<parameter=text>\nhello"},
  };
  std::vector<std::uint8_t> mask;
  for (const auto& k : cases) {
    k.c->Reset();
    for (const char ch : k.prefix) {
      k.c->Advance(find(std::string(1, ch)));
    }
    k.c->Allowed(vocab, mask);  // warm
    constexpr int kIters = 20;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
      k.c->Allowed(vocab, mask);
    }
    const double ms =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count() *
        1e3 / kIters;
    const auto allowed = std::count(mask.begin(), mask.end(), 1);
    std::printf("bench: mask %-20s %7.2f ms per token (%ld allowed)\n", k.label,
                ms, static_cast<long>(allowed));
    Check(ms < 250.0, std::string("bench: ") + k.label + " under 250 ms");
    int single = 0;
    const auto s0 = std::chrono::steady_clock::now();
    for (int id = 0; id < 2000; ++id) {
      single += k.c->IsAllowed(id) ? 1 : 0;
    }
    const double us =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - s0)
            .count() *
        1e6 / 2000;
    std::printf("bench: one-token check %-20s %6.2f us\n", k.label, us);
  }
}

}  // namespace

// --- Structured output together with logprobs, n and several models. ----

void CrossFeatures() {
  using gufo_npu::serve::EngineOptions;
  const json schema = {{"type", "object"},
                       {"properties",
                        {{"name", {{"type", "string"}, {"maxLength", 8}}},
                         {"age", {{"type", "integer"}}}}},
                       {"required", {"name", "age"}},
                       {"additionalProperties", false}};
  const ordered_json format_field = {
      {"type", "json_schema"},
      {"json_schema",
       {{"name", "person"},
        {"strict", true},
        {"schema", ordered_json::parse(schema.dump())}}}};
  const ordered_json messages = {{{"role", "user"}, {"content", "hi"}}};
  auto valid = [&](const json& choice) {
    if (!choice.is_object() || !choice["message"]["content"].is_string()) {
      return false;
    }
    const json v = json::parse(choice["message"]["content"].get<std::string>(),
                               nullptr, false);
    return !v.is_discarded() && Validate(v, schema, schema);
  };
  const auto format = gufo_npu::serve::NewQwen35ChatFormat();
  gufo_npu::serve::ServerOptions options;
  options.host = "127.0.0.1";
  options.port = 0;

  {
    SamplingEngine engine;
    gufo_npu::serve::Server server(engine, *format, options);
    const int port = server.Bind();
    std::thread thread([&] { server.Run(); });
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(60, 0);

    // json_schema + logprobs: the reply validates and every token has a
    // logprob (the raw model's, so it may be any value <= 0).
    auto res = client.Post("/v1/chat/completions",
                           ordered_json({{"messages", messages},
                                         {"max_tokens", 400},
                                         {"logprobs", true},
                                         {"top_logprobs", 2},
                                         {"response_format", format_field}})
                               .dump(),
                           "application/json");
    const json b1 = res && res->status == 200
                        ? json::parse(res->body, nullptr, false)
                        : json();
    const json c1 = b1.is_object() ? b1["choices"][0] : json();
    const json lp = c1.is_object() ? c1["logprobs"]["content"] : json();
    bool lp_ok = lp.is_array() && !lp.empty();
    std::string joined;
    if (lp_ok) {
      for (const auto& e : lp) {
        lp_ok = lp_ok && e["logprob"].get<double>() <= 1e-6 &&
                e["top_logprobs"].size() == 2;
        joined += e["token"].get<std::string>();
      }
    }
    Check(valid(c1) && lp_ok &&
              c1["message"]["content"].get<std::string>().find(joined) !=
                  std::string::npos,
          "json_schema + logprobs: valid reply, one logprob per token");

    // n = 2 with json_schema: both choices validate.
    res = client.Post("/v1/chat/completions",
                      ordered_json({{"messages", messages},
                                    {"max_tokens", 400},
                                    {"n", 2},
                                    {"response_format", format_field}})
                          .dump(),
                      "application/json");
    const json b2 = res && res->status == 200
                        ? json::parse(res->body, nullptr, false)
                        : json();
    Check(b2.is_object() && b2["choices"].size() == 2 &&
              valid(b2["choices"][0]) && valid(b2["choices"][1]),
          "n = 2 + json_schema: both choices validate");
    server.Stop();
    thread.join();
  }

  {
    // Several models: a constrained request on each side of a switch uses
    // that model's vocabulary and validates.
    std::vector<EngineOptions> models(2);
    models[0].model_path = "toy-a";
    models[1].model_path = "toy-b";
    auto make = [](const EngineOptions& o) {
      return std::make_unique<SamplingEngine>(o.model_path);
    };
    gufo_npu::serve::EnginePool pool(models, make, make);
    pool.Preload();
    gufo_npu::serve::Server server(pool, *format, options);
    const int port = server.Bind();
    std::thread thread([&] { server.Run(); });
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(60, 0);
    bool ok = true;
    for (const char* id : {"toy-a", "toy-b", "toy-a"}) {
      auto res = client.Post("/v1/chat/completions",
                             ordered_json({{"model", id},
                                           {"messages", messages},
                                           {"max_tokens", 400},
                                           {"response_format", format_field}})
                                 .dump(),
                             "application/json");
      const json b = res && res->status == 200
                         ? json::parse(res->body, nullptr, false)
                         : json();
      ok = ok && b.is_object() && b["model"] == id && valid(b["choices"][0]) &&
           pool.status().loaded == std::string(id);
    }
    Check(ok, "pool + json_schema: valid replies across model switches");
    server.Stop();
    thread.join();
  }
}

int main(int argc, char** argv) {
  // --bench: only the mask benchmark (the image build prints it).
  if (argc > 1 && std::string(argv[1]) == "--bench") {
    Benchmark();
    return failures > 0 ? 1 : 0;
  }
  Schemas();
  Specifics();
  ToolCalls();
  Errors();
  SamplerAndDrafts();
  ServerEndToEnd();
  CrossFeatures();
  Benchmark();
  if (failures > 0) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("all structured-output checks passed\n");
  return 0;
}
