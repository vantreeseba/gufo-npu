// The server's pieces without HTTP: UTF-8 assembly, stop strings, request
// parsing and the queue.
#include "serve/text.hpp"

#include <cstdio>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "serve/openai.hpp"
#include "serve/scheduler.hpp"

namespace {

using gufo_npu::serve::ApiError;
using gufo_npu::serve::ChatRequest;
using gufo_npu::serve::ParseChatRequest;
using gufo_npu::serve::Scheduler;
using gufo_npu::serve::StopMatcher;
using gufo_npu::serve::Utf8Assembler;
using json = nlohmann::ordered_json;

int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

std::string FeedAll(Utf8Assembler& a, const std::vector<std::string>& in) {
  std::string out;
  for (const auto& piece : in) {
    out += a.Feed(piece);
  }
  return out + a.Finish();
}

void Utf8() {
  Utf8Assembler a;
  Check(a.Feed("\xC3") == "", "utf8: half of é is held");
  Check(a.Feed("\xA9!") == "\xC3\xA9!", "utf8: é completes");
  Check(FeedAll(a, {"\xE2\x82", "\xAC"}) == "\xE2\x82\xAC", "utf8: € in 2");
  Check(FeedAll(a, {"\xF0", "\x9F", "\x98", "\x80"}) == "\xF0\x9F\x98\x80",
        "utf8: emoji byte by byte");
  const std::string fffd = "\xEF\xBF\xBD";
  Check(FeedAll(a, {"a\xFF"
                    "b"}) == "a" + fffd + "b",
        "utf8: bad byte");
  Check(FeedAll(a, {"\xC3"
                    "a"}) == fffd + "a",
        "utf8: broken sequence");
  Check(FeedAll(a, {"x\xE2\x82"}) == "x" + fffd, "utf8: cut at the end");
  Check(FeedAll(a, {"\xED\xA0\x80"}) == fffd + fffd + fffd,
        "utf8: surrogate rejected");
  Check(FeedAll(a, {"\xC0\xAF"}) == fffd + fffd, "utf8: overlong rejected");
}

std::string StopAll(StopMatcher& m, const std::vector<std::string>& in) {
  std::string out;
  for (const auto& piece : in) {
    out += m.Feed(piece);
  }
  return out + m.Finish();
}

void Stops() {
  {
    StopMatcher m({"STOP"});
    Check(m.Feed("ab") == "ab", "stop: plain text passes");
    Check(m.Feed("c ST") == "c ", "stop: possible start held");
    Check(m.Feed("OP x") == "", "stop: match across pieces");
    Check(m.stopped(), "stop: stopped");
    Check(m.Feed("more") == "" && m.Finish() == "", "stop: nothing after");
  }
  {
    StopMatcher m({"STOP"});
    Check(StopAll(m, {"ST", "ART"}) == "START" && !m.stopped(),
          "stop: false start released");
  }
  {
    StopMatcher m({"\n\n", "END"});
    Check(StopAll(m, {"a\n", "b", "E", "N", "D!"}) == "a\nb" && m.stopped(),
          "stop: second of two stops");
  }
  {
    StopMatcher m({});
    Check(StopAll(m, {"x", "y"}) == "xy", "stop: none");
  }
  {
    StopMatcher m({"aab"});
    Check(StopAll(m, {"aa", "aab"}) == "aa" && m.stopped(),
          "stop: overlapping prefix");
  }
}

bool Rejects(const json& body, const char* param) {
  try {
    ParseChatRequest(body);
  } catch (const ApiError& e) {
    return e.status == 400 && e.param == param;
  }
  return false;
}

void Requests() {
  const json messages = json::array({{{"role", "user"}, {"content", "hi"}}});
  {
    const auto r = ParseChatRequest({{"messages", messages}});
    Check(!r.stream && !r.max_tokens && r.sampling.greedy() &&
              !r.prompt.thinking && r.stop.empty(),
          "parse: defaults");
  }
  {
    const auto r =
        ParseChatRequest({{"messages", messages},
                          {"max_tokens", 9},
                          {"max_completion_tokens", 7},
                          {"temperature", 0.7},
                          {"top_p", 0.9},
                          {"top_k", 40},
                          {"min_p", 0.05},
                          {"presence_penalty", 0.5},
                          {"frequency_penalty", -0.5},
                          {"repetition_penalty", 1.1},
                          {"seed", 42},
                          {"stop", "END"},
                          {"stream", true},
                          {"stream_options", {{"include_usage", true}}},
                          {"chat_template_kwargs", {{"enable_thinking", true}}},
                          {"response_format", {{"type", "text"}}},
                          {"n", 1},
                          {"user", "ignored"}});
    Check(r.max_tokens == 7, "parse: max_completion_tokens wins");
    Check(r.sampling.temperature == 0.7f && r.sampling.top_p == 0.9f &&
              r.sampling.top_k == 40 && r.sampling.min_p == 0.05f &&
              r.sampling.presence_penalty == 0.5f &&
              r.sampling.frequency_penalty == -0.5f &&
              r.sampling.repetition_penalty == 1.1f && r.sampling.seed == 42u,
          "parse: sampling");
    Check(r.stop == std::vector<std::string>{"END"}, "parse: stop string");
    Check(r.stream && r.include_usage && r.prompt.thinking,
          "parse: stream, usage, thinking");
  }
  Check(ParseChatRequest({{"messages", messages}, {"reasoning_effort", "high"}})
            .prompt.thinking,
        "parse: reasoning_effort high thinks");
  Check(!ParseChatRequest({{"messages", messages},
                           {"reasoning_effort", "none"},
                           {"chat_template_kwargs", json::object()}})
             .prompt.thinking,
        "parse: reasoning_effort none");
  Check(!ParseChatRequest(
             {{"messages", messages},
              {"reasoning_effort", "high"},
              {"chat_template_kwargs", {{"enable_thinking", false}}}})
             .prompt.thinking,
        "parse: enable_thinking wins");
  const json tools =
      json::array({{{"type", "function"}, {"function", {{"name", "get"}}}}});
  Check(ParseChatRequest({{"messages", messages}, {"tools", tools}})
                .prompt.tools.size() == 1,
        "parse: tools kept");
  Check(ParseChatRequest(
            {{"messages", messages}, {"tools", tools}, {"tool_choice", "none"}})
            .prompt.tools.empty(),
        "parse: tool_choice none drops tools");

  Check(Rejects(json::array(), ""), "reject: not an object");
  Check(Rejects(json::object(), "messages"), "reject: no messages");
  Check(Rejects({{"messages", json::array()}}, "messages"),
        "reject: empty messages");
  Check(Rejects({{"messages", messages}, {"n", 9}}, "n"), "reject: n=9");
  Check(Rejects({{"messages", messages}, {"temperature", 3}}, "temperature"),
        "reject: temperature 3");
  Check(Rejects({{"messages", messages}, {"top_p", 0}}, "top_p"),
        "reject: top_p 0");
  Check(Rejects({{"messages", messages}, {"max_tokens", 0}}, "max_tokens"),
        "reject: max_tokens 0");
  Check(Rejects({{"messages", messages}, {"max_tokens", "9"}}, "max_tokens"),
        "reject: max_tokens string");
  Check(Rejects({{"messages", messages}, {"stop", 5}}, "stop"),
        "reject: stop number");
  {
    const auto r = ParseChatRequest({{"messages", messages},
                                     {"tools", tools},
                                     {"tool_choice", "required"}});
    Check(r.tool_required && r.tool_name.empty() && r.prompt.tools.size() == 1,
          "parse: tool_choice required");
    const auto named = ParseChatRequest(
        {{"messages", messages},
         {"tools", tools},
         {"tool_choice",
          {{"type", "function"}, {"function", {{"name", "get"}}}}}});
    Check(named.tool_required && named.tool_name == "get",
          "parse: tool_choice names a function");
  }
  Check(Rejects({{"messages", messages}, {"tool_choice", "required"}},
                "tool_choice"),
        "reject: tool_choice required without tools");
  Check(Rejects({{"messages", messages},
                 {"tools", tools},
                 {"tool_choice",
                  {{"type", "function"}, {"function", {{"name", "other"}}}}}},
                "tool_choice"),
        "reject: tool_choice names an unknown function");
  Check(Rejects({{"messages", messages},
                 {"tools", tools},
                 {"tool_choice", "sometimes"}},
                "tool_choice"),
        "reject: tool_choice sometimes");
  Check(Rejects({{"messages", messages}, {"tools", json::array({1})}}, "tools"),
        "reject: bad tool");
  Check(Rejects({{"messages", messages}, {"top_logprobs", 3}}, "top_logprobs"),
        "reject: top_logprobs without logprobs");
  Check(ParseChatRequest({{"messages", messages},
                          {"response_format", {{"type", "json_object"}}}})
                .format == ChatRequest::Format::kJsonObject,
        "parse: json_object");
  {
    const auto r = ParseChatRequest({{"messages", messages},
                                     {"response_format",
                                      {{"type", "json_schema"},
                                       {"json_schema",
                                        {{"name", "x"},
                                         {"strict", true},
                                         {"schema", {{"type", "object"}}}}}}}});
    Check(r.format == ChatRequest::Format::kJsonSchema && r.json_strict &&
              r.json_schema == json{{"type", "object"}},
          "parse: json_schema");
  }
  Check(Rejects({{"messages", messages},
                 {"response_format", {{"type", "json_schema"}}}},
                "response_format"),
        "reject: json_schema without a schema");
  Check(
      Rejects({{"messages", messages}, {"response_format", {{"type", "xml"}}}},
              "response_format"),
      "reject: response_format xml");
  Check(Rejects({{"messages", messages}, {"reasoning_effort", "max"}},
                "reasoning_effort"),
        "reject: reasoning_effort max");
}

void Queue() {
  Scheduler s(1);
  auto a = s.Enter();
  Check(a && a->Wait(nullptr) && s.running(), "queue: first runs");
  auto b = s.Enter();
  Check(b && s.waiting() == 1, "queue: second waits");
  Check(!s.Enter(), "queue: third refused");
  bool gave_up = !b->Wait([] { return true; });
  Check(gave_up && s.waiting() == 0, "queue: abandoned place freed");
  b.reset();
  auto c = s.Enter();
  Check(c && s.waiting() == 1, "queue: place reused");
  a.reset();
  Check(c->Wait(nullptr) && s.running() && s.waiting() == 0,
        "queue: next runs when the first ends");
  c.reset();
  Check(!s.running(), "queue: idle");

  Scheduler none(0);
  auto d = none.Enter();
  Check(d != nullptr, "queue 0: idle server takes one");
  Check(!none.Enter(), "queue 0: busy server refuses");
}

}  // namespace

int main() {
  Utf8();
  Stops();
  Requests();
  Queue();
  if (failures > 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::puts("ok: utf8, stop strings, request parsing, queue");
  return 0;
}
