// The server's smaller OpenAI features over HTTP with the scripted engine
// (tests/serve/fakes.hpp): logprobs, n > 1, logit_bias,
// parallel_tool_calls and the legacy /v1/completions API.
#include <httplib/httplib.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "fakes.hpp"
#include "serve/server.hpp"

namespace {

using gufo_npu::serve::Server;
using gufo_npu::serve::ServerOptions;
using gufo_npu::serve::testing::FakeEngine;
using gufo_npu::serve::testing::FakeFormat;
using gufo_npu::serve::testing::FakeLogprob;
using nlohmann::json;

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

struct Running {
  Running(FakeEngine& engine, const FakeFormat& format)
      : server(engine, format, [] {
          ServerOptions o;
          o.host = "127.0.0.1";
          o.port = 0;
          return o;
        }()) {
    port = server.Bind();
    thread = std::thread([this] { server.Run(); });
  }
  ~Running() {
    server.Stop();
    thread.join();
  }
  httplib::Client Client() const {
    httplib::Client c("127.0.0.1", port);
    c.set_read_timeout(30, 0);
    return c;
  }

  Server server;
  int port = 0;
  std::thread thread;
};

json Messages() {
  return json::array({{{"role", "user"}, {"content", "hello"}}});
}

struct Reply {
  int status = 0;
  json body;
};

Reply Post(const Running& s, const std::string& path, const json& body) {
  auto res = s.Client().Post(path, body.dump(), "application/json");
  if (!res) {
    return {};
  }
  return {res->status, json::parse(res->body, nullptr, false)};
}

Reply Chat(const Running& s, const json& body) {
  return Post(s, "/v1/chat/completions", body);
}

Reply Completion(const Running& s, const json& body) {
  return Post(s, "/v1/completions", body);
}

// The `data:` events of a streamed request ("[DONE]" as a string).
std::vector<json> Stream(const Running& s, const std::string& path, json body) {
  body["stream"] = true;
  std::vector<json> events;
  std::string buf;
  httplib::Request req;
  req.method = "POST";
  req.path = path;
  req.body = body.dump();
  req.set_header("Content-Type", "application/json");
  req.content_receiver = [&](const char* data, std::size_t len, std::uint64_t,
                             std::uint64_t) {
    buf.append(data, len);
    for (std::size_t end; (end = buf.find("\n\n")) != std::string::npos;) {
      const std::string event = buf.substr(0, end);
      buf.erase(0, end + 2);
      const std::string payload = event.substr(6);
      events.push_back(payload == "[DONE]"
                           ? json("[DONE]")
                           : json::parse(payload, nullptr, false));
    }
    return true;
  };
  auto c = s.Client();
  Check(static_cast<bool>(c.send(req)), "stream: request failed");
  return events;
}

bool Near(const json& v, double want) {
  return v.is_number() && std::fabs(v.get<double>() - want) < 1e-4;
}

void Logprobs(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format);
  engine.Script({"He", "llo"});
  const auto r = Chat(
      s, {{"messages", Messages()}, {"logprobs", true}, {"top_logprobs", 2}});
  Check(r.status == 200, "logprobs: status " + r.body.dump());
  const auto& lp = r.body["choices"][0]["logprobs"]["content"];
  // Two content tokens; the stop token has none.
  Check(lp.is_array() && lp.size() == 2,
        "logprobs: one per token " + r.body.dump());
  if (lp.size() == 2) {
    Check(lp[0]["token"] == "He" && lp[1]["token"] == "llo" &&
              lp[0]["bytes"] == json::array({72, 101}) &&
              Near(lp[0]["logprob"], FakeLogprob()),
          "logprobs: token, bytes, logprob " + lp.dump());
    Check(lp[0]["top_logprobs"].size() == 2 &&
              lp[0]["top_logprobs"][0]["token"] == "He" &&
              Near(lp[0]["top_logprobs"][0]["logprob"], FakeLogprob()) &&
              lp[0]["top_logprobs"][1].contains("bytes"),
          "logprobs: top_logprobs best first " + lp[0].dump());
  }
  Check(engine.last().sampling.observer == nullptr,
        "logprobs: the fake records requests without the observer");

  // Without logprobs the field is null.
  engine.Script({"x"});
  Check(Chat(s, {{"messages", Messages()}})
            .body["choices"][0]["logprobs"]
            .is_null(),
        "logprobs: null when not asked");

  // Streaming: the entries ride on the chunks, all of them once.
  engine.Script({"a", "b", "c"});
  std::size_t entries = 0;
  for (const auto& e : Stream(s, "/v1/chat/completions",
                              {{"messages", Messages()}, {"logprobs", true}})) {
    if (e.is_object() && !e["choices"].empty() &&
        e["choices"][0]["logprobs"].is_object()) {
      entries += e["choices"][0]["logprobs"]["content"].size();
    }
  }
  Check(entries == 3, "logprobs: streamed entries " + std::to_string(entries));

  Check(Chat(s, {{"messages", Messages()}, {"top_logprobs", 2}}).status == 400,
        "logprobs: top_logprobs needs logprobs");
  Check(
      Chat(s,
           {{"messages", Messages()}, {"logprobs", true}, {"top_logprobs", 21}})
              .status == 400,
      "logprobs: top_logprobs at most 20");
}

void Choices(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format);
  engine.Script({"x", "y"});
  const int before = engine.generations();
  const auto r = Chat(
      s,
      {{"messages", Messages()}, {"n", 3}, {"seed", 7}, {"temperature", 0.8}});
  Check(r.status == 200 && r.body["choices"].size() == 3,
        "n: three choices " + r.body.dump());
  for (int i = 0; i < 3 && i < static_cast<int>(r.body["choices"].size());
       ++i) {
    Check(r.body["choices"][i]["index"] == i &&
              r.body["choices"][i]["message"]["content"] == "xy",
          "n: choice " + std::to_string(i));
  }
  Check(engine.generations() - before == 3, "n: three generations");
  const auto history = engine.history();
  Check(history.size() >= 3 &&
            history[history.size() - 3].sampling.seed == 7u &&
            history[history.size() - 2].sampling.seed == 8u &&
            history[history.size() - 1].sampling.seed == 9u,
        "n: each choice its own seed");
  // The prompt counts once, completions add up (3 tokens each).
  Check(r.body["usage"]["prompt_tokens"] == 6 &&
            r.body["usage"]["completion_tokens"] == 9,
        "n: usage " + r.body["usage"].dump());

  // Streaming: each choice has its role, content and finish, by index.
  std::string text[2];
  int finishes = 0;
  int roles = 0;
  const auto events = Stream(s, "/v1/chat/completions",
                             {{"messages", Messages()},
                              {"n", 2},
                              {"stream_options", {{"include_usage", true}}}});
  for (const auto& e : events) {
    if (!e.is_object() || e["choices"].empty()) {
      continue;
    }
    const auto& c = e["choices"][0];
    const int i = c["index"];
    if (c["delta"].contains("role")) {
      ++roles;
    }
    if (c["delta"].contains("content")) {
      text[i] += c["delta"]["content"].get<std::string>();
    }
    if (c["finish_reason"].is_string()) {
      ++finishes;
    }
  }
  Check(roles == 2 && finishes == 2 && text[0] == "xy" && text[1] == "xy" &&
            events.back() == "[DONE]",
        "n: streamed choices");
  Check(events.size() >= 2 &&
            events[events.size() - 2]["usage"]["completion_tokens"] == 6,
        "n: streamed usage");
}

void LogitBias(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format);
  engine.Script({"x"});
  const auto r = Chat(
      s, {{"messages", Messages()}, {"logit_bias", {{"5", -100}, {"7", 2.5}}}});
  Check(r.status == 200, "logit_bias: status");
  const auto& bias = engine.last().sampling.logit_bias;
  Check(bias.size() == 2 && bias[0] == std::pair<int, float>{5, -100.0f} &&
            bias[1] == std::pair<int, float>{7, 2.5f},
        "logit_bias: reaches the engine");
  Check(Chat(s, {{"messages", Messages()}, {"logit_bias", {{"100", 1}}}})
                .status == 400,
        "logit_bias: id outside the vocab");
  Check(Chat(s, {{"messages", Messages()}, {"logit_bias", {{"3", 101}}}})
                .status == 400,
        "logit_bias: value out of range");
  Check(
      Chat(s, {{"messages", Messages()}, {"logit_bias", {{"x", 1}}}}).status ==
          400,
      "logit_bias: key not an id");
}

void ParallelToolCalls(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format);
  const json tools =
      json::array({{{"type", "function"}, {"function", {{"name", "get"}}}}});
  const std::vector<std::string> script = {
      "<tool_call>", R"({"name":"get","arguments":{"a":1}})", "</tool_call>",
      "<tool_call>", R"({"name":"get","arguments":{"a":2}})", "</tool_call>",
      "tail"};
  engine.Script(script);
  const auto both = Chat(s, {{"messages", Messages()}, {"tools", tools}});
  Check(both.body["choices"][0]["message"]["tool_calls"].size() == 2,
        "parallel_tool_calls: default keeps both");
  engine.Script(script);
  const auto one = Chat(s, {{"messages", Messages()},
                            {"tools", tools},
                            {"parallel_tool_calls", false}});
  const auto& msg = one.body["choices"][0]["message"];
  Check(one.status == 200 && msg["tool_calls"].size() == 1 &&
            msg["tool_calls"][0]["function"]["arguments"] == "{\"a\":1}" &&
            one.body["choices"][0]["finish_reason"] == "tool_calls",
        "parallel_tool_calls: false keeps the first " + one.body.dump());
  Check(engine.last_result().tokens.size() == 3,
        "parallel_tool_calls: generation ends after the first call");
}

void Completions(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format);
  engine.Script({"wor", "ld"});
  auto r = Completion(s, {{"model", "m"}, {"prompt", "hello "}});
  Check(r.status == 200 && r.body["object"] == "text_completion" &&
            r.body["id"].get<std::string>().rfind("cmpl-", 0) == 0 &&
            r.body["choices"][0]["text"] == "world" &&
            r.body["choices"][0]["finish_reason"] == "stop" &&
            r.body["choices"][0]["logprobs"].is_null(),
        "completions: text " + r.body.dump());
  Check(engine.last().prompt.size() == 6 && engine.last().max_tokens == 16,
        "completions: raw prompt, max_tokens 16 by default");
  Check(r.body["usage"]["prompt_tokens"] == 6 &&
            r.body["usage"]["completion_tokens"] == 3,
        "completions: usage");

  engine.Script({"!"});
  r = Completion(s, {{"prompt", "hi"}, {"echo", true}});
  Check(r.body["choices"][0]["text"] == "hi!", "completions: echo");

  // Two prompts, two choices each: index = prompt * n + choice.
  engine.Script({"z"});
  r = Completion(s, {{"prompt", {"a", "bb"}}, {"n", 2}});
  Check(r.status == 200 && r.body["choices"].size() == 4 &&
            r.body["choices"][3]["index"] == 3 &&
            r.body["usage"]["prompt_tokens"] == 3,
        "completions: prompts times n " + r.body.dump());

  // Token-id prompts go to the engine as they are.
  engine.Script({"q"});
  r = Completion(s, {{"prompt", {4, 5, 6}}});
  Check(r.status == 200 && engine.last().prompt == std::vector<int>{4, 5, 6},
        "completions: token ids");
  Check(Completion(s, {{"prompt", {4, 100}}}).status == 400,
        "completions: token id outside the vocab");

  // Legacy logprobs.
  engine.Script({"ab", "c"});
  r = Completion(s, {{"prompt", "x"}, {"logprobs", 2}});
  const auto& lp = r.body["choices"][0]["logprobs"];
  Check(lp["tokens"] == json::array({"ab", "c"}) &&
            lp["text_offset"] == json::array({0, 2}) &&
            lp["token_logprobs"].size() == 2 &&
            Near(lp["token_logprobs"][0], FakeLogprob()) &&
            lp["top_logprobs"][0].size() == 2 &&
            Near(lp["top_logprobs"][0]["ab"], FakeLogprob()),
        "completions: legacy logprobs " + lp.dump());

  // Streaming.
  engine.Script({"s", "t"});
  std::string text;
  std::string finish;
  const auto events = Stream(s, "/v1/completions", {{"prompt", "p"}});
  for (const auto& e : events) {
    if (e.is_object() && !e["choices"].empty()) {
      text += e["choices"][0]["text"].get<std::string>();
      if (e["choices"][0]["finish_reason"].is_string()) {
        finish = e["choices"][0]["finish_reason"];
      }
    }
  }
  Check(text == "st" && finish == "stop" && events.back() == "[DONE]",
        "completions: stream");

  Check(Completion(s, {{"prompt", "x"}, {"suffix", "y"}}).status == 400,
        "completions: suffix refused");
  Check(Completion(s, {{"prompt", "x"}, {"echo", true}, {"logprobs", 1}})
                .status == 400,
        "completions: echo with logprobs refused");
  Check(Completion(s, {{"prompt", json::array()}}).status == 400,
        "completions: empty prompt array");
  Check(Completion(s, {{"prompt", "x"}, {"n", 9}}).status == 400,
        "completions: n above 8");
}

}  // namespace

int main() {
  FakeEngine engine;
  FakeFormat format;
  Logprobs(engine, format);
  Choices(engine, format);
  LogitBias(engine, format);
  ParallelToolCalls(engine, format);
  Completions(engine, format);
  if (failures > 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::puts("ok: logprobs, n, logit_bias, parallel_tool_calls, completions");
  return 0;
}
