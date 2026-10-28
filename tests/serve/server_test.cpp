// The server over HTTP on localhost, with a scripted engine and chat format
// (tests/serve/fakes.hpp): OpenAI request and response shapes, streaming,
// stop strings, UTF-8, tool calls, errors, the queue and cancellation.
#include "serve/server.hpp"

#include <httplib/httplib.h>

#include <chrono>
#include <cstdio>
#include <functional>
#include <future>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "fakes.hpp"

namespace {

using gufo_npu::serve::Server;
using gufo_npu::serve::ServerOptions;
using gufo_npu::serve::testing::FakeEngine;
using gufo_npu::serve::testing::FakeFormat;
using gufo_npu::serve::testing::kStopId;
using nlohmann::json;
using namespace std::chrono_literals;

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

// A server on a free localhost port, serving until destroyed.
struct Running {
  Running(FakeEngine& engine, const FakeFormat& format, ServerOptions options)
      : server(engine, format, [&] {
          options.host = "127.0.0.1";
          options.port = 0;
          return options;
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

json Messages(const std::string& text = "hello") {
  return json::array({{{"role", "user"}, {"content", text}}});
}

struct Reply {
  int status = 0;
  json body;
};

Reply Post(const Running& s, const json& body,
           const httplib::Headers& headers = {}) {
  auto c = s.Client();
  auto res =
      c.Post("/v1/chat/completions", headers, body.dump(), "application/json");
  if (!res) {
    return {};
  }
  Reply r{res->status, json()};
  if (!res->body.empty()) {
    r.body = json::parse(res->body, nullptr, false);
  }
  return r;
}

// Streams a request; returns the parsed `data:` events (the last is the
// string "[DONE]"). `on_event` returning false hangs up.
std::vector<json> Stream(const Running& s, json body,
                         const std::function<bool(const json&)>& on_event = {},
                         int* status = nullptr) {
  body["stream"] = true;
  auto c = s.Client();
  std::vector<json> events;
  std::string buf;
  bool hung_up = false;
  httplib::Request req;
  req.method = "POST";
  req.path = "/v1/chat/completions";
  req.body = body.dump();
  req.set_header("Content-Type", "application/json");
  req.content_receiver = [&](const char* data, std::size_t len, std::uint64_t,
                             std::uint64_t) {
    buf.append(data, len);
    for (std::size_t end; (end = buf.find("\n\n")) != std::string::npos;) {
      const std::string event = buf.substr(0, end);
      buf.erase(0, end + 2);
      if (event.rfind("data: ", 0) != 0) {
        Check(false, "stream: event without data: " + event);
        continue;
      }
      const std::string data = event.substr(6);
      events.push_back(data == "[DONE]" ? json("[DONE]")
                                        : json::parse(data, nullptr, false));
      if (events.back().is_discarded()) {
        Check(false, "stream: bad JSON " + data);
      }
      if (on_event && !on_event(events.back())) {
        hung_up = true;
        return false;
      }
    }
    return true;
  };
  auto res = c.send(req);
  if (status != nullptr) {
    *status = res ? res->status : 0;
  }
  Check(res || hung_up, "stream: request failed");
  return events;
}

std::string Joined(const std::vector<json>& events, const char* field) {
  std::string out;
  for (const auto& e : events) {
    if (e.is_object() && !e["choices"].empty()) {
      const auto& d = e["choices"][0]["delta"];
      if (d.contains(field)) {
        out += d[field].get<std::string>();
      }
    }
  }
  return out;
}

bool WaitFor(const std::function<bool()>& done,
             std::chrono::milliseconds limit = 5000ms) {
  const auto end = std::chrono::steady_clock::now() + limit;
  while (!done()) {
    if (std::chrono::steady_clock::now() > end) {
      return false;
    }
    std::this_thread::sleep_for(5ms);
  }
  return true;
}

void Basic(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format, {});
  engine.Script({"Hel", "lo"});
  const auto r = Post(s, {{"model", "anything"},
                          {"messages", Messages()},
                          {"max_tokens", 5},
                          {"temperature", 0.7},
                          {"top_p", 0.9},
                          {"seed", 42}});
  Check(r.status == 200, "basic: status");
  const auto& choice = r.body["choices"][0];
  Check(r.body["object"] == "chat.completion" &&
            r.body["model"] == "fake-model" &&
            r.body["id"].get<std::string>().rfind("chatcmpl-", 0) == 0,
        "basic: envelope");
  Check(choice["message"]["role"] == "assistant" &&
            choice["message"]["content"] == "Hello",
        "basic: content " + choice.dump());
  Check(choice["finish_reason"] == "stop", "basic: finish_reason");
  Check(r.body["usage"]["prompt_tokens"] == 6 &&
            r.body["usage"]["completion_tokens"] == 3 &&
            r.body["usage"]["total_tokens"] == 9 &&
            r.body["usage"]["prompt_tokens_details"]["cached_tokens"] == 3,
        "basic: usage " + r.body["usage"].dump());
  const auto last = engine.last();
  Check(last.max_tokens == 5 && last.sampling.temperature == 0.7f &&
            last.sampling.top_p == 0.9f && last.sampling.seed == 42u,
        "basic: request reaches the engine");
  Check(last.stop_ids == std::vector<int>{kStopId}, "basic: stop ids");
  Check(!format.last().thinking, "basic: thinking off by default");

  engine.Script({"a", "b", "c", "d"}, false);
  const auto len = Post(s, {{"messages", Messages()}, {"max_tokens", 2}});
  Check(len.body["choices"][0]["finish_reason"] == "length" &&
            len.body["choices"][0]["message"]["content"] == "ab",
        "basic: length " + len.body.dump());

  // No max_tokens: the rest of the context.
  engine.info_.context_length = 50;
  engine.Script({"x"});
  Post(s, {{"messages", Messages()}});
  Check(engine.last().max_tokens == 44, "basic: default max_tokens");
  Post(s, {{"messages", Messages()}, {"max_tokens", 1000}});
  Check(engine.last().max_tokens == 44, "basic: max_tokens clamped");
  const auto too_long = Post(s, {{"messages", Messages(std::string(60, 'x'))}});
  Check(too_long.status == 400 &&
            too_long.body["error"]["code"] == "context_length_exceeded",
        "basic: context exceeded " + too_long.body.dump());
  engine.info_.context_length = 4096;
}

void Streaming(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format, {});
  engine.Script({"Hel", "lo"});
  const auto events =
      Stream(s, {{"messages", Messages()},
                 {"stream_options", {{"include_usage", true}}}});
  Check(events.size() >= 5, "stream: event count");
  if (events.size() < 5) {
    return;
  }
  Check(events[0]["object"] == "chat.completion.chunk" &&
            events[0]["choices"][0]["delta"]["role"] == "assistant",
        "stream: role first");
  Check(Joined(events, "content") == "Hello", "stream: content");
  const auto& fin = events[events.size() - 3];
  Check(fin["choices"][0]["finish_reason"] == "stop" &&
            fin["choices"][0]["delta"].empty(),
        "stream: finish chunk " + fin.dump());
  const auto& usage = events[events.size() - 2];
  Check(usage["choices"].empty() && usage["usage"]["completion_tokens"] == 3,
        "stream: usage chunk " + usage.dump());
  Check(events.back() == "[DONE]", "stream: [DONE]");
  for (std::size_t i = 0; i + 2 < events.size(); ++i) {
    Check(events[i].contains("usage") && events[i]["usage"].is_null(),
          "stream: usage null before the end");
  }

  const auto plain = Stream(s, {{"messages", Messages()}});
  Check(!plain.empty() && !plain[0].contains("usage"),
        "stream: no usage field unless asked");
}

void StopStrings(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format, {});
  engine.Script({"ab", "c ST", "OP x", "never", "sent"}, true);
  const auto r =
      Post(s, {{"messages", Messages()}, {"stop", json::array({"STOP"})}});
  Check(r.body["choices"][0]["message"]["content"] == "abc " &&
            r.body["choices"][0]["finish_reason"] == "stop",
        "stop: non-stream " + r.body.dump());
  Check(engine.last_result().tokens.size() == 3,
        "stop: generation ends at the stop string");

  const auto events = Stream(s, {{"messages", Messages()}, {"stop", "STOP"}});
  Check(Joined(events, "content") == "abc ", "stop: stream content");
  for (const auto& e : events) {
    Check(e.dump().find("ST\"") == std::string::npos,
          "stop: no partial stop leaks: " + e.dump());
  }
}

void Utf8(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format, {});
  engine.Script({"caf", "\xC3", "\xA9", " \xE2\x82", "\xAC", "\xFF"});
  const auto r = Post(s, {{"messages", Messages()}});
  Check(r.body["choices"][0]["message"]["content"] ==
            "caf\xC3\xA9 \xE2\x82\xAC\xEF\xBF\xBD",
        "utf8: non-stream " + r.body.dump());
  const auto events = Stream(s, {{"messages", Messages()}});
  Check(Joined(events, "content") == "caf\xC3\xA9 \xE2\x82\xAC\xEF\xBF\xBD",
        "utf8: stream");
}

void ToolsAndReasoning(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format, {});
  engine.Script({"<think>hm", "m</think>", "Let me <tool_", "call>{\"name\":",
                 "\"get\",\"arguments\":{\"a\":1}}</tool_call>"});
  const json tools =
      json::array({{{"type", "function"}, {"function", {{"name", "get"}}}}});
  const auto r =
      Post(s, {{"messages", Messages()},
               {"tools", tools},
               {"chat_template_kwargs", {{"enable_thinking", true}}}});
  Check(format.last().thinking && format.last().tools.size() == 1,
        "tools: prompt gets tools and thinking");
  const auto& msg = r.body["choices"][0]["message"];
  Check(msg["reasoning_content"] == "hmm" && msg["content"] == "Let me ",
        "tools: reasoning and content " + msg.dump());
  Check(msg["tool_calls"].size() == 1 &&
            msg["tool_calls"][0]["type"] == "function" &&
            msg["tool_calls"][0]["id"].get<std::string>().rfind("call_", 0) ==
                0 &&
            msg["tool_calls"][0]["function"]["name"] == "get" &&
            msg["tool_calls"][0]["function"]["arguments"] == "{\"a\":1}",
        "tools: tool_calls " + msg.dump());
  Check(r.body["choices"][0]["finish_reason"] == "tool_calls",
        "tools: finish_reason");

  const auto events = Stream(s, {{"messages", Messages()}, {"tools", tools}});
  Check(Joined(events, "reasoning_content") == "hmm" &&
            Joined(events, "content") == "Let me ",
        "tools: stream text");
  bool saw_call = false;
  for (const auto& e : events) {
    if (e.is_object() && !e["choices"].empty() &&
        e["choices"][0]["delta"].contains("tool_calls")) {
      const auto& call = e["choices"][0]["delta"]["tool_calls"][0];
      saw_call = call["index"] == 0 && call["function"]["name"] == "get" &&
                 call["function"]["arguments"] == "{\"a\":1}";
    }
  }
  Check(saw_call, "tools: stream tool_calls delta");
  Check(events.size() >= 3 &&
            events[events.size() - 2]["choices"][0]["finish_reason"] ==
                "tool_calls",
        "tools: stream finish_reason");

  // Tool schemas reach the chat format in the client's key order.
  engine.Script({"x"});
  auto ordered = s.Client().Post(
      "/v1/chat/completions",
      R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":)"
      R"("function","function":{"name":"get","parameters":{"z":1,"a":2}}}]})",
      "application/json");
  Check(ordered && ordered->status == 200 &&
            format.last().tools[0]["function"]["parameters"].dump() ==
                R"({"z":1,"a":2})",
        "tools: key order kept");

  engine.Script({"x"});
  const auto none = Post(s, {{"messages", Messages()},
                             {"tools", tools},
                             {"tool_choice", "none"},
                             {"reasoning_effort", "high"}});
  Check(none.status == 200 && format.last().tools.empty() &&
            format.last().thinking,
        "tools: tool_choice none, reasoning_effort high");
}

void Errors(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format, {});
  auto c = s.Client();
  auto bad = c.Post("/v1/chat/completions", "{nope", "application/json");
  Check(bad && bad->status == 400 &&
            json::parse(bad->body)["error"]["type"] == "invalid_request_error",
        "errors: bad JSON");
  Check(Post(s, {{"model", "m"}}).status == 400, "errors: no messages");
  Check(Post(s, {{"messages", Messages()}, {"n", 2}}).status == 400,
        "errors: n=2");
  const auto role = Post(
      s, {{"messages", json::array({{{"role", "bad"}, {"content", "x"}}})}});
  Check(role.status == 400 && role.body["error"]["param"] == "messages",
        "errors: render refused " + role.body.dump());
  auto missing = c.Get("/v1/nothing");
  Check(
      missing && missing->status == 404 &&
          json::parse(missing->body)["error"]["message"] == "no such endpoint",
      "errors: 404");
  auto models = c.Get("/v1/models");
  Check(models && models->status == 200 &&
            json::parse(models->body)["data"][0]["id"] == "fake-model",
        "models: list");
  auto one = c.Get("/v1/models/fake-model");
  Check(one && json::parse(one->body)["object"] == "model", "models: one");
  auto other = c.Get("/v1/models/other");
  Check(other && other->status == 404, "models: unknown");
  auto health = c.Get("/health");
  Check(health && health->status == 200 &&
            json::parse(health->body)["status"] == "ok",
        "health");
}

void ApiKey(FakeEngine& engine, FakeFormat& format) {
  ServerOptions options;
  options.api_key = "sekrit";
  Running s(engine, format, options);
  engine.Script({"x"});
  const auto none = Post(s, {{"messages", Messages()}});
  Check(none.status == 401 && none.body["error"]["code"] == "invalid_api_key",
        "api key: missing");
  Check(Post(s, {{"messages", Messages()}}, {{"Authorization", "Bearer wrong"}})
                .status == 401,
        "api key: wrong");
  Check(
      Post(s, {{"messages", Messages()}}, {{"Authorization", "Bearer sekrit"}})
              .status == 200,
      "api key: right");
  auto health = s.Client().Get("/health");
  Check(health && health->status == 200, "api key: /health stays open");
}

void QueueFull(FakeEngine& engine, FakeFormat& format) {
  ServerOptions options;
  options.max_queue = 1;
  Running s(engine, format, options);
  engine.Script(std::vector<std::string>(20, "x"), true, 20ms);
  const int before = engine.generations();
  auto first = std::async(std::launch::async, [&] {
    return Post(s, {{"messages", Messages()}}).status;
  });
  Check(WaitFor([&] { return engine.running(); }), "queue: first starts");
  auto second = std::async(std::launch::async, [&] {
    return Post(s, {{"messages", Messages()}}).status;
  });
  // The second is admitted and waits; give it time to reach the queue.
  Check(WaitFor([&] {
          auto m = s.Client().Get("/metrics");
          return m &&
                 m->body.find("gufo_npu_queue_waiting 1") != std::string::npos;
        }),
        "queue: second waits");
  const auto third = Post(s, {{"messages", Messages()}});
  Check(third.status == 429 && third.body["error"]["code"] == "queue_full",
        "queue: third refused " + third.body.dump());
  Check(first.get() == 200 && second.get() == 200, "queue: both served");
  Check(engine.generations() - before == 2, "queue: two generations");
}

void Disconnect(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format, {});
  engine.Script(std::vector<std::string>(2000, "y"), true, 2ms);
  int chunks = 0;
  Stream(s, {{"messages", Messages()}}, [&](const json&) {
    return ++chunks < 4;  // hang up after a few tokens
  });
  Check(WaitFor([&] { return !engine.running(); }, 3000ms),
        "disconnect: generation ends");
  const auto r = engine.last_result();
  Check(r.tokens.size() < 1000, "disconnect: cancelled early (" +
                                    std::to_string(r.tokens.size()) +
                                    " tokens)");
  auto m = s.Client().Get("/metrics");
  Check(m && m->body.find("gufo_npu_requests_total{status=\"499\"} 1") !=
                 std::string::npos,
        "disconnect: counted as 499");

  // Non-streaming: the client gives up, the generation stops.
  {
    auto c = s.Client();
    c.set_read_timeout(0, 200000);  // 200 ms
    auto res =
        c.Post("/v1/chat/completions", json({{"messages", Messages()}}).dump(),
               "application/json");
    Check(!res, "disconnect: client timed out");
  }
  Check(WaitFor([&] { return !engine.running(); }, 3000ms) &&
            engine.last_result().tokens.size() < 1000,
        "disconnect: non-stream cancelled");
}

void Metrics(FakeEngine& engine, FakeFormat& format) {
  Running s(engine, format, {});
  engine.Script({"a", "b"});
  Post(s, {{"messages", Messages()}});
  auto m = s.Client().Get("/metrics");
  Check(m && m->status == 200, "metrics: status");
  for (const char* line :
       {"gufo_npu_requests_total{status=\"200\"} 1",
        "gufo_npu_completion_tokens_total 3", "gufo_npu_prompt_tokens_total 6",
        "gufo_npu_prompt_tokens_cached_total 3",
        "gufo_npu_time_to_first_token_seconds_count 1",
        "gufo_npu_queue_waiting 0", "gufo_npu_generation_running 0"}) {
    Check(m && m->body.find(line) != std::string::npos,
          std::string("metrics: ") + line);
  }
}

}  // namespace

int main() {
  FakeEngine engine;
  FakeFormat format;
  Basic(engine, format);
  Streaming(engine, format);
  StopStrings(engine, format);
  Utf8(engine, format);
  ToolsAndReasoning(engine, format);
  Errors(engine, format);
  ApiKey(engine, format);
  QueueFull(engine, format);
  Disconnect(engine, format);
  Metrics(engine, format);
  if (failures > 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::puts("ok: chat completions over HTTP (fake engine)");
  return 0;
}
