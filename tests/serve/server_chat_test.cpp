// The server end to end with the real Qwen3.5 chat format over a scripted
// engine (tests/serve/fakes.hpp): reasoning and tool calls come back as
// OpenAI message parts, streaming and not, and a tool result round trip
// renders the prompt the chat format renders for the same messages.
#include <httplib/httplib.h>

#include <cstdio>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "fakes.hpp"
#include "serve/chat_format.hpp"
#include "serve/server.hpp"

namespace {

using gufo_npu::serve::ChatFormat;
using gufo_npu::serve::ChatPrompt;
using gufo_npu::serve::Server;
using gufo_npu::serve::ServerOptions;
using gufo_npu::serve::testing::FakeEngine;
using nlohmann::json;
using nlohmann::ordered_json;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) {
    ++failures;
  }
}

struct Running {
  Running(FakeEngine& engine, const ChatFormat& format)
      : server(engine, format, [] {
          ServerOptions options;
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

json Post(const Running& s, const ordered_json& body) {
  auto res =
      s.Client().Post("/v1/chat/completions", body.dump(), "application/json");
  if (!res || res->status != 200) {
    std::fprintf(stderr, "status %d: %s\n", res ? res->status : -1,
                 res ? res->body.c_str() : "");
    return json();
  }
  return json::parse(res->body, nullptr, false);
}

// The `data:` events of a streamed request, without the final [DONE].
std::vector<json> Stream(const Running& s, ordered_json body) {
  body["stream"] = true;
  auto res =
      s.Client().Post("/v1/chat/completions", body.dump(), "application/json");
  std::vector<json> events;
  if (!res || res->status != 200) {
    return events;
  }
  std::size_t at = 0;
  const std::string& text = res->body;
  while ((at = text.find("data: ", at)) != std::string::npos) {
    const auto end = text.find("\n\n", at);
    const auto data = text.substr(at + 6, end - at - 6);
    at = end;
    if (data != "[DONE]") {
      events.push_back(json::parse(data, nullptr, false));
    }
  }
  return events;
}

ordered_json WeatherTool() {
  return ordered_json::parse(R"({
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "days": {"type": "integer"}
        },
        "required": ["city"]
      }
    }
  })");
}

// A tool call in Qwen3.5's markup, split across tokens at awkward places.
std::vector<std::string> ToolCallPieces() {
  return {"I'll check.\n\n<tool", "_call>\n<function=get_",
          "weather>\n<parameter=city>\nPar",
          "is\n</parameter>\n<parameter=days>\n3\n</para",
          "meter>\n</function>\n</tool_call>"};
}

void TestReasoning(FakeEngine& engine, const Running& s) {
  engine.Script({"The sky", " scatters blue light.", "\n</think>\n\n",
                 "Rayleigh ", "scattering."});
  ordered_json body = {
      {"messages", {{{"role", "user"}, {"content", "Why is the sky blue?"}}}},
      {"chat_template_kwargs", {{"enable_thinking", true}}}};
  const auto r = Post(s, body);
  const auto& message = r["choices"][0]["message"];
  Check(
      message.value("reasoning_content", "") == "The sky scatters blue light.",
      "reasoning: reasoning_content");
  Check(message.value("content", "") == "Rayleigh scattering.",
        "reasoning: content");

  std::string reasoning;
  std::string content;
  for (const auto& e : Stream(s, body)) {
    const auto& delta = e["choices"][0]["delta"];
    if (delta.contains("reasoning_content")) {
      reasoning += delta["reasoning_content"].get<std::string>();
    }
    if (delta.contains("content") && delta["content"].is_string()) {
      content += delta["content"].get<std::string>();
    }
  }
  Check(reasoning == "The sky scatters blue light." &&
            content == "Rayleigh scattering.",
        "reasoning: streamed deltas");
}

void TestToolRoundTrip(FakeEngine& engine, const Running& s,
                       const ChatFormat& format) {
  ordered_json messages = {
      {{"role", "user"}, {"content", "Weather in Paris for 3 days?"}}};
  ordered_json tools = ordered_json::array({WeatherTool()});

  engine.Script(ToolCallPieces());
  const auto r = Post(s, {{"messages", messages}, {"tools", tools}});
  const auto& choice = r["choices"][0];
  const auto& calls = choice["message"]["tool_calls"];
  Check(choice.value("finish_reason", "") == "tool_calls",
        "tool call: finish_reason tool_calls");
  Check(calls.is_array() && calls.size() == 1 &&
            calls[0]["function"]["name"] == "get_weather",
        "tool call: one get_weather call");
  const auto args =
      calls.is_array() && !calls.empty()
          ? json::parse(calls[0]["function"]["arguments"].get<std::string>(),
                        nullptr, false)
          : json();
  Check(args == json({{"city", "Paris"}, {"days", 3}}),
        "tool call: arguments typed by the schema");
  Check(choice["message"].value("content", "") == "I'll check.",
        "tool call: the text before it is content");

  // Streaming: the same call arrives as tool_calls deltas.
  std::string name;
  std::string arguments;
  std::string finish;
  for (const auto& e : Stream(s, {{"messages", messages}, {"tools", tools}})) {
    const auto& c = e["choices"];
    if (!c.is_array() || c.empty()) {
      continue;
    }
    const auto& delta = c[0]["delta"];
    if (delta.contains("tool_calls")) {
      const auto& f = delta["tool_calls"][0]["function"];
      name += f.value("name", "");
      arguments += f.value("arguments", "");
    }
    if (c[0]["finish_reason"].is_string()) {
      finish = c[0]["finish_reason"];
    }
  }
  Check(name == "get_weather" &&
            json::parse(arguments, nullptr, false) == args &&
            finish == "tool_calls",
        "tool call: streamed deltas");

  // The client runs the tool and sends the result back.
  ordered_json assistant = {{"role", "assistant"},
                            {"content", "I'll check."},
                            {"tool_calls", ordered_json::array()}};
  assistant["tool_calls"].push_back(
      {{"id", calls.is_array() && !calls.empty() ? calls[0]["id"] : "call_x"},
       {"type", "function"},
       {"function",
        {{"name", "get_weather"},
         {"arguments", R"({"city": "Paris", "days": 3})"}}}});
  messages.push_back(assistant);
  messages.push_back({{"role", "tool"},
                      {"tool_call_id", assistant["tool_calls"][0]["id"]},
                      {"content", "Sunny, 21 C."}});
  engine.Script({"Sunny", " and 21 C."});
  const auto final = Post(s, {{"messages", messages}, {"tools", tools}});
  Check(final["choices"][0]["message"].value("content", "") ==
                "Sunny and 21 C." &&
            final["choices"][0].value("finish_reason", "") == "stop",
        "tool result: final answer");

  ChatPrompt prompt;
  prompt.messages = messages;
  prompt.tools = tools;
  const std::string rendered = format.Render(prompt);
  Check(engine.last().prompt.size() == rendered.size() &&
            rendered.find("<tool_response>\nSunny, 21 C.\n</tool_response>") !=
                std::string::npos,
        "tool result: the engine got the chat format's prompt");
}

}  // namespace

int main() {
  FakeEngine engine;
  const auto format = gufo_npu::serve::NewQwen35ChatFormat();
  Running s(engine, *format);
  TestReasoning(engine, s);
  TestToolRoundTrip(engine, s, *format);
  return failures == 0 ? 0 : 1;
}
