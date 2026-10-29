// The server's log lines (docs/SERVING.md, "Logs"): gufo's line format,
// levels, one received and one completed line per request with its
// generation keys, outcomes for errors and hang-ups, the pool's load and
// switch events, and that keys and prompt text never reach the log.
#include "serve/logging.hpp"

#include <httplib/httplib.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <regex>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#include "fakes.hpp"
#include "serve/engine_pool.hpp"
#include "serve/server.hpp"

namespace {

using gufo_npu::serve::Engine;
using gufo_npu::serve::EngineOptions;
using gufo_npu::serve::EnginePool;
using gufo_npu::serve::Logger;
using gufo_npu::serve::LogLevel;
using gufo_npu::serve::Server;
using gufo_npu::serve::ServerOptions;
using gufo_npu::serve::testing::FakeEngine;
using gufo_npu::serve::testing::FakeFormat;
using nlohmann::json;
using namespace std::chrono_literals;

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

// std::clog captured, safe to read while server threads log.
class Capture : public std::streambuf {
public:
  Capture() : old_(std::clog.rdbuf(this)) {}
  ~Capture() override { std::clog.rdbuf(old_); }

  std::string text() const {
    std::lock_guard lock(mutex_);
    return text_;
  }
  void Clear() {
    std::lock_guard lock(mutex_);
    text_.clear();
  }
  // The lines containing every one of `parts`.
  std::vector<std::string> Lines(const std::vector<std::string>& parts) const {
    std::vector<std::string> out;
    const std::string all = text();
    std::size_t start = 0;
    while (start < all.size()) {
      std::size_t end = all.find('\n', start);
      if (end == std::string::npos) {
        end = all.size();
      }
      const std::string line = all.substr(start, end - start);
      bool match = true;
      for (const auto& part : parts) {
        match = match && line.find(part) != std::string::npos;
      }
      if (match) {
        out.push_back(line);
      }
      start = end + 1;
    }
    return out;
  }
  // Waits for a line with every one of `parts` (the completed line is
  // written after the reply reaches the client).
  std::string WaitLine(const std::vector<std::string>& parts,
                       std::chrono::milliseconds timeout = 3000ms) const {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (auto lines = Lines(parts); !lines.empty()) {
        return lines.front();
      }
      std::this_thread::sleep_for(5ms);
    }
    return {};
  }

protected:
  int_type overflow(int_type c) override {
    if (c != traits_type::eof()) {
      std::lock_guard lock(mutex_);
      text_ += static_cast<char>(c);
    }
    return c;
  }
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    std::lock_guard lock(mutex_);
    text_.append(s, static_cast<std::size_t>(n));
    return n;
  }

private:
  std::streambuf* old_;
  mutable std::mutex mutex_;
  std::string text_;
};

template<typename Backend>
struct Running {
  Running(Backend& backend, const FakeFormat& format, ServerOptions o = {})
      : server(backend, format, [&] {
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

json Messages(const std::string& text = "hello") {
  return json::array({{{"role", "user"}, {"content", text}}});
}

// The value of `key=` in a line ("" if absent).
std::string Value(const std::string& line, const std::string& key) {
  const std::string tag = " " + key + "=";
  const auto at = line.find(tag);
  if (at == std::string::npos) {
    return {};
  }
  const auto start = at + tag.size();
  return line.substr(start, line.find(' ', start) - start);
}

void Format(Capture& log) {
  log.Clear();
  Logger::Info("server", "event=test a=1");
  const std::regex line(
      R"(^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \[INFO\] \[server\] event=test a=1\n$)");
  Check(std::regex_match(log.text(), line), "format: " + log.text());

  log.Clear();
  Logger::Warn("x", "a\nb");
  Check(log.text().find("a\\x0ab") != std::string::npos,
        "format: control bytes escaped");

  log.Clear();
  Logger::SetLevel(LogLevel::kWarn);
  Logger::Info("x", "dropped");
  Logger::LogRequest("r1", "GET", "/v1/models", 200, 1.0);
  Logger::LogRequest("r2", "GET", "/nope", 404, 1.0);
  Logger::LogRequest("r3", "POST", "/v1/chat/completions", 200, 1.0, "",
                     "disconnected");
  Logger::LogRequest("r4", "POST", "/v1/chat/completions", 500, 1.0);
  Logger::SetLevel(LogLevel::kInfo);
  const std::string text = log.text();
  Check(text.find("dropped") == std::string::npos &&
            text.find("request=r1 ") == std::string::npos,
        "levels: info dropped at warn");
  Check(!log.Lines({"[WARN]", "request=r2", "status=404"}).empty() &&
            !log.Lines({"[WARN]", "request=r3", "outcome=disconnected"})
                 .empty() &&
            !log.Lines({"[ERROR]", "request=r4", "status=500"}).empty(),
        "levels: 4xx and hang-ups warn, 5xx errors: " + text);
  Check(!log.Lines({"request=r4", "rss_mib=", "host_available_mib="}).empty(),
        "request: memory status");
  Check(gufo_npu::serve::LogLevelFromName("debug") == LogLevel::kDebug &&
            !gufo_npu::serve::LogLevelFromName("verbose"),
        "levels: names");
}

void Requests(Capture& log) {
  FakeEngine engine;
  FakeFormat format;
  ServerOptions options;
  options.api_key = "sk-very-secret";
  Running s(engine, format, options);
  httplib::Headers auth = {{"Authorization", "Bearer sk-very-secret"}};

  // Non-streaming chat: a received and a completed line with one id.
  log.Clear();
  engine.Script({"a", "b"});
  auto res =
      s.Client().Post("/v1/chat/completions", auth,
                      json({{"messages", Messages("my secret prompt")}}).dump(),
                      "application/json");
  Check(res && res->status == 200, "chat: served");
  const std::string done = log.WaitLine({"event=completed", "kind=chat"});
  Check(!done.empty(), "chat: completed line: " + log.text());
  const std::string id =
      Value(" " + done.substr(done.find("request=")), "request");
  const auto received =
      log.Lines({"[INFO] [http] request=" + id + " event=received",
                 "method=POST", "path=/v1/chat/completions", "body_bytes="});
  Check(!id.empty() && received.size() == 1, "chat: received line, id " + id);
  Check(done.find("[INFO] [http]") != std::string::npos &&
            Value(done, "status") == "200" &&
            Value(done, "outcome") == "completed" &&
            Value(done, "model") == "fake-model" &&
            Value(done, "stream") == "0" && Value(done, "n") == "1" &&
            Value(done, "generated_tokens") == "3" &&
            Value(done, "finish") == "stop" && Value(done, "cache") == "hit" &&
            Value(done, "cached_tokens") == "3" &&
            Value(done, "draft") == "none" &&
            !Value(done, "queue_ms").empty() &&
            !Value(done, "ttft_ms").empty() &&
            !Value(done, "prefill_tps").empty() &&
            !Value(done, "decode_tps").empty() &&
            !Value(done, "duration_ms").empty() &&
            !Value(done, "rss_mib").empty(),
        "chat: generation keys: " + done);
  Check(std::stoi(Value(done, "prefill_tokens")) ==
            std::stoi(Value(done, "prompt_tokens")) - 3,
        "chat: prefill_tokens = prompt - cached: " + done);

  // Streaming, with n and a response format.
  log.Clear();
  engine.Script({"{}"});
  res = s.Client().Post("/v1/chat/completions", auth,
                        json({{"messages", Messages()},
                              {"stream", true},
                              {"n", 2},
                              {"response_format", {{"type", "json_object"}}}})
                            .dump(),
                        "application/json");
  Check(res && res->status == 200, "stream: served");
  const std::string streamed =
      log.WaitLine({"event=completed", "kind=chat", "stream=1"});
  Check(Value(streamed, "outcome") == "completed" &&
            Value(streamed, "n") == "2" &&
            Value(streamed, "finish") == "stop,stop" &&
            Value(streamed, "response_format") == "json_object",
        "stream: keys: " + streamed);

  // Errors: bad JSON (400) and a missing key (401) warn.
  log.Clear();
  res = s.Client().Post("/v1/chat/completions", auth, "{not json",
                        "application/json");
  const std::string bad = log.WaitLine({"event=completed", "status=400"});
  Check(bad.find("[WARN]") != std::string::npos &&
            Value(bad, "error_type") == "invalid_request_error",
        "error: 400 line: " + bad);
  res = s.Client().Post("/v1/chat/completions",
                        json({{"messages", Messages()}}).dump(),
                        "application/json");
  Check(!log.WaitLine({"[WARN]", "event=completed", "status=401",
                       "error_code=invalid_api_key"})
             .empty(),
        "error: 401 line");

  // A streaming client that hangs up: outcome disconnected, at WARN.
  log.Clear();
  engine.Script(std::vector<std::string>(2000, "y"), true, 2ms);
  {
    auto c = s.Client();
    int chunks = 0;
    httplib::Request req;
    req.method = "POST";
    req.path = "/v1/chat/completions";
    req.headers = auth;
    req.set_header("Content-Type", "application/json");
    req.body = json({{"messages", Messages()}, {"stream", true}}).dump();
    req.content_receiver = [&](const char*, std::size_t, std::uint64_t,
                               std::uint64_t) { return ++chunks < 4; };
    c.send(req);
  }
  const std::string gone =
      log.WaitLine({"event=completed", "outcome=disconnected"});
  Check(gone.find("[WARN]") != std::string::npos &&
            Value(gone, "finish") == "cancelled",
        "disconnect: line: " + gone);

  // /health logs only at debug.
  log.Clear();
  s.Client().Get("/health");
  std::this_thread::sleep_for(100ms);
  Check(log.Lines({"path=/health"}).empty(), "health: quiet at info");
  Logger::SetLevel(LogLevel::kDebug);
  s.Client().Get("/health");
  Check(!log.WaitLine({"[DEBUG]", "event=completed", "path=/health"}).empty(),
        "health: logged at debug");
  Logger::SetLevel(LogLevel::kInfo);

  // Nothing secret: the API key and prompt text never reach the log.
  const std::string all = log.text();
  Check(all.find("sk-very-secret") == std::string::npos, "secrets: no API key");
  Check(all.find("secret prompt") == std::string::npos,
        "secrets: no prompt text");
}

std::unique_ptr<Engine> Probe(const EngineOptions& options) {
  auto engine = std::make_unique<FakeEngine>();
  engine->info_.id = options.model_path;
  engine->Script({"x"});
  return engine;
}

std::unique_ptr<Engine> Load(const EngineOptions& options) {
  if (options.model_path == "bad") {
    throw std::runtime_error("cannot load bad");
  }
  auto engine = std::make_unique<FakeEngine>();
  engine->info_.id = options.model_path;
  engine->Script({"from-" + options.model_path});
  return engine;
}

void Pool(Capture& log) {
  std::vector<EngineOptions> models(3);
  models[0].model_path = "a";
  models[1].model_path = "b";
  models[2].model_path = "bad";
  EnginePool pool(std::move(models), Probe, Load);
  log.Clear();
  pool.Preload();
  Check(!log.Lines({"[INFO] [loader] event=load_started model=a"}).empty() &&
            !log.Lines(
                    {"event=load_completed model=a", "elapsed_ms=", "rss_mib="})
                 .empty(),
        "pool: preload logged: " + log.text());

  FakeFormat format;
  Running s(pool, format);
  log.Clear();
  auto res =
      s.Client().Post("/v1/chat/completions",
                      json({{"model", "b"}, {"messages", Messages()}}).dump(),
                      "application/json");
  Check(res && res->status == 200, "pool: served by b");
  const std::string done = log.WaitLine({"event=completed", "kind=chat"});
  Check(!log.Lines({"[loader] event=switch from=a to=b"}).empty() &&
            !log.Lines({"event=unload_completed model=a", "elapsed_ms="})
                 .empty() &&
            !log.Lines({"event=load_completed model=b"}).empty() &&
            Value(done, "model") == "b",
        "pool: switch logged: " + log.text());

  log.Clear();
  res =
      s.Client().Post("/v1/chat/completions",
                      json({{"model", "bad"}, {"messages", Messages()}}).dump(),
                      "application/json");
  Check(res && res->status == 500, "pool: failed load is a 500");
  Check(!log.WaitLine({"[ERROR] [loader] event=load_failed model=bad",
                       "reason=cannot load bad"})
                .empty() &&
            !log.WaitLine({"[ERROR] [http]", "event=completed", "status=500"})
                 .empty(),
        "pool: load failure logged: " + log.text());
}

}  // namespace

int main() {
  setenv("NO_COLOR", "1", 1);
  Capture log;
  Format(log);
  Requests(log);
  Pool(log);
  if (failures > 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("ok: serve logs\n");
  return 0;
}
