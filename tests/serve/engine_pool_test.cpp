// Several models behind one server (src/serve/engine_pool.hpp), with
// scripted engines: routing by `model`, switching with at most one model
// loaded, load failures, idle unload, FIFO order across models, and the
// /v1/models, /health and /metrics views. No model, no NPU.
#include "serve/engine_pool.hpp"

#include <httplib/httplib.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "fakes.hpp"
#include "serve/server.hpp"

namespace {

using gufo_npu::serve::Engine;
using gufo_npu::serve::EngineOptions;
using gufo_npu::serve::EnginePool;
using gufo_npu::serve::EnginePoolOptions;
using gufo_npu::serve::GenerateRequest;
using gufo_npu::serve::GenerateResult;
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

// Loaded engines alive now, the most ever alive, and the generation order.
std::atomic<int> live{0};
std::atomic<int> peak{0};
std::mutex order_mutex;
std::vector<std::string> order;

// A loaded model: replies with its own id; counts itself while alive.
class LoadedEngine : public FakeEngine {
public:
  LoadedEngine(const std::string& id, std::chrono::milliseconds delay) {
    info_.id = id;
    Script({"from-" + id}, true, delay);
    const int now = ++live;
    int seen = peak.load();
    while (now > seen && !peak.compare_exchange_weak(seen, now)) {
    }
  }
  ~LoadedEngine() override { --live; }

  GenerateResult Generate(const GenerateRequest& request,
                          const std::function<bool(int)>& on_token) override {
    {
      const std::lock_guard lock(order_mutex);
      order.push_back(info_.id);
    }
    return FakeEngine::Generate(request, on_token);
  }
};

// The model id is the GGUF path; "bad" fails to load. The probe decodes
// the same tokens as the loaded model (one tokenizer per GGUF).
std::unique_ptr<Engine> Probe(const EngineOptions& options) {
  auto engine = std::make_unique<FakeEngine>();
  engine->info_.id = options.model_path;
  engine->Script({"from-" + options.model_path});
  return engine;
}

std::chrono::milliseconds load_delay{0};
std::chrono::milliseconds token_delay{0};

std::unique_ptr<Engine> Load(const EngineOptions& options) {
  Check(live == 0, "load: another model is still loaded");
  if (options.model_path == "bad") {
    throw std::runtime_error("cannot load bad");
  }
  std::this_thread::sleep_for(load_delay);
  return std::make_unique<LoadedEngine>(options.model_path, token_delay);
}

std::vector<EngineOptions> Models(std::vector<std::string> ids) {
  std::vector<EngineOptions> out;
  for (auto& id : ids) {
    out.emplace_back();
    out.back().model_path = std::move(id);
  }
  return out;
}

GenerateResult Run(Engine& engine) {
  GenerateRequest request;
  request.prompt = {0, 0, 0};
  request.max_tokens = 8;
  request.stop_ids = {gufo_npu::serve::testing::kStopId};
  return engine.Generate(request, [](int) { return true; });
}

template<typename F>
bool Throws(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void Construction() {
  Check(Throws([] { EnginePool pool({}, Probe, Load); }),
        "empty pool refused");
  Check(Throws([] { EnginePool pool(Models({"a", "a"}), Probe, Load); }),
        "duplicate ids refused");
  Check(Throws([] {
          EnginePool pool(Models({"a", "b"}), Probe, Load, {.preload = "c"});
        }),
        "unknown preload refused");
  Check(Throws([] {
          EnginePool pool(Models({"a"}), Probe, Load,
                          {.unload_after_seconds = -1});
        }),
        "negative unload_after refused");
}

void Switching() {
  peak = 0;
  EnginePool pool(Models({"a", "b", "c"}), Probe, Load, {.preload = "b"});
  Check(!pool.status().loaded, "nothing loaded before Preload");
  Check(live == 0, "probes are not loaded engines");
  pool.Preload();
  Check(pool.status().loaded == "b", "preload loads b");
  Check(&pool.Default() == pool.Find("b"), "default is the loaded model");
  Check(pool.Find("z") == nullptr, "unknown id");
  Check(pool.Find("a")->info().id == "a", "proxy info before load");

  const char* sequence[] = {"a", "c", "a", "a", "b", "c", "b"};
  for (const char* id : sequence) {
    const auto result = Run(*pool.Find(id));
    Check(pool.status().loaded == std::string(id),
          std::string("loaded after a request to ") + id);
    Check(result.tokens.size() == 2, std::string("reply from ") + id);
  }
  Check(peak == 1, "at most one model loaded at a time");
  const auto s = pool.status();
  // Loads: b (preload), a, c, a, b, c, b; the repeated a needs none.
  const long loads = s.models[0].loads + s.models[1].loads + s.models[2].loads;
  const long unloads =
      s.models[0].unloads + s.models[1].unloads + s.models[2].unloads;
  Check(loads == 7, "loads " + std::to_string(loads));
  Check(unloads == 6, "unloads " + std::to_string(unloads));
  Check(s.models[0].id == "a" && s.models[0].loads == 2, "a loaded twice");
  const std::string metrics = pool.RenderMetrics();
  Check(metrics.find("gufo_npu_model_loaded{model=\"b\"} 1") !=
            std::string::npos,
        "metrics: b loaded");
  Check(metrics.find("gufo_npu_model_loads_total{model=\"a\"} 2") !=
            std::string::npos,
        "metrics: a loads");
  Check(metrics.find("gufo_npu_model_unloads_total{model=\"c\"} 2") !=
            std::string::npos,
        "metrics: c unloads");
}

void LoadFailure() {
  EnginePool pool(Models({"a", "bad"}), Probe, Load);
  pool.Preload();
  bool threw = false;
  try {
    Run(*pool.Find("bad"));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "a failed load reaches the caller");
  const auto s = pool.status();
  Check(!s.loaded && !s.loading, "nothing loaded after a failed load");
  Check(live == 0, "the old model was released before the failed load");
  Run(*pool.Find("a"));
  Check(pool.status().loaded == "a", "the next request loads again");
}

void IdleUnload() {
  EnginePool pool(Models({"a", "b"}), Probe, Load,
                  {.unload_after_seconds = 0.2});
  pool.Preload();
  Run(*pool.Find("b"));
  Check(pool.status().loaded == "b", "b loaded");
  std::this_thread::sleep_for(100ms);
  Check(pool.status().loaded == "b", "not unloaded before the limit");
  std::this_thread::sleep_for(400ms);
  Check(!pool.status().loaded, "unloaded after idling");
  Check(live == 0, "idle unload released the engine");
  Check(pool.status().models[1].unloads == 1, "idle unload counted");
  Run(*pool.Find("b"));
  Check(pool.status().loaded == "b", "reloaded on the next request");
}

// A server on a free localhost port over the pool.
struct Running {
  Running(EnginePool& pool, const FakeFormat& format)
      : server(pool, format, [] {
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

struct Reply {
  int status = 0;
  json body;
};

Reply Chat(const Running& s, const json& model) {
  json body = {{"messages", {{{"role", "user"}, {"content", "hi"}}}}};
  if (!model.is_null()) {
    body["model"] = model;
  }
  auto res = s.Client().Post("/v1/chat/completions", body.dump(),
                             "application/json");
  if (!res) {
    return {};
  }
  return {res->status, json::parse(res->body, nullptr, false)};
}

json Get(const Running& s, const char* path) {
  auto res = s.Client().Get(path);
  Check(res && res->status == 200, std::string("GET ") + path);
  if (!res) {
    return json();
  }
  const std::string ct = res->get_header_value("Content-Type");
  return ct.rfind("application/json", 0) == 0 ? json::parse(res->body)
                                              : json(res->body);
}

std::string Content(const Reply& r) {
  if (!r.body.is_object() || !r.body.contains("choices")) {
    return "";
  }
  return r.body["choices"][0]["message"].value("content", "");
}

void Http() {
  FakeFormat format;
  EnginePool pool(Models({"a", "b", "c"}), Probe, Load);
  pool.Preload();
  Running s(pool, format);

  const json models = Get(s, "/v1/models");
  Check(models["data"].size() == 3, "/v1/models lists three");
  Check(models["data"][0]["id"] == "a" && models["data"][0]["loaded"] == true &&
            models["data"][1]["loaded"] == false,
        "/v1/models marks the loaded model");
  Check(Get(s, "/v1/models/c")["id"] == "c", "/v1/models/c");
  Check(s.Client().Get("/v1/models/z")->status == 404, "/v1/models/z 404");

  auto r = Chat(s, "b");
  Check(r.status == 200 && r.body["model"] == "b" &&
            Content(r) == "from-b",
        "chat with b: " + r.body.dump());
  Check(Get(s, "/health")["model"] == "b", "/health: b loaded");
  Check(Get(s, "/health")["loading"].is_null(), "/health: no load running");

  r = Chat(s, json());
  Check(r.status == 200 && r.body["model"] == "b",
        "no model: the loaded one");
  r = Chat(s, "zzz");
  Check(r.status == 404 && r.body["error"]["code"] == "model_not_found" &&
            r.body["error"]["param"] == "model",
        "unknown model: 404 model_not_found");
  r = Chat(s, 7);
  Check(r.status == 400, "non-string model: 400");

  r = Chat(s, "c");
  Check(r.status == 200 && Content(r) == "from-c", "chat with c");
  const std::string metrics = Get(s, "/metrics").get<std::string>();
  Check(metrics.find("gufo_npu_model_loaded{model=\"c\"} 1") !=
            std::string::npos,
        "/metrics: c loaded");
  Check(metrics.find("gufo_npu_model_loads_total{model=\"b\"} 1") !=
            std::string::npos,
        "/metrics: b loads");
  Check(metrics.find("gufo_npu_queue_waiting") != std::string::npos,
        "/metrics keeps the server's counters");
}

// Requests run in arrival order whatever model they name.
void Fifo() {
  FakeFormat format;
  token_delay = 150ms;
  load_delay = 50ms;
  EnginePool pool(Models({"a", "b"}), Probe, Load);
  pool.Preload();
  Running s(pool, format);
  {
    const std::lock_guard lock(order_mutex);
    order.clear();
  }
  const char* sequence[] = {"b", "a", "b", "a"};
  std::vector<std::thread> clients;
  std::vector<Reply> replies(4);
  for (int i = 0; i < 4; ++i) {
    clients.emplace_back(
        [&, i] { replies[static_cast<std::size_t>(i)] = Chat(s, sequence[i]); });
    std::this_thread::sleep_for(60ms);  // arrive in order
  }
  for (auto& t : clients) {
    t.join();
  }
  for (int i = 0; i < 4; ++i) {
    Check(replies[static_cast<std::size_t>(i)].status == 200 &&
              Content(replies[static_cast<std::size_t>(i)]) ==
                  std::string("from-") + sequence[i],
          "fifo reply " + std::to_string(i));
  }
  const std::lock_guard lock(order_mutex);
  Check(order == std::vector<std::string>{"b", "a", "b", "a"},
        "generations run in arrival order");
  Check(peak == 1, "at most one model loaded while switching under load");
  token_delay = 0ms;
  load_delay = 0ms;
}

// A pool of one (e.g. for --unload-after) takes any model name.
void SingleModelPool() {
  FakeFormat format;
  EnginePool pool(Models({"a"}), Probe, Load, {.unload_after_seconds = 60});
  pool.Preload();
  Running s(pool, format);
  const auto r = Chat(s, "gpt-4o");
  Check(r.status == 200 && r.body["model"] == "a",
        "one model: any name is it");
}

}  // namespace

int main() {
  Construction();
  Switching();
  LoadFailure();
  IdleUnload();
  Http();
  Fifo();
  SingleModelPool();
  Check(live == 0, "every loaded engine destroyed");
  if (failures > 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("engine pool: ok\n");
  return 0;
}
