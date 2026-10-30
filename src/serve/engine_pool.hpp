#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "serve/engine.hpp"

// Several models behind one server, one of them on the NPU at a time.
//
// Each model has a proxy Engine that clients' requests resolve to by id. The
// proxy answers info() and the text methods (EmbeddingIds too) from a probe
// (the model's GGUF metadata and tokenizer, no weights), so requests are
// tokenized and checked before the model loads. Its Generate (or Embed, for
// an embedding model) makes that model the loaded one: the current engine is
// destroyed first (its device buffers and pinned memory released), then the
// requested one is loaded, so at most one model's weights are resident. The
// server already runs one generation at a time, so a switch happens inside a
// request's turn and waiting requests stay in arrival order (docs/SERVING.md,
// "Several models").
namespace gufo_npu::serve {

using EngineFactory =
    std::function<std::unique_ptr<Engine>(const EngineOptions& options)>;

struct EnginePoolOptions {
  // The model loaded at start (its id); empty: the first model.
  std::string preload;
  // Unload the loaded model after this many idle seconds; 0: never.
  double unload_after_seconds = 0;
};

// Load and unload counters for /metrics and /health.
struct EnginePoolStatus {
  std::optional<std::string> loaded;   // the model on the device
  std::optional<std::string> loading;  // a load in progress
  struct Model {
    std::string id;
    long loads = 0;
    long unloads = 0;
    double load_seconds = 0;
    double unload_seconds = 0;
  };
  std::vector<Model> models;  // in registration order
};

class EnginePool {
public:
  // `probe` builds what clients see before a model loads (an Engine whose
  // info() and text methods work; its Generate is never called); `load`
  // builds the model on the device. Model ids must be distinct. Throws
  // std::invalid_argument for an empty list, duplicate ids or an unknown
  // preload id.
  EnginePool(std::vector<EngineOptions> models, EngineFactory probe,
             EngineFactory load, EnginePoolOptions options = {});
  ~EnginePool();
  EnginePool(const EnginePool&) = delete;
  EnginePool& operator=(const EnginePool&) = delete;

  // Loads the preload model (blocking). Call once before serving.
  void Preload();

  // The model's proxy engine, or nullptr for an unknown id.
  Engine* Find(std::string_view id);
  // The proxy for requests that name no model: the loaded model, else the
  // preload model.
  Engine& Default();
  // All proxies, in registration order.
  std::vector<Engine*> engines();

  EnginePoolStatus status() const;
  // Prometheus text for the pool's counters (appended to /metrics).
  std::string RenderMetrics() const;

private:
  class Proxy;
  friend class Proxy;

  GenerateResult Generate(std::size_t index, const GenerateRequest& request,
                          const std::function<bool(int)>& on_token);
  std::vector<std::vector<float>> Embed(
      std::size_t index, const std::vector<std::vector<int>>& inputs);
  // Under device_mutex_: makes model `index` the loaded one, logging a
  // switch from another.
  void SwitchLocked(std::size_t index);
  // Under device_mutex_: makes model `index` the loaded one.
  void LoadLocked(std::size_t index);
  // Under device_mutex_: destroys the loaded engine, if any.
  void UnloadLocked();
  void IdleLoop();

  struct Slot {
    EngineOptions options;
    std::unique_ptr<Engine> probe;
    std::unique_ptr<Proxy> proxy;
  };

  EngineFactory load_;
  EnginePoolOptions options_;
  std::vector<Slot> slots_;
  std::size_t preload_ = 0;

  // Held while a model loads, unloads or generates.
  std::mutex device_mutex_;
  std::unique_ptr<Engine> engine_;  // the loaded model, under device_mutex_
  std::optional<std::size_t> loaded_;
  std::chrono::steady_clock::time_point last_used_;

  // Status for /health and /metrics, readable during a long load.
  mutable std::mutex status_mutex_;
  EnginePoolStatus status_;

  std::mutex idle_mutex_;
  std::condition_variable idle_wake_;
  bool stopping_ = false;
  std::thread idle_thread_;
};

// The model's info and tokenizer from its GGUF, without loading weights
// (EngineFactory for EnginePool's probe). Generate throws.
std::unique_ptr<Engine> ProbeEngine(const EngineOptions& options);

}  // namespace gufo_npu::serve
