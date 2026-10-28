#include "serve/engine_pool.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace gufo_npu::serve {
namespace {

using Clock = std::chrono::steady_clock;

double Since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

}  // namespace

// What a request resolves to: the model's probe for info and text, the pool
// for generation.
class EnginePool::Proxy final : public Engine {
public:
  Proxy(EnginePool& pool, std::size_t index, const Engine& probe)
      : pool_(pool), index_(index), probe_(probe) {}

  const ModelInfo& info() const override { return probe_.info(); }
  const Tokenizer& tokenizer() const override { return probe_.tokenizer(); }
  std::vector<int> Encode(std::string_view text) const override {
    return probe_.Encode(text);
  }
  std::string TokenBytes(int id) const override {
    return probe_.TokenBytes(id);
  }
  int FindToken(std::string_view token) const override {
    return probe_.FindToken(token);
  }
  GenerateResult Generate(const GenerateRequest& request,
                          const std::function<bool(int)>& on_token) override {
    return pool_.Generate(index_, request, on_token);
  }

private:
  EnginePool& pool_;
  std::size_t index_;
  const Engine& probe_;
};

EnginePool::EnginePool(std::vector<EngineOptions> models, EngineFactory probe,
                       EngineFactory load, EnginePoolOptions options)
    : load_(std::move(load)), options_(std::move(options)) {
  if (models.empty()) {
    throw std::invalid_argument("no models to serve");
  }
  if (options_.unload_after_seconds < 0) {
    throw std::invalid_argument("unload_after must not be negative");
  }
  slots_.reserve(models.size());
  for (auto& model : models) {
    Slot slot;
    slot.options = std::move(model);
    slot.probe = probe(slot.options);
    const std::string& id = slot.probe->info().id;
    for (const auto& other : slots_) {
      if (other.probe->info().id == id) {
        throw std::invalid_argument("two models have the id " + id);
      }
    }
    slots_.push_back(std::move(slot));
  }
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    slots_[i].proxy = std::make_unique<Proxy>(*this, i, *slots_[i].probe);
    status_.models.push_back({slots_[i].probe->info().id});
  }
  if (!options_.preload.empty()) {
    bool found = false;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i].probe->info().id == options_.preload) {
        preload_ = i;
        found = true;
      }
    }
    if (!found) {
      throw std::invalid_argument("no model to preload with the id " +
                                  options_.preload);
    }
  }
  last_used_ = Clock::now();
  if (options_.unload_after_seconds > 0) {
    idle_thread_ = std::thread([this] { IdleLoop(); });
  }
}

EnginePool::~EnginePool() {
  {
    const std::lock_guard lock(idle_mutex_);
    stopping_ = true;
  }
  idle_wake_.notify_all();
  if (idle_thread_.joinable()) {
    idle_thread_.join();
  }
  const std::lock_guard lock(device_mutex_);
  engine_.reset();
}

void EnginePool::Preload() {
  const std::lock_guard lock(device_mutex_);
  if (loaded_ != preload_) {
    UnloadLocked();
    LoadLocked(preload_);
  }
  last_used_ = Clock::now();
}

Engine* EnginePool::Find(std::string_view id) {
  for (auto& slot : slots_) {
    if (slot.probe->info().id == id) {
      return slot.proxy.get();
    }
  }
  return nullptr;
}

Engine& EnginePool::Default() {
  const std::lock_guard lock(status_mutex_);
  if (status_.loaded) {
    for (auto& slot : slots_) {
      if (slot.probe->info().id == *status_.loaded) {
        return *slot.proxy;
      }
    }
  }
  return *slots_[preload_].proxy;
}

std::vector<Engine*> EnginePool::engines() {
  std::vector<Engine*> out;
  for (auto& slot : slots_) {
    out.push_back(slot.proxy.get());
  }
  return out;
}

EnginePoolStatus EnginePool::status() const {
  const std::lock_guard lock(status_mutex_);
  return status_;
}

GenerateResult EnginePool::Generate(std::size_t index,
                                    const GenerateRequest& request,
                                    const std::function<bool(int)>& on_token) {
  const std::lock_guard lock(device_mutex_);
  if (loaded_ != index) {
    UnloadLocked();
    LoadLocked(index);
  }
  struct Touch {
    EnginePool& pool;
    ~Touch() { pool.last_used_ = Clock::now(); }
  } touch{*this};
  return engine_->Generate(request, on_token);
}

void EnginePool::LoadLocked(std::size_t index) {
  const std::string& id = slots_[index].probe->info().id;
  {
    const std::lock_guard lock(status_mutex_);
    status_.loading = id;
  }
  const auto start = Clock::now();
  try {
    engine_ = load_(slots_[index].options);
  } catch (...) {
    const std::lock_guard lock(status_mutex_);
    status_.loading.reset();
    throw;
  }
  loaded_ = index;
  const double seconds = Since(start);
  const std::lock_guard lock(status_mutex_);
  status_.loading.reset();
  status_.loaded = id;
  status_.models[index].loads += 1;
  status_.models[index].load_seconds += seconds;
}

void EnginePool::UnloadLocked() {
  if (!engine_) {
    return;
  }
  const auto start = Clock::now();
  engine_.reset();  // releases the model's device buffers and pinned memory
  const double seconds = Since(start);
  const std::size_t index = *loaded_;
  loaded_.reset();
  const std::lock_guard lock(status_mutex_);
  status_.loaded.reset();
  status_.models[index].unloads += 1;
  status_.models[index].unload_seconds += seconds;
}

void EnginePool::IdleLoop() {
  const auto idle =
      std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(options_.unload_after_seconds));
  // Check often enough that an idle model goes within ~10% of the limit.
  const auto poll = std::max<Clock::duration>(
      idle / 10, std::chrono::milliseconds(10));
  std::unique_lock wait_lock(idle_mutex_);
  while (!stopping_) {
    idle_wake_.wait_for(wait_lock, poll);
    if (stopping_) {
      break;
    }
    // A generation holds the device; try again later rather than wait.
    std::unique_lock device(device_mutex_, std::try_to_lock);
    if (device.owns_lock() && engine_ && Clock::now() - last_used_ >= idle) {
      UnloadLocked();
    }
  }
}

std::string EnginePool::RenderMetrics() const {
  const auto s = status();
  std::string out;
  const auto family = [&](const char* name, const char* type,
                          const char* help) {
    out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name +
           " " + type + "\n";
  };
  const auto sample = [&](const char* name, const std::string& id,
                          double value) {
    char number[64];
    std::snprintf(number, sizeof number, "%.6g", value);
    out += std::string(name) + "{model=\"" + id + "\"} " + number + "\n";
  };
  family("gufo_npu_model_loaded", "gauge",
         "1 for the model on the device, 0 otherwise.");
  for (const auto& m : s.models) {
    sample("gufo_npu_model_loaded", m.id, s.loaded == m.id ? 1 : 0);
  }
  family("gufo_npu_model_loading", "gauge", "1 while the model loads.");
  for (const auto& m : s.models) {
    sample("gufo_npu_model_loading", m.id, s.loading == m.id ? 1 : 0);
  }
  family("gufo_npu_model_loads_total", "counter", "Model loads.");
  for (const auto& m : s.models) {
    sample("gufo_npu_model_loads_total", m.id, m.loads);
  }
  family("gufo_npu_model_load_seconds_total", "counter",
         "Seconds spent loading models.");
  for (const auto& m : s.models) {
    sample("gufo_npu_model_load_seconds_total", m.id, m.load_seconds);
  }
  family("gufo_npu_model_unloads_total", "counter", "Model unloads.");
  for (const auto& m : s.models) {
    sample("gufo_npu_model_unloads_total", m.id, m.unloads);
  }
  family("gufo_npu_model_unload_seconds_total", "counter",
         "Seconds spent unloading models.");
  for (const auto& m : s.models) {
    sample("gufo_npu_model_unload_seconds_total", m.id, m.unload_seconds);
  }
  return out;
}

}  // namespace gufo_npu::serve
