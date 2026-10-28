#include "serve/server.hpp"

#include <httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <ctime>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "serve/engine_pool.hpp"
#include "serve/metrics.hpp"
#include "serve/openai.hpp"
#include "serve/reply.hpp"
#include "serve/scheduler.hpp"

namespace gufo_npu::serve {
namespace {

using nlohmann::json;
using Clock = std::chrono::steady_clock;

// Status recorded for a client that left before its reply was complete.
constexpr int kClientClosed = 499;

double Since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string NewId(const char* prefix) {
  thread_local std::mt19937_64 rng(std::random_device{}());
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id = prefix;
  for (int i = 0; i < 24; ++i) {
    id += kHex[rng() & 15];
  }
  return id;
}

// JSON text that never throws on bad UTF-8 (it should not occur).
std::string Dump(const json& j) {
  return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

void SendError(httplib::Response& res, const ApiError& error) {
  res.status = error.status;
  res.set_content(Dump(ErrorJson(error)), "application/json");
}

json ToolCall(const OutputPart& part) {
  return {
      {"id", part.tool_id.empty() ? NewId("call_") : part.tool_id},
      {"type", "function"},
      {"function", {{"name", part.tool_name}, {"arguments", part.arguments}}}};
}

// A chat completion between its admission and its reply.
struct Job {
  Engine* engine = nullptr;  // the requested model's
  ChatRequest chat;
  GenerateRequest gen;
  std::unique_ptr<Scheduler::Ticket> ticket;
  std::string id;
  std::string model;
  std::int64_t created = 0;
  Clock::time_point arrived;
};

struct Outcome {
  GenerateResult result;
  std::string finish;  // "stop", "length" or "tool_calls"
  bool client_gone = false;
  std::string error;  // engine failure
};

}  // namespace

struct Server::Impl {
  Impl(Engine* single, EnginePool* pool, const ChatFormat& format,
       ServerOptions options)
      : single(single),
        pool(pool),
        format(format),
        options(std::move(options)),
        scheduler(this->options.max_queue),
        created(static_cast<std::int64_t>(std::time(nullptr))) {
    // Waiting requests hold a worker each; keep some for /health etc.
    const std::size_t workers =
        static_cast<std::size_t>(std::max(0, this->options.max_queue)) + 8;
    http.new_task_queue = [workers] {
      return new httplib::ThreadPool(workers);
    };
    http.set_payload_max_length(64u << 20);
    http.set_read_timeout(60);
    http.set_write_timeout(60);
    Routes();
  }

  void Routes() {
    http.set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) {
          if (options.api_key.empty() || req.path.rfind("/v1/", 0) != 0 ||
              req.get_header_value("Authorization") ==
                  "Bearer " + options.api_key) {
            return httplib::Server::HandlerResponse::Unhandled;
          }
          SendError(res, ApiError(401, "invalid_request_error",
                                  "invalid_api_key", "invalid API key"));
          return httplib::Server::HandlerResponse::Handled;
        });
    http.set_error_handler([](const httplib::Request&, httplib::Response& res) {
      if (res.body.empty()) {
        SendError(res, ApiError(res.status, "invalid_request_error", "",
                                res.status == 404 ? "no such endpoint"
                                                  : "request failed"));
      }
    });
    http.set_exception_handler([](const httplib::Request&,
                                  httplib::Response& res,
                                  std::exception_ptr e) {
      std::string what = "internal error";
      try {
        std::rethrow_exception(e);
      } catch (const std::exception& ex) {
        what = ex.what();
      } catch (...) {
      }
      SendError(res, ApiError(500, "server_error", "", what));
    });

    http.Get(
        "/health", [this](const httplib::Request&, httplib::Response& res) {
          json health = {{"status", "ok"}};
          if (pool) {
            const auto s = pool->status();
            health["model"] = s.loaded ? json(*s.loaded) : json(nullptr);
            health["loading"] = s.loading ? json(*s.loading) : json(nullptr);
          } else {
            health["model"] = single->info().id;
          }
          res.set_content(Dump(health), "application/json");
        });
    http.Get(
        "/v1/models", [this](const httplib::Request&, httplib::Response& res) {
          json data = json::array();
          for (Engine* engine : Engines()) {
            data.push_back(Model(*engine));
          }
          res.set_content(Dump({{"object", "list"}, {"data", std::move(data)}}),
                          "application/json");
        });
    http.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req,
                                          httplib::Response& res) {
      Engine* engine = FindModel(req.matches[1]);
      if (engine == nullptr) {
        SendError(res, ApiError(404, "invalid_request_error", "model_not_found",
                                "no such model", "model"));
        return;
      }
      res.set_content(Dump(Model(*engine)), "application/json");
    });
    http.Get("/metrics",
             [this](const httplib::Request&, httplib::Response& res) {
               std::string text =
                   metrics.Render(scheduler.waiting(), scheduler.running());
               if (pool) {
                 text += pool->RenderMetrics();
               }
               res.set_content(text, "text/plain; version=0.0.4");
             });
    http.Post("/v1/chat/completions",
              [this](const httplib::Request& req, httplib::Response& res) {
                Chat(req, res);
              });
  }

  json Model(const Engine& engine) const {
    json model = {{"id", engine.info().id},
                  {"object", "model"},
                  {"created", created},
                  {"owned_by", "gufo-npu"}};
    if (pool) {  // extension: which model is on the device
      model["loaded"] = pool->status().loaded == engine.info().id;
    }
    return model;
  }

  std::vector<Engine*> Engines() const {
    return pool ? pool->engines() : std::vector<Engine*>{single};
  }

  Engine* FindModel(const std::string& id) const {
    if (pool) {
      return pool->Find(id);
    }
    return single->info().id == id ? single : nullptr;
  }

  // The engine a request names. One model: any name (or none) is it.
  // Several: the named one (404 if unknown), or the default without a name.
  Engine& RequestedEngine(const nlohmann::ordered_json& body) const {
    if (!pool) {
      return *single;
    }
    if (const auto all = pool->engines(); all.size() == 1) {
      return *all.front();
    }
    const auto it = body.find("model");
    if (it == body.end() || it->is_null() ||
        (it->is_string() && it->get<std::string>().empty())) {
      return pool->Default();
    }
    if (!it->is_string()) {
      throw InvalidRequest("model must be a string", "model");
    }
    Engine* engine = pool->Find(it->get<std::string>());
    if (engine == nullptr) {
      throw ApiError(404, "invalid_request_error", "model_not_found",
                     "the model " + it->get<std::string>() +
                         " does not exist; GET /v1/models lists the models",
                     "model");
    }
    return *engine;
  }

  std::vector<int> StopIds(const Engine& engine) const {
    std::vector<int> ids;
    for (const auto& token : format.StopTokens()) {
      if (const int id = engine.FindToken(token); id >= 0) {
        ids.push_back(id);
      }
    }
    return ids;
  }

  void Chat(const httplib::Request& req, httplib::Response& res) {
    auto job = std::make_shared<Job>();
    job->arrived = Clock::now();
    try {
      nlohmann::ordered_json body;
      try {
        body = nlohmann::ordered_json::parse(req.body);
      } catch (const nlohmann::ordered_json::exception&) {
        throw InvalidRequest("the request body is not valid JSON");
      }
      job->chat = ParseChatRequest(body);
      job->engine = &RequestedEngine(body);
      Engine& engine = *job->engine;
      std::string text;
      try {
        text = format.Render(job->chat.prompt);
      } catch (const std::invalid_argument& e) {
        throw InvalidRequest(e.what(), "messages");
      }
      job->gen.prompt = engine.Encode(text);
      const int context = engine.info().context_length;
      const int prompt = static_cast<int>(job->gen.prompt.size());
      if (context > 0 && prompt >= context) {
        throw ApiError(400, "invalid_request_error", "context_length_exceeded",
                       "the prompt has " + std::to_string(prompt) +
                           " tokens; this model's context is " +
                           std::to_string(context) + " tokens",
                       "messages");
      }
      const int room = context > 0 ? context - prompt : INT_MAX;
      job->gen.max_tokens = std::min(job->chat.max_tokens.value_or(room), room);
      job->gen.sampling = job->chat.sampling;
      job->gen.stop_ids = StopIds(engine);
    } catch (const ApiError& e) {
      metrics.Request(e.status);
      SendError(res, e);
      return;
    }
    job->ticket = scheduler.Enter();
    if (!job->ticket) {
      metrics.Request(429);
      SendError(res, ApiError(429, "rate_limit_error", "queue_full",
                              "the server is busy; retry later"));
      return;
    }
    job->id = NewId("chatcmpl-");
    job->model = job->engine->info().id;
    job->created = static_cast<std::int64_t>(std::time(nullptr));

    if (job->chat.stream) {
      res.set_header("Cache-Control", "no-cache");
      res.set_header("X-Accel-Buffering", "no");
      res.set_chunked_content_provider(
          "text/event-stream",
          [this, job, &req](std::size_t, httplib::DataSink& sink) {
            Stream(*job, req, sink);
            sink.done();
            return true;
          });
    } else {
      Complete(*job, req, res);
    }
  }

  // Waits for the engine, generates, and hands parts to `emit`.
  Outcome Run(Job& job, const std::function<bool()>& gone,
              ReplyStream::Emit emit) {
    Outcome out;
    if (!job.ticket->Wait(gone)) {
      job.ticket.reset();
      out.client_gone = true;
      return out;
    }
    std::atomic<bool> cancel{false};
    job.gen.cancel = &cancel;
    Engine& engine = *job.engine;
    ReplyStream reply(engine, job.gen.stop_ids, job.chat.stop,
                      format.NewParser(job.chat.prompt), std::move(emit));
    bool first = true;
    try {
      out.result = engine.Generate(job.gen, [&](int id) {
        if (first) {
          first = false;
          metrics.TimeToFirstToken(Since(job.arrived));
        }
        if (gone()) {
          cancel = true;
          return false;
        }
        return reply.OnToken(id);
      });
    } catch (const std::exception& e) {
      job.ticket.reset();
      out.error = e.what();
      return out;
    }
    job.ticket.reset();  // the engine is free; the tail is only text
    out.client_gone = cancel || reply.delivery_failed();
    if (!out.client_gone) {
      reply.Finish();
      out.client_gone = reply.delivery_failed();
    }
    const auto& r = out.result;
    metrics.Generation(r.prompt_tokens, r.cached_prompt_tokens,
                       static_cast<int>(r.tokens.size()), r.prefill_seconds,
                       r.decode_seconds, r.steps, r.accepted);
    out.finish = !reply.stop_string_hit() && r.finish == FinishReason::kLength
                     ? "length"
                     : "stop";
    if (out.finish == "stop" && reply.tool_calls() > 0) {
      out.finish = "tool_calls";
    }
    return out;
  }

  json Usage(const Job& job, const GenerateResult& r) const {
    const int prompt = r.prompt_tokens > 0
                           ? r.prompt_tokens
                           : static_cast<int>(job.gen.prompt.size());
    const int completion = static_cast<int>(r.tokens.size());
    return {
        {"prompt_tokens", prompt},
        {"completion_tokens", completion},
        {"total_tokens", prompt + completion},
        {"prompt_tokens_details", {{"cached_tokens", r.cached_prompt_tokens}}}};
  }

  json Chunk(const Job& job, json delta, json finish) const {
    json chunk = {
        {"id", job.id},
        {"object", "chat.completion.chunk"},
        {"created", job.created},
        {"model", job.model},
        {"choices", json::array({{{"index", 0},
                                  {"delta", std::move(delta)},
                                  {"logprobs", nullptr},
                                  {"finish_reason", std::move(finish)}}})}};
    if (job.chat.include_usage) {
      chunk["usage"] = nullptr;
    }
    return chunk;
  }

  void Stream(Job& job, const httplib::Request& req, httplib::DataSink& sink) {
    const auto write = [&](const std::string& data) {
      const std::string event = "data: " + data + "\n\n";
      return sink.write(event.data(), event.size());
    };
    const auto gone = [&req] { return req.is_connection_closed(); };
    int tool_index = 0;
    const auto emit = [&](std::vector<OutputPart>&& parts) {
      for (auto& part : parts) {
        json delta;
        switch (part.kind) {
          case OutputPart::Kind::kContent:
            delta = {{"content", std::move(part.text)}};
            break;
          case OutputPart::Kind::kReasoning:
            delta = {{"reasoning_content", std::move(part.text)}};
            break;
          case OutputPart::Kind::kToolCall: {
            json call = ToolCall(part);
            call["index"] = tool_index++;
            delta = {{"tool_calls", json::array({std::move(call)})}};
            break;
          }
        }
        if (!write(Dump(Chunk(job, std::move(delta), nullptr)))) {
          return false;
        }
      }
      return true;
    };

    Outcome out;
    if (!write(Dump(
            Chunk(job, {{"role", "assistant"}, {"content", ""}}, nullptr)))) {
      out.client_gone = true;
      job.ticket.reset();
    } else {
      out = Run(job, gone, emit);
    }
    if (!out.error.empty()) {
      metrics.Request(500);
      write(Dump(ErrorJson(ApiError(500, "server_error", "", out.error))));
    } else if (out.client_gone) {
      metrics.Request(kClientClosed);
    } else {
      write(Dump(Chunk(job, json::object(), out.finish)));
      if (job.chat.include_usage) {
        write(Dump({{"id", job.id},
                    {"object", "chat.completion.chunk"},
                    {"created", job.created},
                    {"model", job.model},
                    {"choices", json::array()},
                    {"usage", Usage(job, out.result)}}));
      }
      write("[DONE]");
      metrics.Request(200);
    }
    metrics.RequestDuration(Since(job.arrived));
  }

  void Complete(Job& job, const httplib::Request& req, httplib::Response& res) {
    std::string content;
    std::string reasoning;
    json tool_calls = json::array();
    const auto emit = [&](std::vector<OutputPart>&& parts) {
      for (auto& part : parts) {
        switch (part.kind) {
          case OutputPart::Kind::kContent:
            content += part.text;
            break;
          case OutputPart::Kind::kReasoning:
            reasoning += part.text;
            break;
          case OutputPart::Kind::kToolCall:
            tool_calls.push_back(ToolCall(part));
            break;
        }
      }
      return true;
    };
    const Outcome out =
        Run(job, [&req] { return req.is_connection_closed(); }, emit);
    metrics.RequestDuration(Since(job.arrived));
    if (!out.error.empty()) {
      metrics.Request(500);
      SendError(res, ApiError(500, "server_error", "", out.error));
      return;
    }
    if (out.client_gone) {
      metrics.Request(kClientClosed);
      res.status = kClientClosed;
      return;
    }
    json message = {
        {"role", "assistant"},
        {"content", content.empty() && !tool_calls.empty() ? json(nullptr)
                                                           : json(content)}};
    if (!reasoning.empty()) {
      message["reasoning_content"] = reasoning;
    }
    if (!tool_calls.empty()) {
      message["tool_calls"] = std::move(tool_calls);
    }
    const json body = {
        {"id", job.id},
        {"object", "chat.completion"},
        {"created", job.created},
        {"model", job.model},
        {"choices", json::array({{{"index", 0},
                                  {"message", std::move(message)},
                                  {"logprobs", nullptr},
                                  {"finish_reason", out.finish}}})},
        {"usage", Usage(job, out.result)}};
    metrics.Request(200);
    res.set_content(Dump(body), "application/json");
  }

  Engine* single;    // one model, or
  EnginePool* pool;  // several
  const ChatFormat& format;
  ServerOptions options;
  httplib::Server http;
  Scheduler scheduler;
  Metrics metrics;
  std::int64_t created;
};

Server::Server(Engine& engine, const ChatFormat& format, ServerOptions options)
    : impl_(std::make_unique<Impl>(&engine, nullptr, format,
                                   std::move(options))) {}

Server::Server(EnginePool& pool, const ChatFormat& format,
               ServerOptions options)
    : impl_(
          std::make_unique<Impl>(nullptr, &pool, format, std::move(options))) {}

Server::~Server() = default;

int Server::Bind() {
  auto& o = impl_->options;
  if (o.port == 0) {
    const int port = impl_->http.bind_to_any_port(o.host);
    if (port <= 0) {
      throw std::runtime_error("cannot bind " + o.host);
    }
    o.port = port;
  } else if (!impl_->http.bind_to_port(o.host, o.port)) {
    throw std::runtime_error("cannot bind " + o.host + ":" +
                             std::to_string(o.port));
  }
  return o.port;
}

void Server::Run() {
  impl_->http.listen_after_bind();
}

void Server::Stop() {
  impl_->http.stop();
}

}  // namespace gufo_npu::serve
