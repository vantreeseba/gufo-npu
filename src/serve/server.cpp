#include "serve/server.hpp"

#include <httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "serve/engine_pool.hpp"
#include "serve/logging.hpp"
#include "serve/logprobs.hpp"
#include "serve/metrics.hpp"
#include "serve/openai.hpp"
#include "serve/reply.hpp"
#include "serve/scheduler.hpp"
#include "serve/structured.hpp"
#include "serve/text.hpp"

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

// The log line of the request this thread is serving (httplib runs a
// request's routing, handler, streamed body and logger on one thread).
struct RequestLog {
  const httplib::Request* req = nullptr;
  std::string id;
  std::string details;
  std::string outcome = "completed";
};

RequestLog& CurrentLog() {
  thread_local RequestLog log;
  return log;
}

void AddDetail(const std::string& detail) {
  auto& log = CurrentLog();
  if (!log.details.empty()) {
    log.details += ' ';
  }
  log.details += detail;
}

// Polled endpoints log at DEBUG, everything else at INFO.
LogLevel RequestLevel(const httplib::Request& req) {
  return req.method == "GET" &&
                 (req.path == "/health" || req.path == "/metrics")
             ? LogLevel::kDebug
             : LogLevel::kInfo;
}

void SendError(httplib::Response& res, const ApiError& error) {
  res.status = error.status;
  res.set_content(Dump(ErrorJson(error)), "application/json");
  std::string detail = "error_type=" + error.type;
  if (!error.code.empty()) {
    detail += " error_code=" + error.code;
  }
  if (!error.param.empty()) {
    detail += " param=" + error.param;
  }
  AddDetail(detail);
}

json ToolCall(const OutputPart& part) {
  return {
      {"id", part.tool_id.empty() ? NewId("call_") : part.tool_id},
      {"type", "function"},
      {"function", {{"name", part.tool_name}, {"arguments", part.arguments}}}};
}

json Bytes(const std::string& text) {
  json out = json::array();
  for (const unsigned char c : text) {
    out.push_back(static_cast<int>(c));
  }
  return out;
}

// A request between its admission and its reply: a chat completion or a
// legacy completion.
struct Job {
  Engine* engine = nullptr;  // the requested model's
  ChatRequest chat;
  CompletionRequest completion;
  // One per prompt (a chat has one; a completion may have several).
  std::vector<GenerateRequest> gens;
  std::vector<std::string> echo;  // completion echo text per prompt
  std::unique_ptr<Scheduler::Ticket> ticket;
  std::string id;
  std::string model;
  std::int64_t created = 0;
  Clock::time_point arrived;
  // For the request's log line.
  int queue_depth = 0;  // requests waiting when it was admitted
  std::optional<Clock::time_point> acquired;     // the engine became ours
  std::optional<Clock::time_point> first_token;  // its first generated token
};

struct Outcome {
  GenerateResult result;
  std::string finish;  // "stop", "length" or "tool_calls"
  bool client_gone = false;
  std::string error;  // engine failure
};

// What one generation reports besides its text parts.
struct Hooks {
  std::function<bool()> gone;
  ReplyStream::Emit emit;
  // Each generated token's logprobs (not the stop token's), before its text
  // reaches `emit`; unset: no logprobs.
  std::function<void(TokenLogprobs&&)> logprob;
  int top_logprobs = 0;
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

  // Opens the log line of a request this thread now serves.
  static void StartLog(const httplib::Request& req) {
    static std::atomic<std::uint64_t> next{0};
    auto& log = CurrentLog();
    log = RequestLog{};
    log.req = &req;
    log.id = "r" + std::to_string(next.fetch_add(1) + 1);
    const LogLevel level = RequestLevel(req);
    if (Logger::Enabled(level)) {
      Logger::Log(level, "http",
                  "request=" + log.id + " event=received method=" + req.method +
                      " path=" + req.path +
                      " body_bytes=" + std::to_string(req.body.size()));
    }
  }

  void Routes() {
    http.set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) {
          StartLog(req);
          if (options.api_key.empty() || req.path.rfind("/v1/", 0) != 0 ||
              req.get_header_value("Authorization") ==
                  "Bearer " + options.api_key) {
            return httplib::Server::HandlerResponse::Unhandled;
          }
          SendError(res, ApiError(401, "invalid_request_error",
                                  "invalid_api_key", "invalid API key"));
          return httplib::Server::HandlerResponse::Handled;
        });
    http.set_logger(
        [](const httplib::Request& req, const httplib::Response& res) {
          auto& log = CurrentLog();
          if (log.req != &req) {  // refused before routing (malformed)
            StartLog(req);
          }
          Logger::LogRequest(log.id, req.method, req.path, res.status,
                             std::chrono::duration<double, std::milli>(
                                 Clock::now() - req.start_time_)
                                 .count(),
                             log.details, log.outcome, RequestLevel(req));
          log.req = nullptr;
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
    http.Post("/v1/completions",
              [this](const httplib::Request& req, httplib::Response& res) {
                Completions(req, res);
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

  static nlohmann::ordered_json ParseBody(const httplib::Request& req) {
    try {
      return nlohmann::ordered_json::parse(req.body);
    } catch (const nlohmann::ordered_json::exception&) {
      throw InvalidRequest("the request body is not valid JSON");
    }
  }

  void CheckBias(const Engine& engine, const SamplingParams& s) const {
    for (const auto& [id, bias] : s.logit_bias) {
      if (id >= engine.info().vocab) {
        throw InvalidRequest("logit_bias token id " + std::to_string(id) +
                                 " is outside the vocabulary",
                             "logit_bias");
      }
    }
  }

  // A generation request for `prompt`, checked against the context.
  GenerateRequest Gen(const Engine& engine, std::vector<int> prompt,
                      std::optional<int> max_tokens,
                      const SamplingParams& sampling, const char* param) const {
    GenerateRequest gen;
    gen.prompt = std::move(prompt);
    const int context = engine.info().context_length;
    const int size = static_cast<int>(gen.prompt.size());
    if (context > 0 && size >= context) {
      throw ApiError(400, "invalid_request_error", "context_length_exceeded",
                     "the prompt has " + std::to_string(size) +
                         " tokens; this model's context is " +
                         std::to_string(context) + " tokens",
                     param);
    }
    const int room = context > 0 ? context - size : INT_MAX;
    gen.max_tokens = std::min(max_tokens.value_or(room), room);
    gen.sampling = sampling;
    gen.stop_ids = StopIds(engine);
    gen.draft = options.draft;
    return gen;
  }

  // Admits `job` to the queue; false after sending 429.
  bool Admit(Job& job, httplib::Response& res, const char* id_prefix) {
    job.queue_depth = scheduler.waiting();
    job.ticket = scheduler.Enter();
    if (!job.ticket) {
      metrics.Request(429);
      SendError(res, ApiError(429, "rate_limit_error", "queue_full",
                              "the server is busy; retry later"));
      return false;
    }
    job.id = NewId(id_prefix);
    job.model = job.engine->info().id;
    job.created = static_cast<std::int64_t>(std::time(nullptr));
    return true;
  }

  void Chat(const httplib::Request& req, httplib::Response& res) {
    auto job = std::make_shared<Job>();
    job->arrived = Clock::now();
    try {
      const auto body = ParseBody(req);
      job->chat = ParseChatRequest(body);
      job->engine = &RequestedEngine(body);
      Engine& engine = *job->engine;
      CheckBias(engine, job->chat.sampling);
      std::string text;
      try {
        text = format.Render(job->chat.prompt);
      } catch (const std::invalid_argument& e) {
        throw InvalidRequest(e.what(), "messages");
      }
      job->gens.push_back(Gen(engine, engine.Encode(text), job->chat.max_tokens,
                              job->chat.sampling, "messages"));
      job->gens.back().sampling.constraint =
          Constraint(job->chat, engine, job->gens.back().stop_ids);
    } catch (const ApiError& e) {
      metrics.Request(e.status);
      SendError(res, e);
      return;
    }
    if (!Admit(*job, res, "chatcmpl-")) {
      return;
    }
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

  // The constraint a request's response_format or tool_choice asks for,
  // or none.
  std::shared_ptr<TokenConstraint> Constraint(
      const ChatRequest& chat, const Engine& engine,
      const std::vector<int>& stop_ids) {
    std::shared_ptr<const Grammar> grammar;
    try {
      if (chat.tool_required) {
        grammar = ToolCallGrammar(chat.prompt.tools, chat.tool_name);
      } else if (chat.format == ChatRequest::Format::kJsonSchema) {
        grammar = JsonSchemaGrammar(chat.json_schema, chat.json_strict);
      } else if (chat.format == ChatRequest::Format::kJsonObject) {
        grammar = JsonObjectGrammar();
      } else {
        return nullptr;
      }
    } catch (const GrammarError& e) {
      throw InvalidRequest(
          e.what(), chat.tool_required ? "tool_choice" : "response_format");
    }
    return NewGrammarConstraint(std::move(grammar), Vocab(engine), stop_ids,
                                chat.prompt.thinking);
  }

  // An engine's vocabulary bytes for constraints, built on first use (one
  // per model; a pool's engines keep their tokenizer while unloaded).
  std::shared_ptr<const VocabIndex> Vocab(const Engine& engine) {
    std::lock_guard lock(vocab_mutex);
    auto& vocab = vocabs[&engine];
    if (!vocab) {
      const int n = std::max(0, engine.info().vocab);
      std::vector<std::string> bytes(static_cast<std::size_t>(n));
      std::vector<std::uint8_t> special(static_cast<std::size_t>(n), 0);
      for (int id = 0; id < n; ++id) {
        try {
          bytes[static_cast<std::size_t>(id)] = engine.TokenBytes(id);
        } catch (const std::exception&) {
          continue;  // padding past the tokenizer: never allowed
        }
        try {
          special[static_cast<std::size_t>(id)] =
              engine.IsSpecialToken(id) ? 1 : 0;
        } catch (const std::exception&) {
          // An engine without a tokenizer (tests): no added tokens.
        }
      }
      vocab = std::make_shared<const VocabIndex>(std::move(bytes),
                                                 std::move(special));
    }
    return vocab;
  }

  // Waits for the engine. False when the client left first.
  bool Acquire(Job& job, const std::function<bool()>& gone) {
    if (!job.ticket->Wait(gone)) {
      job.ticket.reset();
      return false;
    }
    job.acquired = Clock::now();
    return true;
  }

  // One generation of `base` as choice `choice` (the engine is ours).
  Outcome Run(Job& job, const GenerateRequest& base, int choice,
              std::unique_ptr<OutputParser> parser,
              const std::vector<std::string>& stop, int max_tool_calls,
              const Hooks& hooks) {
    Engine& engine = *job.engine;
    Outcome out;
    GenerateRequest gen = base;
    // Choices differ: each draws from its own seed (seed + choice).
    if (gen.sampling.seed && choice > 0) {
      gen.sampling.seed = *gen.sampling.seed + static_cast<unsigned>(choice);
    }
    std::optional<LogprobRecorder> recorder;
    if (hooks.logprob) {
      recorder.emplace(hooks.top_logprobs);
      gen.sampling.observer = &*recorder;
    }
    std::atomic<bool> cancel{false};
    gen.cancel = &cancel;
    ReplyStream reply(engine, gen.stop_ids, stop, std::move(parser),
                      hooks.emit);
    reply.set_max_tool_calls(max_tool_calls);
    const auto is_stop = [&](int id) {
      return std::find(gen.stop_ids.begin(), gen.stop_ids.end(), id) !=
             gen.stop_ids.end();
    };
    // Time to first token: the request's first generation only.
    bool first = choice == 0 && &base == &job.gens.front();
    try {
      out.result = engine.Generate(gen, [&](int id) {
        if (first) {
          first = false;
          job.first_token = Clock::now();
          metrics.TimeToFirstToken(Since(job.arrived));
        }
        if (hooks.gone()) {
          cancel = true;
          return false;
        }
        if (recorder && !is_stop(id)) {
          if (auto lp = recorder->Take(id)) {
            hooks.logprob(std::move(*lp));
          }
        }
        return reply.OnToken(id);
      });
    } catch (const std::exception& e) {
      out.error = e.what();
      return out;
    }
    out.client_gone = cancel || reply.delivery_failed();
    if (!out.client_gone) {
      reply.Finish();
      out.client_gone = reply.delivery_failed();
    }
    const auto& r = out.result;
    metrics.Generation(r.prompt_tokens, r.cached_prompt_tokens,
                       static_cast<int>(r.tokens.size()), r.prefill_seconds,
                       r.decode_seconds, r.steps, r.accepted);
    out.finish = !reply.stop_string_hit() && !reply.tool_limit_hit() &&
                         r.finish == FinishReason::kLength
                     ? "length"
                     : "stop";
    if (out.finish == "stop" && reply.tool_calls() > 0) {
      out.finish = "tool_calls";
    }
    return out;
  }

  // The request's generation keys for its log line, as gufo names them:
  // tokens, finish, cache, queue, time to first token, prefill and decode
  // rates, and MTP drafts, summed over the request's choices.
  void LogGeneration(const Job& job, const std::vector<Outcome>& outs) const {
    const bool chat = job.completion.prompts() == 0;
    int prompt = 0;
    for (const auto& gen : job.gens) {
      prompt += static_cast<int>(gen.prompt.size());
    }
    int prefill = 0;
    int cached = 0;
    int generated = 0;
    int steps = 0;
    int accepted = 0;
    double prefill_seconds = 0;
    double decode_seconds = 0;
    std::string finish;
    for (const auto& out : outs) {
      const auto& r = out.result;
      prefill += std::max(0, r.prompt_tokens - r.cached_prompt_tokens);
      cached += r.cached_prompt_tokens;
      generated += static_cast<int>(r.tokens.size());
      steps += r.steps;
      accepted += r.accepted;
      prefill_seconds += r.prefill_seconds;
      decode_seconds += r.decode_seconds;
      finish += finish.empty() ? "" : ",";
      finish += !out.error.empty() ? "error"
                : out.client_gone  ? "cancelled"
                                   : out.finish;
    }
    const auto ms = [](Clock::time_point from, Clock::time_point to) {
      return std::chrono::duration<double, std::milli>(to - from).count();
    };
    const bool draft = options.draft && job.engine->info().can_draft;
    std::ostringstream o;
    o << std::fixed << std::setprecision(1) << "model=" << job.model
      << " kind=" << (chat ? "chat" : "completion") << " stream="
      << ((chat ? job.chat.stream : job.completion.stream) ? 1 : 0)
      << " n=" << (chat ? job.chat.n : job.completion.n)
      << " prompt_tokens=" << prompt << " prefill_tokens=" << prefill
      << " generated_tokens=" << generated
      << " finish=" << (finish.empty() ? "none" : finish)
      << " cache=" << (cached > 0 ? "hit" : "miss")
      << " cached_tokens=" << cached << " queue_depth=" << job.queue_depth
      << " queue_ms=" << ms(job.arrived, job.acquired.value_or(Clock::now()))
      << " ttft_ms="
      << (job.first_token ? ms(job.arrived, *job.first_token) : 0.0)
      << " prefill_tps="
      << (prefill_seconds > 0 ? prefill / prefill_seconds : 0.0)
      << " decode_tps="
      << (decode_seconds > 0 ? generated / decode_seconds : 0.0)
      << " draft=" << (draft ? "mtp" : "none");
    if (draft) {
      o << " decode_steps=" << steps << " draft_accepted=" << accepted
        << " acceptance_pct=" << (steps > 0 ? 100.0 * accepted / steps : 0.0);
    }
    if (chat) {
      if (job.chat.format == ChatRequest::Format::kJsonSchema) {
        o << " response_format=json_schema";
      } else if (job.chat.format == ChatRequest::Format::kJsonObject) {
        o << " response_format=json_object";
      }
      if (!job.chat.prompt.tools.empty()) {
        o << " tools=" << job.chat.prompt.tools.size();
      }
      if (job.chat.tool_required) {
        o << " tool_choice="
          << (job.chat.tool_name.empty() ? "required" : "function");
      }
      if (job.chat.logprobs) {
        o << " logprobs=" << job.chat.top_logprobs;
      }
    } else if (job.completion.logprobs) {
      o << " logprobs=" << *job.completion.logprobs;
    }
    for (const auto& out : outs) {
      if (!out.error.empty()) {
        o << " reason=" << out.error;
        break;
      }
    }
    AddDetail(o.str());
  }

  // Usage over all choices: each prompt counted once, completions summed.
  static json Usage(const Job& job, const std::vector<Outcome>& outs,
                    int per_prompt) {
    int prompt = 0;
    int completion = 0;
    int cached = 0;
    for (std::size_t i = 0; i < outs.size(); ++i) {
      const auto& r = outs[i].result;
      completion += static_cast<int>(r.tokens.size());
      if (i % static_cast<std::size_t>(per_prompt) == 0) {
        const std::size_t p = i / static_cast<std::size_t>(per_prompt);
        prompt += p < job.gens.size()
                      ? static_cast<int>(job.gens[p].prompt.size())
                      : r.prompt_tokens;
        cached += r.cached_prompt_tokens;
      }
    }
    return {{"prompt_tokens", prompt},
            {"completion_tokens", completion},
            {"total_tokens", prompt + completion},
            {"prompt_tokens_details", {{"cached_tokens", cached}}}};
  }

  json LogprobEntry(const Engine& engine, int id, float logprob) const {
    const std::string text = engine.TokenBytes(id);
    return {{"token", text}, {"logprob", logprob}, {"bytes", Bytes(text)}};
  }

  json ChatLogprob(const Engine& engine, const TokenLogprobs& lp) const {
    json entry = LogprobEntry(engine, lp.id, lp.logprob);
    json top = json::array();
    for (const auto& [id, logprob] : lp.top) {
      top.push_back(LogprobEntry(engine, id, logprob));
    }
    entry["top_logprobs"] = std::move(top);
    return entry;
  }

  json Chunk(const Job& job, int index, json delta, json finish,
             json logprobs = nullptr) const {
    json chunk = {
        {"id", job.id},
        {"object", "chat.completion.chunk"},
        {"created", job.created},
        {"model", job.model},
        {"choices", json::array({{{"index", index},
                                  {"delta", std::move(delta)},
                                  {"logprobs", std::move(logprobs)},
                                  {"finish_reason", std::move(finish)}}})}};
    if (job.chat.include_usage) {
      chunk["usage"] = nullptr;
    }
    return chunk;
  }

  int MaxToolCalls(const Job& job) const {
    return job.chat.parallel_tool_calls ? 0 : 1;
  }

  void Stream(Job& job, const httplib::Request& req, httplib::DataSink& sink) {
    const auto write = [&](const std::string& data) {
      const std::string event = "data: " + data + "\n\n";
      return sink.write(event.data(), event.size());
    };
    const auto gone = [&req] { return req.is_connection_closed(); };
    std::vector<Outcome> outs;
    std::string error;
    bool client_gone = !Acquire(job, gone);
    for (int c = 0; c < job.chat.n && !client_gone && error.empty(); ++c) {
      int tool_index = 0;
      json pending = json::array();  // logprobs not yet sent
      const auto logprobs = [&]() -> json {
        if (!job.chat.logprobs || pending.empty()) {
          return nullptr;
        }
        json out = {{"content", std::move(pending)}};
        pending = json::array();
        return out;
      };
      const auto emit = [&](std::vector<OutputPart>&& parts) {
        for (auto& part : parts) {
          std::vector<json> deltas;
          switch (part.kind) {
            case OutputPart::Kind::kContent:
              deltas.push_back({{"content", std::move(part.text)}});
              break;
            case OutputPart::Kind::kReasoning:
              deltas.push_back({{"reasoning_content", std::move(part.text)}});
              break;
            case OutputPart::Kind::kToolCall: {
              // OpenAI's fragment shape: the call's id and name with empty
              // arguments, then the arguments.
              const json call = ToolCall(part);
              json head = call;
              head["index"] = tool_index;
              head["function"]["arguments"] = "";
              deltas.push_back({{"tool_calls", json::array({head})}});
              deltas.push_back(
                  {{"tool_calls",
                    json::array({{{"index", tool_index},
                                  {"function",
                                   {{"arguments",
                                     call["function"]["arguments"]}}}}})}});
              ++tool_index;
              break;
            }
          }
          for (auto& delta : deltas) {
            if (!write(Dump(
                    Chunk(job, c, std::move(delta), nullptr, logprobs())))) {
              return false;
            }
          }
        }
        return true;
      };
      Hooks hooks;
      hooks.gone = gone;
      hooks.emit = emit;
      if (job.chat.logprobs) {
        hooks.top_logprobs = job.chat.top_logprobs;
        hooks.logprob = [&](TokenLogprobs&& lp) {
          pending.push_back(ChatLogprob(*job.engine, lp));
        };
      }
      if (!write(Dump(Chunk(job, c, {{"role", "assistant"}, {"content", ""}},
                            nullptr)))) {
        client_gone = true;
        break;
      }
      Outcome out = Run(job, job.gens[0], c, format.NewParser(job.chat.prompt),
                        job.chat.stop, MaxToolCalls(job), hooks);
      if (!out.error.empty()) {
        error = out.error;
      } else if (out.client_gone) {
        client_gone = true;
      } else if (!write(Dump(
                     Chunk(job, c, json::object(), out.finish, logprobs())))) {
        client_gone = true;
      }
      outs.push_back(std::move(out));
    }
    job.ticket.reset();
    LogGeneration(job, outs);
    if (!error.empty()) {
      metrics.Request(500);
      CurrentLog().outcome = "stream_error";
      write(Dump(ErrorJson(ApiError(500, "server_error", "", error))));
    } else if (client_gone) {
      metrics.Request(kClientClosed);
      CurrentLog().outcome = "disconnected";
    } else {
      if (job.chat.include_usage) {
        write(Dump({{"id", job.id},
                    {"object", "chat.completion.chunk"},
                    {"created", job.created},
                    {"model", job.model},
                    {"choices", json::array()},
                    {"usage", Usage(job, outs, job.chat.n)}}));
      }
      write("[DONE]");
      metrics.Request(200);
    }
    metrics.RequestDuration(Since(job.arrived));
  }

  void Complete(Job& job, const httplib::Request& req, httplib::Response& res) {
    const auto gone = [&req] { return req.is_connection_closed(); };
    std::vector<Outcome> outs;
    json choices = json::array();
    std::string error;
    bool client_gone = !Acquire(job, gone);
    for (int c = 0; c < job.chat.n && !client_gone && error.empty(); ++c) {
      std::string content;
      std::string reasoning;
      json tool_calls = json::array();
      json logprobs = json::array();
      Hooks hooks;
      hooks.gone = gone;
      hooks.emit = [&](std::vector<OutputPart>&& parts) {
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
      if (job.chat.logprobs) {
        hooks.top_logprobs = job.chat.top_logprobs;
        hooks.logprob = [&](TokenLogprobs&& lp) {
          logprobs.push_back(ChatLogprob(*job.engine, lp));
        };
      }
      Outcome out = Run(job, job.gens[0], c, format.NewParser(job.chat.prompt),
                        job.chat.stop, MaxToolCalls(job), hooks);
      if (!out.error.empty()) {
        error = out.error;
      } else if (out.client_gone) {
        client_gone = true;
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
      choices.push_back(
          {{"index", c},
           {"message", std::move(message)},
           {"logprobs", job.chat.logprobs
                            ? json({{"content", std::move(logprobs)}})
                            : json(nullptr)},
           {"finish_reason", out.finish}});
      outs.push_back(std::move(out));
    }
    job.ticket.reset();
    metrics.RequestDuration(Since(job.arrived));
    LogGeneration(job, outs);
    if (!error.empty()) {
      metrics.Request(500);
      SendError(res, ApiError(500, "server_error", "", error));
      return;
    }
    if (client_gone) {
      metrics.Request(kClientClosed);
      CurrentLog().outcome = "disconnected";
      res.status = kClientClosed;
      return;
    }
    const json body = {{"id", job.id},
                       {"object", "chat.completion"},
                       {"created", job.created},
                       {"model", job.model},
                       {"choices", std::move(choices)},
                       {"usage", Usage(job, outs, job.chat.n)}};
    metrics.Request(200);
    res.set_content(Dump(body), "application/json");
  }

  // ---- /v1/completions (legacy) ----

  // The prompt's text for echo: its tokens' bytes as whole characters.
  std::string Text(const Engine& engine, const std::vector<int>& ids) const {
    Utf8Assembler utf8;
    std::string text;
    for (const int id : ids) {
      text += utf8.Feed(engine.TokenBytes(id));
    }
    return text + utf8.Finish();
  }

  void Completions(const httplib::Request& req, httplib::Response& res) {
    auto job = std::make_shared<Job>();
    job->arrived = Clock::now();
    try {
      const auto body = ParseBody(req);
      job->completion = ParseCompletionRequest(body);
      job->engine = &RequestedEngine(body);
      Engine& engine = *job->engine;
      auto& c = job->completion;
      CheckBias(engine, c.sampling);
      for (std::size_t p = 0; p < c.prompts(); ++p) {
        std::vector<int> ids;
        if (c.texts.empty()) {
          ids = c.ids[p];
          for (const int id : ids) {
            if (id >= engine.info().vocab) {
              throw InvalidRequest("prompt token id " + std::to_string(id) +
                                       " is outside the vocabulary",
                                   "prompt");
            }
          }
        } else {
          ids = engine.Encode(c.texts[p]);
          if (ids.empty()) {
            throw InvalidRequest("prompt has no tokens", "prompt");
          }
        }
        if (c.echo) {
          job->echo.push_back(c.texts.empty() ? Text(engine, ids) : c.texts[p]);
        }
        job->gens.push_back(
            Gen(engine, std::move(ids), c.max_tokens, c.sampling, "prompt"));
      }
    } catch (const ApiError& e) {
      metrics.Request(e.status);
      SendError(res, e);
      return;
    }
    if (!Admit(*job, res, "cmpl-")) {
      return;
    }
    if (job->completion.stream) {
      res.set_header("Cache-Control", "no-cache");
      res.set_header("X-Accel-Buffering", "no");
      res.set_chunked_content_provider(
          "text/event-stream",
          [this, job, &req](std::size_t, httplib::DataSink& sink) {
            Generate(*job, req, &sink, nullptr);
            sink.done();
            return true;
          });
    } else {
      Generate(*job, req, nullptr, &res);
    }
  }

  // Legacy logprobs for some tokens: {tokens, token_logprobs, top_logprobs,
  // text_offset}, offsets counted from `offset`.
  struct Legacy {
    json tokens = json::array();
    json token_logprobs = json::array();
    json top = json::array();
    json offsets = json::array();
    std::size_t offset = 0;

    bool empty() const { return tokens.empty(); }
    json Take() {
      json out = {{"tokens", std::move(tokens)},
                  {"token_logprobs", std::move(token_logprobs)},
                  {"top_logprobs", std::move(top)},
                  {"text_offset", std::move(offsets)}};
      tokens = token_logprobs = top = offsets = json::array();
      return out;
    }
  };

  void AddLegacy(const Engine& engine, Legacy& legacy,
                 const TokenLogprobs& lp) const {
    const std::string text = engine.TokenBytes(lp.id);
    legacy.tokens.push_back(text);
    legacy.token_logprobs.push_back(lp.logprob);
    json top = json::object();
    for (const auto& [id, logprob] : lp.top) {
      top[engine.TokenBytes(id)] = logprob;
    }
    legacy.top.push_back(std::move(top));
    legacy.offsets.push_back(legacy.offset);
    legacy.offset += text.size();
  }

  // Runs every prompt's n choices; streams to `sink` or answers in `res`.
  void Generate(Job& job, const httplib::Request& req, httplib::DataSink* sink,
                httplib::Response* res) {
    const auto& c = job.completion;
    const auto gone = [&req] { return req.is_connection_closed(); };
    const auto write = [&](const std::string& data) {
      const std::string event = "data: " + data + "\n\n";
      return sink->write(event.data(), event.size());
    };
    const auto chunk = [&](int index, std::string text, json logprobs,
                           json finish) {
      json out = {
          {"id", job.id},
          {"object", "text_completion"},
          {"created", job.created},
          {"model", job.model},
          {"choices", json::array({{{"text", std::move(text)},
                                    {"index", index},
                                    {"logprobs", std::move(logprobs)},
                                    {"finish_reason", std::move(finish)}}})}};
      if (c.include_usage) {
        out["usage"] = nullptr;
      }
      return Dump(out);
    };
    std::vector<Outcome> outs;
    json choices = json::array();
    std::string error;
    bool client_gone = !Acquire(job, gone);
    for (std::size_t p = 0;
         p < job.gens.size() && !client_gone && error.empty(); ++p) {
      for (int i = 0; i < c.n && !client_gone && error.empty(); ++i) {
        const int index = static_cast<int>(p) * c.n + i;
        std::string text = c.echo ? job.echo[p] : std::string();
        Legacy legacy;
        legacy.offset = text.size();
        if (sink != nullptr && !text.empty() &&
            !write(chunk(index, text, nullptr, nullptr))) {
          client_gone = true;
          break;
        }
        Hooks hooks;
        hooks.gone = gone;
        hooks.emit = [&](std::vector<OutputPart>&& parts) {
          for (auto& part : parts) {
            if (sink == nullptr) {
              text += part.text;
              continue;
            }
            json lp =
                c.logprobs && !legacy.empty() ? legacy.Take() : json(nullptr);
            if (!write(chunk(index, std::move(part.text), std::move(lp),
                             nullptr))) {
              return false;
            }
          }
          return true;
        };
        if (c.logprobs) {
          hooks.top_logprobs = *c.logprobs;
          hooks.logprob = [&](TokenLogprobs&& lp) {
            AddLegacy(*job.engine, legacy, lp);
          };
        }
        Outcome out =
            Run(job, job.gens[p], i, std::make_unique<PlainTextParser>(),
                c.stop, 0, hooks);
        if (!out.error.empty()) {
          error = out.error;
        } else if (out.client_gone) {
          client_gone = true;
        } else if (sink != nullptr) {
          json lp =
              c.logprobs && !legacy.empty() ? legacy.Take() : json(nullptr);
          if (!write(chunk(index, "", std::move(lp), out.finish))) {
            client_gone = true;
          }
        } else {
          choices.push_back(
              {{"text", std::move(text)},
               {"index", index},
               {"logprobs", c.logprobs ? legacy.Take() : json(nullptr)},
               {"finish_reason", out.finish}});
        }
        outs.push_back(std::move(out));
      }
    }
    job.ticket.reset();
    metrics.RequestDuration(Since(job.arrived));
    LogGeneration(job, outs);
    if (!error.empty()) {
      metrics.Request(500);
      if (sink != nullptr) {
        CurrentLog().outcome = "stream_error";
        write(Dump(ErrorJson(ApiError(500, "server_error", "", error))));
      } else {
        SendError(*res, ApiError(500, "server_error", "", error));
      }
      return;
    }
    if (client_gone) {
      metrics.Request(kClientClosed);
      CurrentLog().outcome = "disconnected";
      if (res != nullptr) {
        res->status = kClientClosed;
      }
      return;
    }
    metrics.Request(200);
    const json usage = Usage(job, outs, c.n);
    if (sink != nullptr) {
      if (c.include_usage) {
        write(Dump({{"id", job.id},
                    {"object", "text_completion"},
                    {"created", job.created},
                    {"model", job.model},
                    {"choices", json::array()},
                    {"usage", usage}}));
      }
      write("[DONE]");
      return;
    }
    res->set_content(Dump({{"id", job.id},
                           {"object", "text_completion"},
                           {"created", job.created},
                           {"model", job.model},
                           {"choices", std::move(choices)},
                           {"usage", usage}}),
                     "application/json");
  }

  Engine* single;    // one model, or
  EnginePool* pool;  // several
  const ChatFormat& format;
  ServerOptions options;
  httplib::Server http;
  Scheduler scheduler;
  Metrics metrics;
  std::int64_t created;
  std::mutex vocab_mutex;
  std::map<const Engine*, std::shared_ptr<const VocabIndex>> vocabs;
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
