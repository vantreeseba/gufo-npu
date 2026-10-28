#pragma once

#include <memory>
#include <string>

#include "serve/chat_format.hpp"
#include "serve/engine.hpp"

// The OpenAI-compatible HTTP server (docs/SERVING.md): /v1/chat/completions
// (streaming and not), the legacy /v1/completions, /v1/models, /health
// and /metrics over one engine.
namespace gufo_npu::serve {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;      // 0: any free port (tests)
  int max_queue = 8;    // requests waiting behind the running one; then 429
  std::string api_key;  // when set, /v1/* needs "Authorization: Bearer KEY"
};

class Server {
public:
  Server(Engine& engine, const ChatFormat& format, ServerOptions options);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  // Binds the socket; returns the port. Throws std::runtime_error.
  int Bind();
  // Serves until Stop(). Call after Bind().
  void Run();
  // From any thread: stops Run().
  void Stop();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::serve
