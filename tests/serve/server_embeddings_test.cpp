// POST /v1/embeddings over HTTP on localhost with a scripted embedding
// engine (no model, no NPU): input shapes, usage, dimensions, base64,
// the instruction extension, limits, and the chat routes refusing an
// embedding model (and the embeddings route a chat model).
#include <httplib/httplib.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "fakes.hpp"
#include "serve/server.hpp"

namespace {

using gufo_npu::serve::EmbeddingInput;
using gufo_npu::serve::Engine;
using gufo_npu::serve::GenerateRequest;
using gufo_npu::serve::GenerateResult;
using gufo_npu::serve::ModelInfo;
using gufo_npu::serve::Server;
using gufo_npu::serve::ServerOptions;
using gufo_npu::serve::testing::FakeEngine;
using gufo_npu::serve::testing::FakeFormat;
using nlohmann::json;

constexpr int kLength = 8;
constexpr int kMinDimensions = 4;
constexpr int kContext = 32;

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

// One token per byte of the text (or the given ids), after a template of
// two tokens; vector k of an input is proportional to (n + k + 1).
class FakeEmbedder : public Engine {
public:
  FakeEmbedder() {
    info_.id = "fake-embed";
    info_.architecture = "fake";
    info_.context_length = kContext;
    info_.vocab = 1000;
    info_.embedding_length = kLength;
    info_.embedding_min_dimensions = kMinDimensions;
  }
  const ModelInfo& info() const override { return info_; }
  const gufo_npu::Tokenizer& tokenizer() const override {
    throw std::logic_error("FakeEmbedder has no tokenizer");
  }
  GenerateResult Generate(const GenerateRequest&,
                          const std::function<bool(int)>&) override {
    throw std::logic_error("FakeEmbedder does not generate");
  }
  std::vector<int> EmbeddingIds(const EmbeddingInput& input,
                                std::string_view instruction) const override {
    {
      std::lock_guard lock(mutex_);
      instruction_ = std::string(instruction);
    }
    std::vector<int> ids = {1, 2};
    if (input.tokens) {
      ids.insert(ids.end(), input.tokens->begin(), input.tokens->end());
    } else {
      for (const unsigned char c : input.text) {
        ids.push_back(c);
      }
    }
    return ids;
  }
  std::vector<std::vector<float>> Embed(
      const std::vector<std::vector<int>>& inputs) override {
    std::vector<std::vector<float>> out;
    for (const auto& ids : inputs) {
      out.push_back(Vector(static_cast<int>(ids.size())));
    }
    return out;
  }
  static std::vector<float> Vector(int n) {
    std::vector<float> v(kLength);
    double norm = 0;
    for (int k = 0; k < kLength; ++k) {
      v[k] = static_cast<float>(n + k + 1);
      norm += static_cast<double>(v[k]) * v[k];
    }
    for (float& x : v) {
      x = static_cast<float>(x / std::sqrt(norm));
    }
    return v;
  }
  std::string instruction() const {
    std::lock_guard lock(mutex_);
    return instruction_;
  }

private:
  ModelInfo info_;
  mutable std::mutex mutex_;
  mutable std::string instruction_;
};

struct Running {
  Running(Engine& engine, const FakeFormat& format)
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
  httplib::Result Post(const std::string& path, const json& body) const {
    httplib::Client c("127.0.0.1", port);
    c.set_read_timeout(30, 0);
    return c.Post(path, body.dump(), "application/json");
  }

  Server server;
  int port = 0;
  std::thread thread;
};

std::vector<float> Floats(const json& embedding) {
  return embedding.get<std::vector<float>>();
}

std::vector<float> FromBase64(const std::string& text) {
  static const std::string kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<unsigned char> bytes;
  unsigned buffer = 0;
  int bits = 0;
  for (const char c : text) {
    if (c == '=') {
      break;
    }
    buffer = (buffer << 6) | static_cast<unsigned>(kAlphabet.find(c));
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes.push_back(static_cast<unsigned char>((buffer >> bits) & 0xFF));
    }
  }
  std::vector<float> out(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
  return out;
}

bool Near(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > 1e-6f) {
      return false;
    }
  }
  return true;
}

std::string ErrorCode(const httplib::Result& r) {
  if (!r) {
    return "no response";
  }
  const auto body = json::parse(r->body, nullptr, false);
  if (!body.is_object() || !body.contains("error")) {
    return "no error body";
  }
  const auto& error = body["error"];
  const auto text = [&](const char* key) {
    return error.contains(key) && error[key].is_string()
               ? error[key].get<std::string>()
               : std::string();
  };
  return text("param") + "/" + text("code");
}

void TestShapes(const Running& server) {
  // One string: "hi" is 2 template tokens + 2 bytes.
  auto r = server.Post("/v1/embeddings", {{"input", "hi"}});
  Check(r && r->status == 200, "string input: 200");
  if (r && r->status == 200) {
    const auto body = json::parse(r->body);
    Check(body["object"] == "list", "object is list");
    Check(body["model"] == "fake-embed", "model id");
    Check(body["data"].size() == 1, "one embedding");
    Check(body["data"][0]["object"] == "embedding", "data object");
    Check(body["data"][0]["index"] == 0, "index 0");
    Check(Near(Floats(body["data"][0]["embedding"]), FakeEmbedder::Vector(4)),
          "vector of the formatted ids");
    Check(body["usage"]["prompt_tokens"] == 4 &&
              body["usage"]["total_tokens"] == 4,
          "usage counts the formatted tokens");
  }
  // Several strings keep their order.
  r = server.Post("/v1/embeddings", {{"input", json::array({"a", "bbb"})}});
  Check(r && r->status == 200, "array of strings: 200");
  if (r && r->status == 200) {
    const auto body = json::parse(r->body);
    Check(body["data"].size() == 2 && body["data"][1]["index"] == 1,
          "two embeddings in order");
    Check(
        Near(Floats(body["data"][0]["embedding"]), FakeEmbedder::Vector(3)) &&
            Near(Floats(body["data"][1]["embedding"]), FakeEmbedder::Vector(5)),
        "each input's own vector");
    Check(body["usage"]["prompt_tokens"] == 8, "usage sums the inputs");
  }
  // Token arrays: one, and several.
  r = server.Post("/v1/embeddings", {{"input", json::array({5, 6, 7})}});
  Check(r && r->status == 200 &&
            json::parse(r->body)["usage"]["prompt_tokens"] == 5,
        "token array: one input of 3 + 2 tokens");
  r = server.Post(
      "/v1/embeddings",
      {{"input", json::array({json::array({5}), json::array({6, 7})})}});
  Check(r && r->status == 200 && json::parse(r->body)["data"].size() == 2,
        "array of token arrays: two inputs");
}

void TestOptions(const Running& server, const FakeEmbedder& engine) {
  // dimensions: the first n values, back to unit length.
  auto r = server.Post("/v1/embeddings", {{"input", "hi"}, {"dimensions", 4}});
  Check(r && r->status == 200, "dimensions 4: 200");
  if (r && r->status == 200) {
    const auto v = Floats(json::parse(r->body)["data"][0]["embedding"]);
    double norm = 0;
    for (const float x : v) {
      norm += static_cast<double>(x) * x;
    }
    Check(v.size() == 4 && std::fabs(norm - 1.0) < 1e-5, "cut and renormed");
  }
  r = server.Post("/v1/embeddings", {{"input", "hi"}, {"dimensions", 3}});
  Check(r && r->status == 400 && ErrorCode(r).rfind("dimensions", 0) == 0,
        "dimensions below the model's minimum: 400");
  r = server.Post("/v1/embeddings",
                  {{"input", "hi"}, {"dimensions", kLength + 1}});
  Check(r && r->status == 400, "dimensions above the length: 400");
  // base64: the same floats, little-endian.
  r = server.Post("/v1/embeddings",
                  {{"input", "hi"}, {"encoding_format", "base64"}});
  Check(r && r->status == 200, "base64: 200");
  if (r && r->status == 200) {
    const auto& e = json::parse(r->body)["data"][0]["embedding"];
    Check(e.is_string() &&
              Near(FromBase64(e.get<std::string>()), FakeEmbedder::Vector(4)),
          "base64 decodes to the float vector");
  }
  r = server.Post("/v1/embeddings",
                  {{"input", "hi"}, {"encoding_format", "int8"}});
  Check(r && r->status == 400, "unknown encoding_format: 400");
  // The instruction reaches the engine.
  r = server.Post("/v1/embeddings",
                  {{"input", "q"}, {"instruction", "Find the receipt"}});
  Check(r && r->status == 200 && engine.instruction() == "Find the receipt",
        "instruction passed to EmbeddingIds");
  r = server.Post("/v1/embeddings", {{"input", "q"}});
  Check(engine.instruction().empty(), "no instruction: empty");
}

void TestErrors(const Running& server) {
  Check(server.Post("/v1/embeddings", json::object())->status == 400,
        "missing input: 400");
  Check(server.Post("/v1/embeddings", {{"input", ""}})->status == 400,
        "empty string: 400");
  Check(
      server.Post("/v1/embeddings", {{"input", json::array()}})->status == 400,
      "empty array: 400");
  Check(server.Post("/v1/embeddings", {{"input", json::array({1, "a"})}})
                ->status == 400,
        "mixed array: 400");
  Check(
      server.Post("/v1/embeddings", {{"input", json::array({5000})}})->status ==
          400,
      "token id outside the vocabulary: 400");
  const auto r =
      server.Post("/v1/embeddings", {{"input", std::string(kContext, 'x')}});
  Check(
      r && r->status == 400 && ErrorCode(r) == "input/context_length_exceeded",
      "input over the context: 400 context_length_exceeded");
  // An embedding model does not chat.
  const auto chat = server.Post(
      "/v1/chat/completions",
      {{"messages", json::array({json{{"role", "user"}, {"content", "hi"}}})}});
  Check(chat && chat->status == 400 && ErrorCode(chat).rfind("model", 0) == 0,
        "chat on an embedding model: 400 for model");
  const auto completion =
      server.Post("/v1/completions", {{"prompt", "hi"}, {"max_tokens", 2}});
  Check(completion && completion->status == 400,
        "completion on an embedding model: 400");
}

}  // namespace

int main() {
  FakeFormat format;
  {
    FakeEmbedder engine;
    const Running server(engine, format);
    TestShapes(server);
    TestOptions(server, engine);
    TestErrors(server);
  }
  {
    // A chat model refuses embeddings.
    FakeEngine chat;
    const Running server(chat, format);
    const auto r = server.Post("/v1/embeddings", {{"input", "hi"}});
    Check(r && r->status == 400 && ErrorCode(r).rfind("model", 0) == 0,
          "embeddings on a chat model: 400 for model");
  }
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::puts("PASS: /v1/embeddings");
  return 0;
}
