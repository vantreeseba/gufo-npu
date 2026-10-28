#pragma once

// A scripted engine and a small chat format for the server tests: no model,
// no GGUF, no NPU.
#include <atomic>
#include <chrono>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "serve/chat_format.hpp"
#include "serve/engine.hpp"

namespace gufo_npu::serve::testing {

inline constexpr int kStopId = 1000000;
inline constexpr char kStopToken[] = "<|im_end|>";

// Replies with `pieces` (one token each), then the stop token unless
// `stop_at_end` is false. Each token takes `token_delay`.
class FakeEngine : public Engine {
public:
  FakeEngine() {
    info_.id = "fake-model";
    info_.architecture = "fake";
    info_.context_length = 4096;
    info_.vocab = 100;
  }

  const ModelInfo& info() const override { return info_; }
  const Tokenizer& tokenizer() const override {
    throw std::logic_error("FakeEngine has no tokenizer");
  }
  std::vector<int> Encode(std::string_view text) const override {
    return std::vector<int>(text.size(), 0);  // one token per byte
  }
  std::string TokenBytes(int id) const override {
    if (id == kStopId) {
      return kStopToken;
    }
    std::lock_guard lock(mutex_);
    return pieces_.at(static_cast<std::size_t>(id));
  }
  int FindToken(std::string_view token) const override {
    return token == kStopToken ? kStopId : -1;
  }

  void Script(std::vector<std::string> pieces, bool stop_at_end = true,
              std::chrono::milliseconds token_delay = {}) {
    std::lock_guard lock(mutex_);
    pieces_ = std::move(pieces);
    stop_at_end_ = stop_at_end;
    delay_ = token_delay;
  }

  GenerateResult Generate(const GenerateRequest& request,
                          const std::function<bool(int)>& on_token) override {
    if (running_.exchange(true)) {
      throw std::logic_error("FakeEngine: two generations at once");
    }
    std::vector<int> ids;
    std::chrono::milliseconds delay;
    {
      std::lock_guard lock(mutex_);
      last_ = request;
      last_.cancel = nullptr;
      for (std::size_t i = 0; i < pieces_.size(); ++i) {
        ids.push_back(static_cast<int>(i));
      }
      if (stop_at_end_) {
        ids.push_back(kStopId);
      }
      delay = delay_;
    }
    ++generations_;
    GenerateResult result;
    result.prompt_tokens = static_cast<int>(request.prompt.size());
    result.cached_prompt_tokens = 3;
    result.finish = FinishReason::kLength;
    for (const int id : ids) {
      if (static_cast<int>(result.tokens.size()) >= request.max_tokens) {
        break;
      }
      if (request.cancel != nullptr && request.cancel->load()) {
        result.finish = FinishReason::kCancelled;
        break;
      }
      if (delay.count() > 0) {
        std::this_thread::sleep_for(delay);
      }
      result.tokens.push_back(id);
      if (!on_token(id)) {
        result.finish = FinishReason::kCancelled;
        break;
      }
      if (id == kStopId) {
        result.finish = FinishReason::kStop;
        break;
      }
    }
    if (result.finish == FinishReason::kLength &&
        static_cast<int>(result.tokens.size()) < request.max_tokens &&
        !stop_at_end_) {
      result.finish = FinishReason::kStop;  // script ended
    }
    {
      std::lock_guard lock(mutex_);
      last_result_ = result;
    }
    running_ = false;
    return result;
  }

  GenerateRequest last() const {
    std::lock_guard lock(mutex_);
    return last_;
  }
  GenerateResult last_result() const {
    std::lock_guard lock(mutex_);
    return last_result_;
  }
  bool running() const { return running_; }
  int generations() const { return generations_; }

  ModelInfo info_;

private:
  mutable std::mutex mutex_;
  std::vector<std::string> pieces_;
  bool stop_at_end_ = true;
  std::chrono::milliseconds delay_{};
  GenerateRequest last_;
  GenerateResult last_result_;
  std::atomic<bool> running_{false};
  std::atomic<int> generations_{0};
};

// <think>…</think> is reasoning and <tool_call>{"name", "arguments"}
// </tool_call> a tool call; the rest is content. Holds back a tail that
// may be the start of a tag.
class FakeParser : public OutputParser {
public:
  std::vector<OutputPart> Feed(const std::string& text) override {
    buf_ += text;
    std::vector<OutputPart> out;
    for (;;) {
      if (state_ == State::kTool) {
        const auto end = buf_.find("</tool_call>");
        if (end == std::string::npos) {
          return out;
        }
        const auto call = nlohmann::json::parse(buf_.substr(0, end));
        OutputPart part;
        part.kind = OutputPart::Kind::kToolCall;
        part.tool_name = call.at("name").get<std::string>();
        part.arguments = call.at("arguments").dump();
        out.push_back(part);
        buf_.erase(0, end + 12);
        state_ = State::kContent;
        continue;
      }
      const bool think = state_ == State::kThink;
      const std::vector<std::string> tags =
          think ? std::vector<std::string>{"</think>"}
                : std::vector<std::string>{"<think>", "<tool_call>"};
      std::size_t at = std::string::npos;
      std::string tag;
      for (const auto& t : tags) {
        const auto p = buf_.find(t);
        if (p < at) {
          at = p;
          tag = t;
        }
      }
      if (at == std::string::npos) {
        std::size_t hold = 0;
        for (const auto& t : tags) {
          for (std::size_t n = std::min(t.size() - 1, buf_.size()); n > hold;
               --n) {
            if (buf_.compare(buf_.size() - n, n, t, 0, n) == 0) {
              hold = n;
              break;
            }
          }
        }
        Emit(out, buf_.substr(0, buf_.size() - hold));
        buf_.erase(0, buf_.size() - hold);
        return out;
      }
      Emit(out, buf_.substr(0, at));
      buf_.erase(0, at + tag.size());
      state_ = tag == "<think>"       ? State::kThink
               : tag == "<tool_call>" ? State::kTool
                                      : State::kContent;
    }
  }

  std::vector<OutputPart> Finish() override {
    std::vector<OutputPart> out;
    if (state_ != State::kTool) {
      Emit(out, buf_);
    }
    buf_.clear();
    return out;
  }

private:
  enum class State { kContent, kThink, kTool };

  void Emit(std::vector<OutputPart>& out, const std::string& text) const {
    if (text.empty()) {
      return;
    }
    OutputPart part;
    part.kind = state_ == State::kThink ? OutputPart::Kind::kReasoning
                                        : OutputPart::Kind::kContent;
    part.text = text;
    out.push_back(part);
  }

  std::string buf_;
  State state_ = State::kContent;
};

// Renders each message's content on its own line; a "bad" role is refused.
class FakeFormat : public ChatFormat {
public:
  std::string Render(const ChatPrompt& prompt) const override {
    std::string text;
    for (const auto& m : prompt.messages) {
      if (m.value("role", "") == "bad") {
        throw std::invalid_argument("unknown role: bad");
      }
      text += m.value("content", "") + "\n";
    }
    std::lock_guard lock(mutex_);
    last_ = prompt;
    return text;
  }
  std::unique_ptr<OutputParser> NewParser(const ChatPrompt&) const override {
    return std::make_unique<FakeParser>();
  }
  std::vector<std::string> StopTokens() const override { return {kStopToken}; }

  ChatPrompt last() const {
    std::lock_guard lock(mutex_);
    return last_;
  }

private:
  mutable std::mutex mutex_;
  mutable ChatPrompt last_;
};

}  // namespace gufo_npu::serve::testing
