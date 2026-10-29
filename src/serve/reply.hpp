#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "serve/chat_format.hpp"
#include "serve/engine.hpp"
#include "serve/text.hpp"

// One reply as it is generated: token ids to whole UTF-8 characters, stop
// strings cut out, then the chat format's parser splits the text into
// content, reasoning and tool calls, handed to `emit` as they complete.
namespace gufo_npu::serve {

class ReplyStream {
public:
  // `emit` returns false when the parts cannot be delivered (the client
  // left); the stream then ends the generation.
  using Emit = std::function<bool(std::vector<OutputPart>&&)>;

  ReplyStream(const Engine& engine, std::vector<int> stop_ids,
              std::vector<std::string> stop_strings,
              std::unique_ptr<OutputParser> parser, Emit emit);

  // Engine::Generate's on_token: false ends the generation.
  bool OnToken(int id);
  // After the generation: hands over what was held back.
  void Finish();

  // Ends the generation after `n` tool calls (parallel_tool_calls: false
  // is 1); anything after the last is dropped. 0: no limit.
  void set_max_tool_calls(int n) { max_tool_calls_ = n; }

  bool stop_string_hit() const { return stops_.stopped(); }
  bool tool_limit_hit() const { return tool_limit_hit_; }
  bool delivery_failed() const { return failed_; }
  int tool_calls() const { return tool_calls_; }

private:
  void Send(std::vector<OutputPart>&& parts);

  const Engine& engine_;
  std::vector<int> stop_ids_;
  Utf8Assembler utf8_;
  StopMatcher stops_;
  std::unique_ptr<OutputParser> parser_;
  Emit emit_;
  bool failed_ = false;
  int tool_calls_ = 0;
  int max_tool_calls_ = 0;
  bool tool_limit_hit_ = false;
};

// Everything is content: the parser for /v1/completions' raw text.
class PlainTextParser : public OutputParser {
public:
  std::vector<OutputPart> Feed(const std::string& text) override;
  std::vector<OutputPart> Finish() override { return {}; }
};

}  // namespace gufo_npu::serve
