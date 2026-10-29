#include "serve/reply.hpp"

#include <algorithm>
#include <utility>

namespace gufo_npu::serve {

ReplyStream::ReplyStream(const Engine& engine, std::vector<int> stop_ids,
                         std::vector<std::string> stop_strings,
                         std::unique_ptr<OutputParser> parser, Emit emit)
    : engine_(engine),
      stop_ids_(std::move(stop_ids)),
      stops_(std::move(stop_strings)),
      parser_(std::move(parser)),
      emit_(std::move(emit)) {}

bool ReplyStream::OnToken(int id) {
  if (failed_ || stops_.stopped() || tool_limit_hit_) {
    return false;
  }
  if (std::find(stop_ids_.begin(), stop_ids_.end(), id) != stop_ids_.end()) {
    return true;  // the engine ends the generation on its stop ids
  }
  const std::string text = stops_.Feed(utf8_.Feed(engine_.TokenBytes(id)));
  if (!text.empty()) {
    Send(parser_->Feed(text));
  }
  return !failed_ && !stops_.stopped() && !tool_limit_hit_;
}

void ReplyStream::Finish() {
  if (failed_ || tool_limit_hit_) {
    return;
  }
  std::string text;
  if (!stops_.stopped()) {
    text = stops_.Feed(utf8_.Finish());
    text += stops_.Finish();
  }
  auto parts = text.empty() ? std::vector<OutputPart>{} : parser_->Feed(text);
  auto rest = parser_->Finish();
  parts.insert(parts.end(), std::make_move_iterator(rest.begin()),
               std::make_move_iterator(rest.end()));
  Send(std::move(parts));
}

void ReplyStream::Send(std::vector<OutputPart>&& parts) {
  if (parts.empty() || failed_) {
    return;
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (parts[i].kind != OutputPart::Kind::kToolCall) {
      continue;
    }
    ++tool_calls_;
    if (max_tool_calls_ > 0 && tool_calls_ >= max_tool_calls_) {
      tool_limit_hit_ = true;
      parts.resize(i + 1);
      break;
    }
  }
  if (!emit_(std::move(parts))) {
    failed_ = true;
  }
}

std::vector<OutputPart> PlainTextParser::Feed(const std::string& text) {
  if (text.empty()) {
    return {};
  }
  OutputPart part;
  part.text = text;
  return {std::move(part)};
}

}  // namespace gufo_npu::serve
