#include "models/qwen35/chat.hpp"

#include <stdexcept>

namespace gufo_npu::qwen35 {
namespace {

std::string Trim(const std::string& s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

}  // namespace

std::string RenderChat(const std::vector<Message>& messages, bool thinking) {
  if (messages.empty()) {
    throw std::invalid_argument("chat: no messages");
  }
  std::string out;
  for (const auto& message : messages) {
    if (message.role != "system" && message.role != "user" &&
        message.role != "assistant") {
      throw std::invalid_argument("chat: unsupported role " + message.role);
    }
    // The template trims every message's content.
    out +=
        kImStart + message.role + "\n" + Trim(message.content) + kImEnd + "\n";
  }
  out += std::string(kImStart) + "assistant\n";
  out += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  return out;
}

}  // namespace gufo_npu::qwen35
