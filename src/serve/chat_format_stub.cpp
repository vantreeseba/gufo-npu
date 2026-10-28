// TEMPORARY (remove at integration): NewQwen35ChatFormat until the chat
// part (src/serve/chat_format.cpp) lands, so `gufo-npu serve` links on this
// branch.
#include <stdexcept>

#include "serve/chat_format.hpp"

namespace gufo_npu::serve {

std::unique_ptr<ChatFormat> NewQwen35ChatFormat() {
  throw std::runtime_error(
      "no chat format in this build (chat_format_stub.cpp "
      "placeholder)");
}

}  // namespace gufo_npu::serve
