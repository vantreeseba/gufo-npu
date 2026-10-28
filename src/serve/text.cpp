#include "serve/text.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace gufo_npu::serve {
namespace {

constexpr char kReplacement[] = "\xEF\xBF\xBD";  // U+FFFD

// Bytes in the character a lead byte starts, or 0 if it cannot start one.
int SequenceLength(unsigned char lead) {
  if (lead < 0x80) {
    return 1;
  }
  if (lead >= 0xC2 && lead <= 0xDF) {
    return 2;
  }
  if (lead >= 0xE0 && lead <= 0xEF) {
    return 3;
  }
  if (lead >= 0xF0 && lead <= 0xF4) {
    return 4;
  }
  return 0;
}

// Whether `byte` may follow `lead` at offset `i` (1-based) of a character:
// continuation bytes, without overlong forms, surrogates or values past
// U+10FFFF.
bool ValidContinuation(unsigned char lead, int i, unsigned char byte) {
  if (i == 1) {
    switch (lead) {
      case 0xE0:
        return byte >= 0xA0 && byte <= 0xBF;
      case 0xED:
        return byte >= 0x80 && byte <= 0x9F;
      case 0xF0:
        return byte >= 0x90 && byte <= 0xBF;
      case 0xF4:
        return byte >= 0x80 && byte <= 0x8F;
      default:
        break;
    }
  }
  return byte >= 0x80 && byte <= 0xBF;
}

}  // namespace

std::string Utf8Assembler::Feed(std::string_view bytes) {
  std::string in = std::move(pending_);
  pending_.clear();
  in.append(bytes);
  std::string out;
  out.reserve(in.size());
  std::size_t i = 0;
  while (i < in.size()) {
    const auto lead = static_cast<unsigned char>(in[i]);
    const int n = SequenceLength(lead);
    if (n == 0) {
      out += kReplacement;
      ++i;
      continue;
    }
    int good = 1;  // bytes of the character that are valid so far
    while (good < n && i + good < in.size() &&
           ValidContinuation(lead, good,
                             static_cast<unsigned char>(in[i + good]))) {
      ++good;
    }
    if (good == n) {
      out.append(in, i, n);
      i += n;
    } else if (i + good == in.size()) {
      pending_.assign(in, i, good);  // may still be completed
      break;
    } else {
      out += kReplacement;  // the bytes so far, then a byte that breaks it
      i += good;
    }
  }
  return out;
}

std::string Utf8Assembler::Finish() {
  std::string out = pending_.empty() ? "" : kReplacement;
  pending_.clear();
  return out;
}

StopMatcher::StopMatcher(std::vector<std::string> stops) {
  for (auto& stop : stops) {
    if (!stop.empty()) {
      stops_.push_back(std::move(stop));
    }
  }
}

std::string StopMatcher::Feed(std::string_view text) {
  if (stopped_) {
    return {};
  }
  std::string buf = std::move(held_);
  held_.clear();
  buf.append(text);
  std::size_t first = std::string::npos;
  for (const auto& stop : stops_) {
    first = std::min(first, buf.find(stop));
  }
  if (first != std::string::npos) {
    stopped_ = true;
    buf.resize(first);
    return buf;
  }
  // Hold the longest tail that is the start of some stop string.
  std::size_t hold = 0;
  for (const auto& stop : stops_) {
    const std::size_t most = std::min(stop.size() - 1, buf.size());
    for (std::size_t len = most; len > hold; --len) {
      if (buf.compare(buf.size() - len, len, stop, 0, len) == 0) {
        hold = len;
        break;
      }
    }
  }
  held_.assign(buf, buf.size() - hold, hold);
  buf.resize(buf.size() - hold);
  return buf;
}

std::string StopMatcher::Finish() {
  std::string out = std::move(held_);
  held_.clear();
  return out;
}

}  // namespace gufo_npu::serve
