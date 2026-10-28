#pragma once

#include <string>
#include <string_view>
#include <vector>

// Streaming text helpers for the server: whole UTF-8 characters out of
// token bytes, and stop strings found across token boundaries.
namespace gufo_npu::serve {

// Token bytes in, valid UTF-8 out. A character split across tokens is held
// until its last byte arrives; bytes that cannot form a character become
// U+FFFD, so the output is always valid UTF-8 (JSON needs it).
class Utf8Assembler {
public:
  std::string Feed(std::string_view bytes);
  // End of the stream: an unfinished character becomes U+FFFD.
  std::string Finish();

private:
  std::string pending_;  // the start of an unfinished character
};

// Finds the first of `stops` in text fed piece by piece. Feed returns the
// text that can no longer be part of a stop string; a tail that could still
// grow into one is held back, so a stop string never reaches the client.
class StopMatcher {
public:
  explicit StopMatcher(std::vector<std::string> stops);

  // Text before any stop string. After a stop is found, stopped() is true
  // and later calls return nothing.
  std::string Feed(std::string_view text);
  // End of the stream: what was held back (empty after a stop).
  std::string Finish();
  bool stopped() const { return stopped_; }

private:
  std::vector<std::string> stops_;
  std::string held_;
  bool stopped_ = false;
};

}  // namespace gufo_npu::serve
