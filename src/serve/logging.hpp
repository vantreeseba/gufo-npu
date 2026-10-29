#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

// Server logs in gufo's format (docs/SERVING.md, "Logs"): one line per event
// on stderr,
//   YYYY-MM-DD HH:MM:SS [LEVEL] [component] key=value ...
// coloured when stderr is a terminal and NO_COLOR is unset. Request and
// generation lines use the same keys as gufo's, so the two servers' logs
// read alike. Prompt and reply text are never logged.
namespace gufo_npu::serve {

// Ordered from chattiest to quietest: a line is written when its level is at
// or above the threshold.
enum class LogLevel : std::uint8_t { kDebug, kInfo, kWarn, kError };

// "debug", "info", "warn" or "error"; nullopt otherwise.
std::optional<LogLevel> LogLevelFromName(std::string_view name);
std::string_view LogLevelName(LogLevel level);

class Logger {
public:
  // The process-wide threshold (default kInfo). Set once at startup.
  static void SetLevel(LogLevel level);
  static LogLevel Level();
  // False when a line at `level` would be dropped: check before building an
  // expensive message.
  static bool Enabled(LogLevel level);

  static void Log(LogLevel level, std::string_view component,
                  std::string_view message);

  // A finished request: `request=ID event=completed method= path= status=
  // duration_ms= outcome=` then `details` and the memory status. 5xx and a
  // failed stream log at ERROR, 4xx and a client that left at WARN, the rest
  // at `success_level` (kDebug for polled endpoints such as /health).
  static void LogRequest(std::string_view id, std::string_view method,
                         std::string_view path, int status_code,
                         double duration_ms, std::string_view details = "",
                         std::string_view outcome = "completed",
                         LogLevel success_level = LogLevel::kInfo);

  // `rss_mib=N host_available_mib=N` (process RSS, host MemAvailable).
  static std::string MemoryStatus();

  static void Debug(std::string_view component, std::string_view message) {
    Log(LogLevel::kDebug, component, message);
  }
  static void Info(std::string_view component, std::string_view message) {
    Log(LogLevel::kInfo, component, message);
  }
  static void Warn(std::string_view component, std::string_view message) {
    Log(LogLevel::kWarn, component, message);
  }
  static void Error(std::string_view component, std::string_view message) {
    Log(LogLevel::kError, component, message);
  }
};

}  // namespace gufo_npu::serve
